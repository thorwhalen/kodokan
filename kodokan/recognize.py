"""Learned-vs-DTW technique recognition experiments (data-appropriate, no GPU).

The feature bake-off showed DTW on hand-crafted angles can't tell techniques apart.
This module tests whether the bottleneck is the *classifier*, the *features*, or
tori/uke *role inconsistency*, with leave-one-out CV on the small dataset:

- feature modes: ``primary_angles`` (baseline), ``tori_angles`` (role-consistent —
  tori picked as the person left standing), ``tori_angles_pos``.
- methods: ``dtw_1nn`` (DTW nearest-neighbour), ``pool_centroid`` / ``pool_knn``
  (temporal-pyramid pooled descriptor + nearest-centroid / kNN), and ``pool_lda_knn``
  (LDA-reduced + kNN, if scikit-learn is available).

Everything is numpy (sklearn optional), appropriate for ~80 samples where deep nets
would overfit; the same harness scales to more techniques once more data is collected.
"""

from __future__ import annotations

import numpy as np

from kodokan.compare import _clean, compare
from kodokan.descriptors import _angles, _norm_positions, _window
from kodokan.pose import PoseSequence


def tori_index(pose_seq: PoseSequence, start_s: float, end_s: float) -> int:
    """Heuristic tori (thrower) = the person whose hips stay highest at the finish.

    In image coordinates y grows downward, so uke (thrown to the ground) has larger
    hip-y over the last third of the demo; tori (still standing) has smaller hip-y.
    Falls back to the most-present person if undecidable.
    """
    kp = _window(pose_seq, start_s, end_s)  # (F, P, 17, 3)
    if len(kp) == 0:
        return 0
    last = kp[int(2 * len(kp) / 3):]
    hip_y = np.nanmean(last[:, :, [11, 12], 1], axis=(0, 2))  # (P,)
    if np.all(np.isnan(hip_y)):
        present = ~np.all(np.isnan(kp[..., 0]), axis=2)
        return int(np.argmax(present.mean(0)))
    return int(np.nanargmin(hip_y))


def demo_feature(pose_seq: PoseSequence, start_s: float, end_s: float, *, mode: str) -> np.ndarray:
    """Per-frame features for one demo under a role/feature mode."""
    kp = _window(pose_seq, start_s, end_s)
    F = len(kp)
    if F == 0:
        return np.empty((0, 1))
    if mode == "primary_angles":
        present = ~np.all(np.isnan(kp[..., 0]), axis=2)
        p = int(np.argmax(present.mean(0)))
    elif mode in ("tori_angles", "tori_angles_pos"):
        p = tori_index(pose_seq, start_s, end_s)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "tori_angles_pos":
        feat = np.stack([np.concatenate([_angles(kp[f, p]), _norm_positions(kp[f, p])]) for f in range(F)])
    else:
        feat = np.stack([_angles(kp[f, p]) for f in range(F)])
    return _clean(feat)


def pooled_descriptor(feat: np.ndarray, *, levels=(1, 2, 4)) -> np.ndarray:
    """Temporal-pyramid pooled fixed-length descriptor (mean+std per segment)."""
    if len(feat) == 0:
        return np.zeros(1)
    parts = []
    for lv in levels:
        idx = np.linspace(0, len(feat), lv + 1).astype(int)
        for a, b in zip(idx[:-1], idx[1:]):
            seg = feat[a:max(b, a + 1)]
            parts.append(seg.mean(0))
            parts.append(seg.std(0))
    return np.concatenate(parts)


# ---- leave-one-out evaluators -> accuracy ----

def dtw_1nn_accuracy(feature_seqs: list[np.ndarray], labels: list[int]) -> float:
    n = len(feature_seqs)
    correct = 0
    for i in range(n):
        best, best_j = np.inf, -1
        for j in range(n):
            if i == j:
                continue
            d = compare(feature_seqs[i], feature_seqs[j])["normalized"]
            if np.isfinite(d) and d < best:
                best, best_j = d, j
        if best_j >= 0 and labels[best_j] == labels[i]:
            correct += 1
    return correct / n


def _standardize(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / (sd + 1e-9)


def pool_centroid_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    Xs = _standardize(X)
    classes = np.unique(y)
    correct = 0
    for i in range(len(Xs)):
        cents = {}
        for c in classes:
            mask = (y == c) & (np.arange(len(Xs)) != i)
            if mask.any():
                cents[c] = Xs[mask].mean(0)
        pred = min(cents, key=lambda c: np.linalg.norm(Xs[i] - cents[c]))
        correct += int(pred == y[i])
    return correct / len(Xs)


def pool_knn_accuracy(X: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    Xs = _standardize(X)
    n = len(Xs)
    correct = 0
    for i in range(n):
        d = np.linalg.norm(Xs - Xs[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        vals, cnts = np.unique(y[nn], return_counts=True)
        correct += int(vals[np.argmax(cnts)] == y[i])
    return correct / n


def pool_lda_knn_accuracy(X: np.ndarray, y: np.ndarray, k: int = 3) -> float | None:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception:
        return None
    Xs = _standardize(X)
    n = len(Xs)
    correct = 0
    for i in range(n):
        tr = np.arange(n) != i
        lda = LinearDiscriminantAnalysis()
        try:
            Z = lda.fit(Xs[tr], y[tr]).transform(Xs)
        except Exception:
            return None
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        vals, cnts = np.unique(y[nn], return_counts=True)
        correct += int(vals[np.argmax(cnts)] == y[i])
    return correct / n
