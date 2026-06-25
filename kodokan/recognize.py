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
    last = kp[int(2 * len(kp) / 3) :]
    hip_y = np.nanmean(last[:, :, [11, 12], 1], axis=(0, 2))  # (P,)
    if np.all(np.isnan(hip_y)):
        present = ~np.all(np.isnan(kp[..., 0]), axis=2)
        return int(np.argmax(present.mean(0)))
    return int(np.nanargmin(hip_y))


def demo_feature(
    pose_seq: PoseSequence, start_s: float, end_s: float, *, mode: str
) -> np.ndarray:
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
        feat = np.stack(
            [
                np.concatenate([_angles(kp[f, p]), _norm_positions(kp[f, p])])
                for f in range(F)
            ]
        )
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
            seg = feat[a : max(b, a + 1)]
            parts.append(seg.mean(0))
            parts.append(seg.std(0))
    return np.concatenate(parts)


# ---- leave-one-out evaluation (LEAKAGE-FREE: all preprocessing fit on train fold) ----
#
# NOTE on a deeper confound: in the Kodokan dataset each technique class == one source
# video, so leave-one-DEMO-out trains and tests on reps from the SAME clip. The model
# can exploit clip-identity cues (people/gi/camera/background), so these numbers are an
# UPPER BOUND on true technique recognition. Honest validation needs >=2 independent
# source clips per technique and leave-one-CLIP-out (group CV by video_id). See
# misc/docs/adversarial-review.md.


def dtw_1nn_predict(feature_seqs: list[np.ndarray], labels) -> list:
    """Leave-one-out 1-NN by DTW distance; returns predicted label per item."""
    n = len(feature_seqs)
    preds: list = [None] * n
    for i in range(n):
        best, best_j = np.inf, -1
        for j in range(n):
            if i == j:
                continue
            d = compare(feature_seqs[i], feature_seqs[j])["normalized"]
            if np.isfinite(d) and d < best:
                best, best_j = d, j
        preds[i] = labels[best_j] if best_j >= 0 else None
    return preds


def loo_pooled_predict(
    X: np.ndarray, y: np.ndarray, *, method: str = "lda_knn", k: int = 3, groups=None
) -> np.ndarray:
    """Leave-one-out (or leave-one-GROUP-out) predictions; preprocessing fit on train only.

    If ``groups`` is given (e.g. source ``video_id`` per sample), each fold excludes ALL
    samples sharing the held-out sample's group — i.e. **leave-one-clip-out**, which removes
    within-clip leakage. Otherwise it is plain leave-one-sample-out.
    """
    lda_cls = None
    if method == "lda_knn":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda_cls
    n = len(X)
    idx = np.arange(n)
    groups = np.asarray(groups) if groups is not None else None
    preds = np.empty(n, dtype=y.dtype)
    for i in range(n):
        tr = (groups != groups[i]) if groups is not None else (idx != i)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9  # fit on train fold only
        Xtr, xi, ytr = (X[tr] - mu) / sd, (X[i] - mu) / sd, y[tr]
        if method == "centroid":
            classes = np.unique(ytr)
            cents = np.stack([Xtr[ytr == c].mean(0) for c in classes])
            preds[i] = classes[int(np.argmin(np.linalg.norm(cents - xi, axis=1)))]
        elif method in ("knn", "lda_knn"):
            if method == "lda_knn":
                try:
                    m = lda_cls()
                    Xtr, xi = m.fit(Xtr, ytr).transform(Xtr), m.transform(xi[None])[0]
                except Exception:
                    pass  # singular fold: fall back to standardized kNN
            d = np.linalg.norm(Xtr - xi, axis=1)
            nn = np.argsort(d)[:k]
            vals, cnts = np.unique(ytr[nn], return_counts=True)
            preds[i] = vals[int(np.argmax(cnts))]
        else:
            raise ValueError(f"unknown method {method!r}")
    return preds


def classification_metrics(y_true, y_pred) -> dict:
    """Top-1, balanced (macro-recall), majority-class baseline, n_classes, n."""
    y_true = np.asarray(y_true)
    yp = np.array([p if p is not None else -1 for p in y_pred])
    classes = np.unique(y_true)
    recalls = [float((yp[y_true == c] == c).mean()) for c in classes]
    _, cnts = np.unique(y_true, return_counts=True)
    return {
        "top1": round(float((yp == y_true).mean()), 3),
        "balanced": round(float(np.mean(recalls)), 3),
        "majority_baseline": round(float(cnts.max() / len(y_true)), 3),
        "n_classes": int(len(classes)),
        "n": int(len(y_true)),
    }


# Thin backward-compatible accuracy wrappers (top-1).
def dtw_1nn_accuracy(feature_seqs, labels) -> float:
    return classification_metrics(labels, dtw_1nn_predict(feature_seqs, labels))["top1"]


def pool_centroid_accuracy(X, y) -> float:
    return classification_metrics(y, loo_pooled_predict(X, y, method="centroid"))[
        "top1"
    ]


def pool_knn_accuracy(X, y, k: int = 3) -> float:
    return classification_metrics(y, loo_pooled_predict(X, y, method="knn", k=k))[
        "top1"
    ]


def pool_lda_knn_accuracy(X, y, k: int = 3) -> float | None:
    try:
        import sklearn.discriminant_analysis  # noqa: F401
    except Exception:
        return None
    return classification_metrics(y, loo_pooled_predict(X, y, method="lda_knn", k=k))[
        "top1"
    ]
