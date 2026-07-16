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

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from kodokan.compare import _clean, compare
from kodokan.descriptors import _angles, _norm_positions, _window
from kodokan.pose import PoseSequence


@dataclass
class ToriDecision:
    """Outcome of the tori/uke role heuristic, with a confidence margin.

    The "role-consistency is the dominant recognition lever" claim rests on this
    heuristic, so — per the adversarial review — every call is now instrumented.

    Attributes:
        index: Chosen tori slot (always a usable slot, even when abstaining).
        margin: Finish-frame hip-y separation between the two judoka in
            *torso-length* units (how much lower uke's hips sit than tori's).
            Larger = more confident; ``nan`` when the cue was unusable.
        fell_back: The hip-y cue was unusable, so the most-present slot was used.
        abstained: Low-confidence decision (``margin`` below the abstain
            threshold, or a fall-back) — the role label should not be trusted.
    """

    index: int
    margin: float
    fell_back: bool
    abstained: bool


def _window_torso_scale(kp: np.ndarray) -> float:
    """Robust body scale (median torso length) over a ``(F, P, 17, 3)`` window."""
    sho = (kp[:, :, 5, :2] + kp[:, :, 6, :2]) / 2
    hip = (kp[:, :, 11, :2] + kp[:, :, 12, :2]) / 2
    torso = np.linalg.norm(sho - hip, axis=-1)  # (F, P)
    scale = np.nanmedian(torso)
    return float(scale) if np.isfinite(scale) and scale > 1e-6 else 1.0


def tori_decision(
    pose_seq: PoseSequence,
    start_s: float,
    end_s: float,
    *,
    abstain_margin: float = 0.15,
) -> ToriDecision:
    """Tori (thrower) heuristic with a confidence margin and an abstain flag.

    Tori is the judoka whose hips stay highest (smallest image-y) over the last
    third of the demo — uke has been thrown to the ground. The margin is the two
    judoka's finish hip-y separation in torso-length units; below
    ``abstain_margin`` (or when the cue is unusable and we fall back to presence)
    the role assignment is unreliable and ``abstained`` is set. See
    :func:`tori_decision_stats` for aggregate fall-back/abstain rates.
    """
    kp = _window(pose_seq, start_s, end_s)  # (F, P, 17, 3)
    if len(kp) == 0:
        return ToriDecision(0, float("nan"), fell_back=True, abstained=True)
    last = kp[int(2 * len(kp) / 3) :]
    with warnings.catch_warnings():  # all-NaN slices are expected (handled below)
        warnings.simplefilter("ignore", RuntimeWarning)
        hip_y = np.nanmean(last[:, :, [11, 12], 1], axis=(0, 2))  # (P,)
    finite = np.isfinite(hip_y)
    if not finite.any():  # no finish-hip info at all -> presence fallback
        present = ~np.all(np.isnan(kp[..., 0]), axis=2)
        return ToriDecision(
            int(np.argmax(present.mean(0))),
            float("nan"),
            fell_back=True,
            abstained=True,
        )
    # lowest image-y = highest hips = tori. Slot choice matches the pre-instrumentation
    # tori_index exactly (nanargmin over available hips); only the margin is new.
    idx = int(np.nanargmin(hip_y))
    if finite.sum() < 2:  # only one judoka visible at finish -> margin unmeasurable
        return ToriDecision(idx, float("nan"), fell_back=False, abstained=True)
    margin = float((np.nanmax(hip_y) - np.nanmin(hip_y)) / _window_torso_scale(last))
    return ToriDecision(idx, margin, fell_back=False, abstained=margin < abstain_margin)


def tori_index(pose_seq: PoseSequence, start_s: float, end_s: float) -> int:
    """Heuristic tori slot (thin wrapper over :func:`tori_decision`).

    Backward-compatible: returns just the chosen slot. Use :func:`tori_decision`
    when you need the confidence margin / abstain flag.
    """
    return tori_decision(pose_seq, start_s, end_s).index


def tori_decision_stats(decisions: Iterable[ToriDecision]) -> dict:
    """Aggregate fall-back / abstain rates and median margin over decisions.

    The instrumentation the review asked for: run this over every demo's
    :func:`tori_decision` to see how often the "dominant lever" heuristic is
    actually confident vs. abstaining.
    """
    ds = list(decisions)
    n = len(ds)
    if n == 0:
        return {
            "n": 0,
            "fell_back_rate": 0.0,
            "abstain_rate": 0.0,
            "median_margin": float("nan"),
        }
    margins = [d.margin for d in ds if np.isfinite(d.margin)]
    return {
        "n": n,
        "fell_back_rate": round(sum(d.fell_back for d in ds) / n, 3),
        "abstain_rate": round(sum(d.abstained for d in ds) / n, 3),
        "median_margin": (
            round(float(np.median(margins)), 3) if margins else float("nan")
        ),
    }


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
        try:
            from sklearn.discriminant_analysis import (
                LinearDiscriminantAnalysis as lda_cls,
            )
        except Exception:
            lda_cls = None  # sklearn optional: per-fold code falls back to kNN
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


def _bootstrap_ci(y_true, yp, classes, *, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CIs for (top-1, balanced) accuracy over resampled items."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    t1, bal = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yb = y_true[idx], yp[idx]
        t1[b] = float((yt == yb).mean())
        recs = [(yb[yt == c] == c).mean() for c in classes if np.any(yt == c)]
        bal[b] = float(np.mean(recs)) if recs else np.nan

    def ci(v):
        v = v[np.isfinite(v)]
        return (
            [
                round(float(np.quantile(v, alpha / 2)), 3),
                round(float(np.quantile(v, 1 - alpha / 2)), 3),
            ]
            if len(v)
            else [float("nan"), float("nan")]
        )

    return ci(t1), ci(bal)


def classification_metrics(y_true, y_pred, *, n_boot: int = 0, seed: int = 0) -> dict:
    """Top-1, balanced (macro-recall), majority-class baseline, n_classes, n.

    Pass ``n_boot > 0`` to add 95% percentile-bootstrap confidence intervals
    (``top1_ci``/``balanced_ci``) — the rigor the review asked for, so small-n
    accuracy differences aren't over-read. Default ``n_boot=0`` is unchanged.
    """
    y_true = np.asarray(y_true)
    if len(y_true) == 0:  # empty dataset -> clean n=0 report (no crash / warnings)
        out = {
            "top1": float("nan"),
            "balanced": float("nan"),
            "majority_baseline": float("nan"),
            "n_classes": 0,
            "n": 0,
        }
        if n_boot > 0:
            out["top1_ci"] = out["balanced_ci"] = [float("nan"), float("nan")]
        return out
    yp = np.array([p if p is not None else -1 for p in y_pred])
    classes = np.unique(y_true)
    recalls = [float((yp[y_true == c] == c).mean()) for c in classes]
    _, cnts = np.unique(y_true, return_counts=True)
    out = {
        "top1": round(float((yp == y_true).mean()), 3),
        "balanced": round(float(np.mean(recalls)), 3),
        "majority_baseline": round(float(cnts.max() / len(y_true)), 3),
        "n_classes": int(len(classes)),
        "n": int(len(y_true)),
    }
    if n_boot > 0:
        out["top1_ci"], out["balanced_ci"] = _bootstrap_ci(
            y_true, yp, classes, n_boot=n_boot, seed=seed
        )
    return out


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (tied values share their mean rank), numpy-only.

    Equivalent to ``scipy.stats.rankdata(a)`` (method ``"average"``); implemented
    here so :func:`roc_auc_distances` needs no scipy (keeping the metric usable in
    the numpy-only base install).
    """
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = a[order]
    i, n = 0, len(a)
    while i < n:  # average the ranks within each run of equal values
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc_distances(genuine, impostor) -> float:
    """Standard ROC-AUC for a *distance* score: ``P(impostor > genuine)`` (ties=0.5).

    Equals the Mann–Whitney U statistic divided by ``|genuine|·|impostor|`` — the
    "standard ROC-AUC" the review asked for (replacing the ad-hoc self-zero-dropping
    estimate). Larger = better separation (impostor demos sit farther than genuine
    ones). NaNs are dropped; returns ``nan`` if either side is empty.
    """
    g = np.asarray([v for v in genuine if np.isfinite(v)], float)
    im = np.asarray([v for v in impostor if np.isfinite(v)], float)
    if not len(g) or not len(im):
        return float("nan")
    r = _rankdata_avg(np.concatenate([g, im]))
    u_g = r[: len(g)].sum() - len(g) * (len(g) + 1) / 2  # #(genuine>impostor)+0.5·ties
    # AUC = P(impostor > genuine) + 0.5·P(tie) = 1 - U_g/(|g|·|im|)
    return float(1.0 - u_g / (len(g) * len(im)))


def _pairwise(feats: list, distance) -> np.ndarray:
    """Symmetric pairwise distance matrix over a list of feature sequences."""
    n = len(feats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(feats[i], feats[j])
    return D


def loo_medoid_separability(feats_by_tech: list, *, distance) -> dict:
    """Leak-free (leave-one-demo-out medoid) technique-separability report.

    The earlier feature bake-off built each technique's medoid reference from
    *all* its demos, including the one being tested — so genuine distances were
    optimistically low and needed an ad-hoc self-zero filter. Here a held-out
    demo's GENUINE distance is measured to the medoid of its technique's *other*
    demos (true leave-one-out, no self-zero), and its IMPOSTOR distances to every
    other technique's medoid. Reports leave-one-out nearest-medoid accuracy and
    the standard ROC-AUC (:func:`roc_auc_distances`).

    Args:
        feats_by_tech: per technique, a list of per-demo feature arrays.
        distance: ``(feat_a, feat_b) -> float`` (e.g. normalized DTW).
    """
    techs = [t for t in feats_by_tech if len(t) >= 2]
    n = len(techs)
    intra = [_pairwise(t, distance) for t in techs]
    full_medoid = [int(np.argmin(D.sum(1))) for D in intra]  # impostor reference
    genuine: list[float] = []
    impostor: list[float] = []
    correct = total = 0
    for i, t in enumerate(techs):
        for d in range(len(t)):
            others = [j for j in range(len(t)) if j != d]
            sub = intra[i][np.ix_(others, others)]
            loo_med = others[int(np.argmin(sub.sum(1)))]  # medoid of technique i \ {d}
            g = float(intra[i][d, loo_med])
            genuine.append(g)
            imp_here = [
                distance(t[d], techs[j][full_medoid[j]]) for j in range(n) if j != i
            ]
            impostor.extend(imp_here)
            if imp_here:
                total += 1
                correct += int(g < min(imp_here))
    auc = roc_auc_distances(genuine, impostor)
    return {
        "n_tech": n,
        "n_demos": len(genuine),
        # None (not nan) on degenerate input so the eval JSON stays valid.
        "accuracy": round(correct / total, 3) if total else None,
        "genuine_median": round(float(np.median(genuine)), 4) if genuine else None,
        "impostor_median": round(float(np.median(impostor)), 4) if impostor else None,
        "auc": round(auc, 3) if np.isfinite(auc) else None,
    }


def confusion_pairs(y_true, y_pred, *, top: int | None = None) -> list[dict]:
    """Most-frequent misclassifications as ``{true, pred, count}`` (off-diagonal).

    Per-class confusion that stays readable at many classes: instead of a dense
    N×N matrix it returns only the non-zero true→pred error cells, sorted by
    count. A ``pred`` of ``-1`` means an abstain/no-prediction.
    """
    y_true = np.asarray(y_true)
    yp = np.array([p if p is not None else -1 for p in y_pred])
    tally: dict[tuple, int] = {}
    for t, p in zip(y_true.tolist(), yp.tolist()):
        if t != p:
            tally[(t, p)] = tally.get((t, p), 0) + 1
    pairs = [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in sorted(tally.items(), key=lambda kv: -kv[1])
    ]
    return pairs[:top] if top is not None else pairs


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
