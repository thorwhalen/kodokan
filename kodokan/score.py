"""Score a demonstration/attempt against a reference (Stage 5 scaffold).

Judo has no labeled "quality" data, so scoring is calibrated to the *spread of
genuine demonstrations*: for a technique we take the **medoid** demo (minimum total
joint-angle-DTW distance to the others) as the canonical reference, measure how far
each genuine demo sits from it (the baseline distribution), and score a query by
where its distance falls relative to that baseline. Feedback decomposes the
difference per joint-angle and per phase (entry / middle / finish ≈ kuzushi /
tsukuri / kake). See research §7 (Stage 5).

Honest limits: 2D joint angles are speed-invariant but not viewpoint-invariant, and
"reference = medoid" assumes the demos are mostly correct. This is a usable scaffold,
not a calibrated judge.
"""

from __future__ import annotations

import numpy as np

from kodokan.compare import (
    ANGLE_NAMES,
    _clean,
    angle_features,
    compare,
    per_angle_deviation,
    primary_person,
)
from kodokan.pose import PoseSequence


def demo_features(
    pose_seq: PoseSequence, start_s: float, end_s: float, *, person: int | None = None
) -> np.ndarray:
    """Cleaned joint-angle features for one demo window (primary person by default)."""
    fps = pose_seq.fps
    fr = (int(start_s * fps), int(end_s * fps))
    p = person if person is not None else primary_person(pose_seq, frame_range=fr)
    return _clean(angle_features(pose_seq, person=p, frame_range=fr))


def build_reference(feature_list: list[np.ndarray]) -> dict:
    """Pick the medoid demo as reference and record the baseline distance spread.

    Returns ``{medoid, reference, baseline, distance_matrix}`` where ``baseline`` is
    the array of normalized-DTW distances from every other demo to the medoid.
    """
    feats = [f for f in feature_list if len(f) >= 2]
    n = len(feats)
    if n == 0:
        return {
            "medoid": None,
            "reference": None,
            "baseline": np.array([]),
            "distance_matrix": np.zeros((0, 0)),
        }
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = compare(feats[i], feats[j])["normalized"]
    medoid = int(np.argmin(D.sum(axis=1)))
    baseline = np.array([D[medoid, j] for j in range(n) if j != medoid])
    return {
        "medoid": medoid,
        "reference": feats[medoid],
        "baseline": baseline,
        "distance_matrix": D,
    }


def score(query_features: np.ndarray, reference: dict) -> dict:
    """Score a query against a reference, calibrated to the genuine-demo spread.

    ``score`` is 0–100: 100 ≈ as close as the closest genuine demo, 0 ≈ as far as the
    90th-percentile (or worse). ``closer_than_pct`` is the percent of genuine demos
    the query is closer-to-reference than.
    """
    d = compare(query_features, reference["reference"])["normalized"]
    b = reference["baseline"]
    if b.size == 0:
        return {"distance": round(float(d), 4), "score": None, "closer_than_pct": None}
    lo, hi = float(np.min(b)), float(np.quantile(b, 0.9))
    s = 100.0 * float(np.clip((hi - d) / (hi - lo + 1e-9), 0.0, 1.0))
    return {
        "distance": round(float(d), 4),
        "score": round(s, 1),
        "closer_than_pct": round(100.0 * float((b >= d).mean()), 0),
    }


def feedback(query_features: np.ndarray, reference: dict, *, n_phases: int = 3) -> dict:
    """Interpretable difference: per joint-angle and per phase (degrees)."""
    res = compare(query_features, reference["reference"])
    per_angle = np.degrees(per_angle_deviation(res))
    a, b, path = res["a"], res["b"], res["path"]
    phases = []
    L = len(path)
    for ph in range(n_phases):
        seg = path[ph * L // n_phases : (ph + 1) * L // n_phases]
        if seg:
            diffs = np.array([np.abs(a[i] - b[j]) for i, j in seg])
            phases.append(round(float(np.degrees(diffs.mean())), 1))
        else:
            phases.append(float("nan"))
    return {
        "per_angle_deg": {
            k: round(float(v), 1) for k, v in zip(ANGLE_NAMES, per_angle)
        },
        "per_phase_deg": phases,
        "worst_joint": ANGLE_NAMES[int(np.nanargmax(per_angle))]
        if per_angle.size
        else None,
    }
