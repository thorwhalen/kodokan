"""Compare two demonstrations of a throw (training-free — Stage 1).

We represent each frame by a vector of **joint angles** (e.g. elbow, shoulder,
hip, knee flexion). Angles are inherently invariant to translation and scale —
and far more occlusion-tolerant than raw pixel coordinates — so no skeleton
normalization is required. Two demonstrations are then aligned with **Dynamic
Time Warping**, which absorbs differences in execution *speed* and yields a
similarity distance plus the warping path (reused to draw per-angle deviation).

Caveat (honest): joint angles computed from 2D keypoints are **not** invariant to
camera *viewpoint*, so two demos shot from very different angles will read as more
different than they are. Viewpoint-invariant comparison (JEANIE-style joint
temporal+viewpoint warping, or 3D lifting) is the planned Stage-3 upgrade; see
``misc/docs/research-architecture.md`` §7.
"""

from __future__ import annotations

import numpy as np

from kodokan.pose import PoseSequence

#: COCO-17 joint angles: (name, point_a, vertex, point_c) → angle at the vertex.
ANGLE_DEFS: tuple[tuple[str, int, int, int], ...] = (
    ("l_elbow", 5, 7, 9),
    ("r_elbow", 6, 8, 10),
    ("l_shoulder", 11, 5, 7),
    ("r_shoulder", 12, 6, 8),
    ("l_hip", 5, 11, 13),
    ("r_hip", 6, 12, 14),
    ("l_knee", 11, 13, 15),
    ("r_knee", 12, 14, 16),
)
ANGLE_NAMES: tuple[str, ...] = tuple(d[0] for d in ANGLE_DEFS)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (radians) at vertex ``b`` of the triangle a-b-c."""
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return np.nan
    return float(np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)))


def joint_angles(keypoints_17x3: np.ndarray) -> np.ndarray:
    """Map one person's COCO-17 keypoints ``(17, 3)`` to the angle vector."""
    xy = keypoints_17x3[:, :2]
    return np.array([_angle(xy[a], xy[b], xy[c]) for _, a, b, c in ANGLE_DEFS])


def primary_person(
    pose_seq: PoseSequence, *, frame_range: tuple[int, int] | None = None
) -> int:
    """Person slot with the highest presence over the (optional) frame window."""
    kp, fi = pose_seq.keypoints, pose_seq.frame_indices
    if frame_range is not None:
        m = (fi >= frame_range[0]) & (fi <= frame_range[1])
        kp = kp[m]
    present = ~np.all(np.isnan(kp[..., 0]), axis=2)  # (F, P)
    return int(np.argmax(present.mean(axis=0))) if len(kp) else 0


def angle_features(
    pose_seq: PoseSequence,
    *,
    person: int = 0,
    frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Per-frame joint-angle features ``(F, n_angles)`` for one person."""
    kp, fi = pose_seq.keypoints, pose_seq.frame_indices
    if frame_range is not None:
        m = (fi >= frame_range[0]) & (fi <= frame_range[1])
        kp = kp[m]
    if len(kp) == 0:
        return np.empty((0, len(ANGLE_DEFS)))
    return np.stack([joint_angles(kp[f, person]) for f in range(kp.shape[0])])


def _clean(feats: np.ndarray) -> np.ndarray:
    """Drop frames with any missing angle (occluded/absent person)."""
    return feats[~np.any(np.isnan(feats), axis=1)]


def compare(features_a: np.ndarray, features_b: np.ndarray) -> dict:
    """DTW-align two angle-feature sequences.

    Returns a dict with ``distance`` (total DTW cost), ``normalized`` (cost per
    aligned step — comparable across different-length demos), the warping ``path``,
    and the cleaned input arrays ``a``/``b``.
    """
    from dtaidistance import dtw_ndim

    a = np.ascontiguousarray(_clean(features_a), dtype=np.double)
    b = np.ascontiguousarray(_clean(features_b), dtype=np.double)
    if len(a) < 2 or len(b) < 2:
        return {"distance": np.nan, "normalized": np.nan, "path": [], "a": a, "b": b}
    path = dtw_ndim.warping_path(a, b)  # list of (i, j)
    # Mean Euclidean cost per aligned step. (dist/len(path) would leave a ~1/sqrt(len)
    # length bias because dist is an L2 path cost, not a sum of per-step distances.)
    steps = [float(np.linalg.norm(a[i] - b[j])) for i, j in path]
    return {
        "distance": float(dtw_ndim.distance(a, b)),
        "normalized": float(np.mean(steps)) if steps else float("nan"),
        "path": path,
        "a": a,
        "b": b,
    }


def distance_matrix(feature_seqs: list[np.ndarray]) -> np.ndarray:
    """Pairwise normalized-DTW distance matrix over a list of feature sequences."""
    n = len(feature_seqs)
    d = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            d[i, j] = compare(feature_seqs[i], feature_seqs[j])["normalized"]
    return d


def per_angle_deviation(result: dict) -> np.ndarray:
    """Mean absolute per-angle difference along a DTW alignment (radians)."""
    a, b, path = result["a"], result["b"], result["path"]
    if not path:
        return np.full(len(ANGLE_NAMES), np.nan)
    diffs = np.array([np.abs(a[i] - b[j]) for i, j in path])  # (len_path, n_angles)
    return diffs.mean(axis=0)


def time_stretch(features: np.ndarray, factor: float) -> np.ndarray:
    """Linearly resample a feature sequence to ``factor``× its length (speed change)."""
    n = len(features)
    if n < 2:
        return features.copy()
    m = max(2, int(round(n * factor)))
    src = np.linspace(0, n - 1, m)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (src - lo)[:, None]
    return features[lo] * (1 - frac) + features[hi] * frac
