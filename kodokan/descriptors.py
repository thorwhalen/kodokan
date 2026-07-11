"""Experimental per-demo feature descriptors (to test what makes throws separable).

The Stage-5 eval showed plain single-person joint angles don't discriminate
technique (dominated by viewpoint + tori/uke role inconsistency). This module
offers alternative descriptors so the eval harness can measure which lever helps:

- ``angles``      — primary person's joint angles (the current baseline).
- ``angles_vel``  — angles + their frame-to-frame velocity (dynamics).
- ``angles_pos``  — angles + root-centered, torso-scaled keypoint positions.
- ``angles_both`` — both people's joint angles, ordered by activity (captures the
                    tori/uke *interaction*, robust to which slot is which).
- ``pos_both``    — both people's normalized positions, activity-ordered.

All are 2D (still viewpoint-variant); the point is to isolate how much of the
failure is role/interaction/descriptor vs genuinely needing 3D.
"""

from __future__ import annotations

import numpy as np

from kodokan.compare import ANGLE_DEFS, _clean, primary_person
from kodokan.pose import PoseSequence

_HIP = (11, 12)
_SHO = (5, 6)


def _angles(kp17: np.ndarray) -> np.ndarray:
    xy = kp17[:, :2]
    out = np.empty(len(ANGLE_DEFS))
    for i, (_, a, b, c) in enumerate(ANGLE_DEFS):
        ba, bc = xy[a] - xy[b], xy[c] - xy[b]
        nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
        out[i] = (
            np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1))
            if nba > 1e-6 and nbc > 1e-6
            else np.nan
        )
    return out


def _torso_scale(xy: np.ndarray, *, floor_frac: float = 0.1) -> np.floating:
    """Robust body scale for position normalization (hip→shoulder length, floored).

    The raw torso length collapses toward zero when the torso is edge-on to the
    camera (shoulders projecting onto the hips). Dividing positions by a near-zero
    scale explodes them into per-clip *outlier fingerprints* — exactly the kind of
    spurious cue the adversarial review flagged. We therefore floor the torso
    length by ``floor_frac`` of the person's keypoint bounding-box diagonal, a
    projection-robust body-size proxy that stays finite even edge-on. Missing
    hip/shoulder keypoints yield ``nan`` (so the frame is dropped downstream), as
    before.
    """
    hip = (xy[_HIP[0]] + xy[_HIP[1]]) / 2
    sho = (xy[_SHO[0]] + xy[_SHO[1]]) / 2
    torso = np.linalg.norm(sho - hip)  # nan if hip/shoulder missing
    finite = xy[np.all(np.isfinite(xy), axis=1)]
    bbox_diag = (
        float(np.hypot(*(finite.max(0) - finite.min(0)))) if len(finite) >= 2 else 0.0
    )
    # np.maximum propagates nan, so a missing torso stays nan (frame dropped).
    return np.maximum(np.maximum(torso, floor_frac * bbox_diag), 1e-6)


def _norm_positions(kp17: np.ndarray, *, clip: float | None = 8.0) -> np.ndarray:
    """Hip-centered, torso-scaled keypoint positions ``(34,)`` (robust scale).

    Positions are in torso-length units, so a whole body spans only a few units
    from the hip center; ``clip`` bounds residual outliers (``None`` disables it).
    ``nan`` passes through both the scale and the clip, so occluded frames are
    still dropped by :func:`kodokan.compare._clean`.
    """
    xy = kp17[:, :2].astype(float)
    hip = (xy[_HIP[0]] + xy[_HIP[1]]) / 2
    out = (xy - hip) / _torso_scale(xy)
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out.reshape(-1)  # (34,)


def _window(pose_seq: PoseSequence, start_s: float, end_s: float):
    fps = pose_seq.fps
    fi = pose_seq.frame_indices
    m = (fi >= int(start_s * fps)) & (fi <= int(end_s * fps))
    return pose_seq.keypoints[m]  # (F, P, 17, 3)


def _slot_activity(kp_fp: np.ndarray) -> float:
    """Total confidence-weighted keypoint speed for one (F,17,3) slot sequence."""
    xy, conf = kp_fp[..., :2], kp_fp[..., 2]
    if len(xy) < 2:
        return 0.0
    disp = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    w = np.minimum(conf[1:], conf[:-1])
    return float(np.nansum(np.where(np.isfinite(disp), disp * w, 0.0)))


def demo_descriptor(
    pose_seq: PoseSequence, start_s: float, end_s: float, *, mode: str = "angles"
) -> np.ndarray:
    """Per-frame feature matrix for one demo window under the chosen ``mode``."""
    kp = _window(pose_seq, start_s, end_s)  # (F, P, 17, 3)
    F = len(kp)
    if F == 0:
        return np.empty((0, 1))

    if mode in ("angles", "angles_vel", "angles_pos"):
        p = primary_person(
            PoseSequence(
                kp, np.arange(F), pose_seq.fps, pose_seq.width, pose_seq.height, "", ""
            )
        )
        ang = np.stack([_angles(kp[f, p]) for f in range(F)])
        if mode == "angles":
            feat = ang
        elif mode == "angles_vel":
            vel = np.vstack([np.zeros((1, ang.shape[1])), np.diff(ang, axis=0)])
            feat = np.concatenate([ang, vel], axis=1)
        else:  # angles_pos
            pos = np.stack([_norm_positions(kp[f, p]) for f in range(F)])
            feat = np.concatenate([ang, pos], axis=1)
        return _clean(feat)

    if mode in ("angles_both", "pos_both"):
        P = kp.shape[1]
        order = sorted(
            range(P), key=lambda s: -_slot_activity(kp[:, s])
        )  # most-active first
        per_frame = []
        for f in range(F):
            parts = []
            ok = True
            for s in order:
                vec = (
                    _angles(kp[f, s])
                    if mode == "angles_both"
                    else _norm_positions(kp[f, s])
                )
                parts.append(vec)
                if not np.all(np.isfinite(vec)):
                    ok = False
            per_frame.append(
                np.concatenate(parts)
                if ok
                else np.full(sum(len(p) for p in parts), np.nan)
            )
        return _clean(np.stack(per_frame))

    raise ValueError(f"unknown mode {mode!r}")
