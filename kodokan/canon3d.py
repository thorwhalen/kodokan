"""Pelvis-frame (Procrustes) canonicalization of 3D world landmarks.

The Stage-5 *"3D refutes viewpoint"* conclusion was **withdrawn** by the
adversarial review (``misc/docs/adversarial-review.md``): MediaPipe per-crop
world landmarks are camera-aligned — not canonicalized — and the bounding-box
crop distorts their scale/orientation, so 3D joint angles computed straight off
them are **not** actually viewpoint-invariant. Before a fair 2D-vs-3D test, each
frame's landmarks must be expressed in a *subject-centered anatomical frame*:
origin at the pelvis, axes fixed to the body (hip line + spine), scale normalized
by torso length. This module provides that canonicalization as a pure, testable
geometric function.

After :func:`canonicalize_pose`, two demonstrations of the same articulated pose
seen from different camera azimuths map to the **same** canonical coordinates
(invariant to camera rotation, translation, and — with ``scale=True`` — subject
size). Downstream angle/position features are then genuinely viewpoint-invariant.

Indices default to the MediaPipe **BlazePose-33** convention used by the repo's
3D lift (``scripts/lift_3d_mediapipe.py``); pass ``joints=COCO17`` (or a custom
:class:`BodyJoints`) to reuse it for any layout exposing both hips and shoulders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyJoints:
    """Landmark indices needed to build the pelvis anatomical frame."""

    l_hip: int
    r_hip: int
    l_shoulder: int
    r_shoulder: int


#: MediaPipe BlazePose-33 (the repo's 3D-lift output format).
BLAZEPOSE33 = BodyJoints(l_hip=23, r_hip=24, l_shoulder=11, r_shoulder=12)
#: COCO-17 (the 2D pose format), for reusing the canonicalization on 2.5D data.
COCO17 = BodyJoints(l_hip=11, r_hip=12, l_shoulder=5, r_shoulder=6)


def _unit(v: np.ndarray) -> np.ndarray:
    # A degenerate (near-zero) axis means the pose can't define a frame; surface it
    # as nan so it propagates through R and the frame is dropped downstream (rather
    # than silently returning a finite-but-meaningless basis).
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.full_like(np.asarray(v, dtype=float), np.nan)


def pelvis_frame(
    landmarks: np.ndarray, *, joints: BodyJoints = BLAZEPOSE33
) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical frame of a single pose: ``(pelvis_origin, R)``.

    ``R`` (shape ``(3, 3)``, rows are the new basis) maps world coordinates into
    the subject frame: **x** = subject's right (hip line), **y** = up
    (pelvis→neck spine), **z** = forward (``y × x``). The hip line is
    orthogonalized against the spine so the basis is orthonormal (Gram-Schmidt).
    """
    lm = np.asarray(landmarks, dtype=float)[:, :3]
    pelvis = (lm[joints.l_hip] + lm[joints.r_hip]) / 2
    neck = (lm[joints.l_shoulder] + lm[joints.r_shoulder]) / 2
    up = _unit(neck - pelvis)
    right = _unit(lm[joints.r_hip] - lm[joints.l_hip])
    right = _unit(right - np.dot(right, up) * up)  # orthogonalize against the spine
    fwd = np.cross(up, right)
    return pelvis, np.stack([right, up, fwd], axis=0)


def canonicalize_pose(
    landmarks: np.ndarray, *, joints: BodyJoints = BLAZEPOSE33, scale: bool = True
) -> np.ndarray:
    """Express one frame's 3D landmarks ``(K, C>=3)`` in the pelvis frame ``(K, 3)``.

    Removes camera rotation + translation (and, with ``scale=True``, subject size
    by dividing by torso length). Columns beyond ``xyz`` (e.g. visibility) are
    dropped; ``nan`` inputs propagate (so occluded frames stay droppable).
    """
    xyz = np.asarray(landmarks, dtype=float)[:, :3]
    pelvis, R = pelvis_frame(xyz, joints=joints)
    out = (xyz - pelvis) @ R.T
    if scale:
        neck = (out[joints.l_shoulder] + out[joints.r_shoulder]) / 2
        torso = float(np.linalg.norm(neck))
        out = out / (torso if torso > 1e-9 else 1.0)
    return out


def canonicalize_sequence(
    world: np.ndarray, *, joints: BodyJoints = BLAZEPOSE33, scale: bool = True
) -> np.ndarray:
    """Canonicalize every frame/person of a ``(..., K, C)`` landmark stack → ``(..., K, 3)``.

    Each frame (and person) is canonicalized to *its own* pelvis frame — the
    viewpoint-invariant choice for pose-only technique features.
    """
    world = np.asarray(world, dtype=float)
    lead, K = world.shape[:-2], world.shape[-2]
    flat = world.reshape(-1, K, world.shape[-1])
    out = np.stack([canonicalize_pose(fr, joints=joints, scale=scale) for fr in flat])
    return out.reshape(*lead, K, 3)
