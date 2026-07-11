"""Tests for pelvis-frame (Procrustes) 3D canonicalization — viewpoint invariance."""

import numpy as np

from kodokan.canon3d import BLAZEPOSE33, canonicalize_pose, canonicalize_sequence


def _synthetic_pose33(seed=0):
    """A non-degenerate BlazePose-33 pose (distinct hips/shoulders/limbs)."""
    rng = np.random.RandomState(seed)
    lm = rng.randn(33, 3).astype(float)
    # force a well-defined torso frame
    lm[BLAZEPOSE33.l_hip] = [-1.0, 0.0, 0.0]
    lm[BLAZEPOSE33.r_hip] = [1.0, 0.0, 0.0]
    lm[BLAZEPOSE33.l_shoulder] = [-0.9, 3.0, 0.2]
    lm[BLAZEPOSE33.r_shoulder] = [0.9, 3.0, -0.1]
    return lm


def _random_rotation(seed):
    """A proper rotation matrix (det = +1) via QR of a random matrix."""
    rng = np.random.RandomState(seed)
    q, r = np.linalg.qr(rng.randn(3, 3))
    q = q @ np.diag(np.sign(np.diag(r)))  # fix signs -> orthonormal
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_canonicalize_invariant_to_rotation_translation_scale():
    lm = _synthetic_pose33()
    Q = _random_rotation(42)
    s, t = 3.7, np.array([12.0, -5.0, 8.0])
    lm2 = s * (lm @ Q.T) + t  # arbitrary camera pose + zoom

    c1 = canonicalize_pose(lm, joints=BLAZEPOSE33, scale=True)
    c2 = canonicalize_pose(lm2, joints=BLAZEPOSE33, scale=True)
    assert np.allclose(c1, c2, atol=1e-6)


def test_canonicalize_pelvis_at_origin():
    lm = _synthetic_pose33(1)
    c = canonicalize_pose(lm, joints=BLAZEPOSE33, scale=True)
    pelvis = (c[BLAZEPOSE33.l_hip] + c[BLAZEPOSE33.r_hip]) / 2
    assert np.allclose(pelvis, 0.0, atol=1e-6)


def test_canonicalize_degenerate_pose_propagates_nan():
    """A pose with no torso (hips==shoulders) can't define a frame -> NaN (droppable)."""
    lm = _synthetic_pose33(2)
    lm[BLAZEPOSE33.l_shoulder] = lm[BLAZEPOSE33.l_hip]  # collapse the spine
    lm[BLAZEPOSE33.r_shoulder] = lm[BLAZEPOSE33.r_hip]
    out = canonicalize_pose(lm, joints=BLAZEPOSE33)
    assert np.isnan(out).any()  # not finite garbage


def test_canonicalize_sequence_shape_and_nan():
    world = np.stack([_synthetic_pose33(i) for i in range(4)])  # (4, 33, 3)
    out = canonicalize_sequence(world, joints=BLAZEPOSE33)
    assert out.shape == (4, 33, 3)
    world[0, 0] = np.nan  # a missing landmark propagates (frame stays droppable)
    out2 = canonicalize_sequence(world, joints=BLAZEPOSE33)
    assert np.isnan(out2[0]).any()
