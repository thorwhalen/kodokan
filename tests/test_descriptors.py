"""Tests for experimental feature descriptors (shapes per mode)."""

import pytest

EXPECTED_DIM = {
    "angles": 8,
    "angles_vel": 16,
    "angles_pos": 42,
    "angles_both": 16,
    "pos_both": 68,
}


@pytest.mark.parametrize("mode,dim", list(EXPECTED_DIM.items()))
def test_demo_descriptor_shapes(synth_seq, mode, dim):
    from kodokan.descriptors import demo_descriptor

    f = demo_descriptor(synth_seq, 0.0, 2.0, mode=mode)
    assert f.ndim == 2
    assert f.shape[1] == dim
    assert len(f) > 0


def test_unknown_mode_raises(synth_seq):
    from kodokan.descriptors import demo_descriptor

    with pytest.raises(ValueError):
        demo_descriptor(synth_seq, 0.0, 2.0, mode="nope")


def test_norm_positions_robust_to_collapsed_torso():
    """A torso that projects to ~zero length must not explode positions (review fix)."""
    import numpy as np

    from kodokan.descriptors import _norm_positions

    kp = np.zeros((17, 3), dtype=float)
    kp[:, 2] = 0.9
    # a spread-out body (nose high, ankles low) but shoulders coincident with hips
    kp[0, :2] = [500, 100]  # nose (defines a large bbox)
    kp[[15, 16], :2] = [[490, 900], [510, 900]]  # ankles
    kp[[5, 6], :2] = [[500, 500], [500, 500]]  # shoulders on the hip line
    kp[[11, 12], :2] = [[500, 500], [500, 500]]  # hips (torso length == 0)

    out = _norm_positions(kp)  # default clip=8.0
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() <= 8.0 + 1e-9  # bounded, not an outlier fingerprint


def test_norm_positions_nan_hip_propagates():
    import numpy as np

    from kodokan.descriptors import _norm_positions

    kp = np.zeros((17, 3), dtype=float)
    kp[:, 2] = 0.9
    kp[11, :2] = np.nan  # a missing hip -> frame should stay droppable
    assert np.isnan(_norm_positions(kp)).any()
