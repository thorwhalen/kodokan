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
