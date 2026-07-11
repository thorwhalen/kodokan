"""Tests for the tracker identity-swap-rate diagnostic."""

import numpy as np
import pytest

from kodokan.pose import PoseSequence


def _seq(kps):
    return PoseSequence(kps, np.arange(len(kps)), 25.0, 1280, 720, "test", "v.mp4")


def test_swap_rate_zero_for_stable_tracks(synth_seq):
    pytest.importorskip("scipy")
    from kodokan.track import identity_swap_rate

    rep = identity_swap_rate(synth_seq)
    assert rep["n_pairs"] > 0
    assert rep["swap_rate"] == 0.0  # two well-separated, non-crossing people


def test_swap_rate_detects_label_swap():
    pytest.importorskip("scipy")
    from kodokan.track import identity_swap_rate

    # two persons at fixed, separated locations; swap the two slot labels halfway
    F = 20
    kps = np.zeros((F, 2, 17, 3), dtype=np.float32)
    kps[:, :, :, 2] = 0.9
    for f in range(F):
        left = np.full((17, 2), 200.0, dtype=np.float32)
        right = np.full((17, 2), 900.0, dtype=np.float32)
        if f < F // 2:
            kps[f, 0, :, :2], kps[f, 1, :, :2] = left, right
        else:  # slots swapped: an identity discontinuity at the midpoint
            kps[f, 0, :, :2], kps[f, 1, :, :2] = right, left

    rep = identity_swap_rate(_seq(kps))
    assert rep["n_swaps"] >= 1
    assert rep["swap_rate"] > 0.0
