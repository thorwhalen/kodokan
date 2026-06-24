"""Tests for motion-energy hysteresis segmentation."""

import numpy as np


def _bump(t, center, width, height):
    return height * np.exp(-((t - center) ** 2) / (2 * width**2))


def test_two_bumps_two_segments():
    from kodokan.segment import find_segments

    F = 300
    t = np.arange(F)
    e = _bump(t, 85, 22, 50) + _bump(t, 225, 22, 50)
    segs = find_segments(
        e, np.arange(F), 25.0, smooth_sigma=2, low_quantile=0.25, high_quantile=0.5,
        min_duration_s=0.5, merge_gap_s=0.2,
    )
    assert len(segs) == 2
    assert segs[0].end_s < segs[1].start_s


def test_hysteresis_no_split_on_midrep_dip():
    from kodokan.segment import find_segments

    F = 220
    t = np.arange(F)
    e = _bump(t, 110, 40, 40)
    e[100:120] *= 0.6  # a dip mid-rep that stays above t_low
    segs = find_segments(
        e, np.arange(F), 25.0, smooth_sigma=2, low_quantile=0.2, high_quantile=0.5,
        min_duration_s=0.5, merge_gap_s=0.0,
    )
    assert len(segs) == 1


def test_pose_motion_energy_static_vs_moving(synth_seq):
    from kodokan.segment import pose_motion_energy

    e = pose_motion_energy(synth_seq)
    assert e.shape[0] == synth_seq.n_frames
    assert e[1:].mean() > 0  # the synthetic sequence has motion

    static = synth_seq.keypoints.copy()
    static[:] = static[0]  # freeze every frame
    from dataclasses import replace

    e0 = pose_motion_energy(replace(synth_seq, keypoints=static))
    assert e0[1:].max() < 1e-6
