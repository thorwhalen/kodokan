"""Tests for joint-angle features and DTW comparison."""

import numpy as np
import pytest


def test_joint_angles_right_angle():
    from kodokan.compare import joint_angles

    kp = np.zeros((17, 3), dtype=float)
    kp[:, 2] = 1.0
    # L shoulder(5)-elbow(7)-wrist(9) forming a right angle at the elbow
    kp[5, :2] = [0, 0]
    kp[7, :2] = [0, 1]
    kp[9, :2] = [1, 1]
    ang = joint_angles(kp)  # l_elbow is index 0 in ANGLE_DEFS
    assert abs(ang[0] - np.pi / 2) < 0.05


def test_compare_identical_is_zero():
    pytest.importorskip("dtaidistance")
    from kodokan.compare import compare

    a = np.cumsum(np.ones((30, 4)), axis=0) * 0.01
    assert compare(a, a.copy())["normalized"] < 1e-6


def test_compare_speed_invariance():
    pytest.importorskip("dtaidistance")
    from kodokan.compare import compare, time_stretch

    t = np.linspace(0, 4 * np.pi, 40)
    a = np.stack([np.sin(t + k) for k in range(4)], axis=1)
    b = time_stretch(a, 1.6)  # same motion, slower
    diff = np.stack([np.sin(2.2 * t + k) for k in range(4)], axis=1)  # different motion
    assert compare(a, b)["normalized"] < compare(a, diff)["normalized"]
