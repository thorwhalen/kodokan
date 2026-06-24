"""Shared fixtures: synthetic two-person COCO-17 pose sequences (no I/O, no models)."""

import numpy as np
import pytest

from kodokan.pose import PoseSequence

# rough standing COCO-17 layout (x, y), nose at top, ankles at bottom
_BASE = np.array(
    [
        [0, -80], [-5, -85], [5, -85], [-10, -83], [10, -83],  # nose, eyes, ears
        [-20, -50], [20, -50], [-30, -20], [30, -20], [-30, 10], [30, 10],  # shoulders/elbows/wrists
        [-12, 0], [12, 0], [-14, 40], [14, 40], [-14, 80], [14, 80],  # hips/knees/ankles
    ],
    dtype=np.float32,
)


def make_keypoints(F=60, P=2, fps=25.0):
    kps = np.zeros((F, P, 17, 3), dtype=np.float32)
    for p in range(P):
        offset = np.array([300 + p * 400, 300], dtype=np.float32)
        amp = 5.0 + 4.0 * p  # distinct per-person activity (stable ordering)
        for f in range(F):
            jitter = np.zeros((17, 2), dtype=np.float32)
            jitter[[9, 10], 0] = amp * np.sin(2 * np.pi * f / 15.0)  # wrists swing in x
            jitter[[7, 8], 0] = 0.5 * amp * np.sin(2 * np.pi * f / 15.0)  # elbows
            kps[f, p, :, :2] = _BASE + offset + jitter
            kps[f, p, :, 2] = 0.9
    return kps


@pytest.fixture
def synth_seq():
    kps = make_keypoints()
    return PoseSequence(
        kps, np.arange(len(kps)), 25.0, 1280, 720, "test", "video.mp4", "http://example/x"
    )
