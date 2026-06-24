"""Tests for reference-based scoring + feedback."""

import numpy as np
import pytest


def _demo_set():
    t = np.linspace(0, 4 * np.pi, 30)
    base = np.stack([np.sin(t + k) for k in range(4)], axis=1)
    return base, [base + 0.01 * i for i in range(5)]  # a tight cluster of "genuine" demos


def test_build_reference_and_score_monotonic():
    pytest.importorskip("dtaidistance")
    from kodokan.score import build_reference, score

    base, feats = _demo_set()
    ref = build_reference(feats)
    assert ref["medoid"] is not None
    assert ref["baseline"].size == len(feats) - 1

    near = score(base + 0.02, ref)  # close to the cluster
    far = score(base * 3.0 + 5.0, ref)  # very different motion
    assert 0 <= near["score"] <= 100
    assert near["score"] >= far["score"]
    assert near["distance"] < far["distance"]


def test_feedback_shape():
    pytest.importorskip("dtaidistance")
    from kodokan.compare import ANGLE_NAMES
    from kodokan.score import build_reference, feedback

    base, feats = _demo_set()  # 4-d here; feedback uses angle-name count -> use angle-shaped feats
    # build angle-shaped (8-d) feats so per_angle maps to ANGLE_NAMES
    t = np.linspace(0, 2 * np.pi, 25)
    afeats = [np.stack([np.sin(t + k) + 0.01 * i for k in range(len(ANGLE_NAMES))], axis=1) for i in range(4)]
    ref = build_reference(afeats)
    fb = feedback(afeats[0], ref)
    assert set(fb["per_angle_deg"]) == set(ANGLE_NAMES)
    assert len(fb["per_phase_deg"]) == 3
