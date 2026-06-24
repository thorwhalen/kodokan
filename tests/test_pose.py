"""Tests for PoseSequence constants, properties, and NPZ round-trip."""

import numpy as np


def test_coco_constants():
    from kodokan.pose import COCO17_KEYPOINTS, COCO17_SKELETON

    assert len(COCO17_KEYPOINTS) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in COCO17_SKELETON)


def test_properties(synth_seq):
    assert synth_seq.n_frames == synth_seq.keypoints.shape[0]
    assert synth_seq.n_persons == 2
    assert len(synth_seq.times()) == synth_seq.n_frames
    assert synth_seq.times()[0] == 0.0


def test_npz_roundtrip(tmp_path, synth_seq):
    from kodokan.pose import PoseSequence

    p = synth_seq.save_npz(tmp_path / "s.npz")
    s2 = PoseSequence.load_npz(p)
    assert s2.keypoints.shape == synth_seq.keypoints.shape
    assert np.allclose(s2.keypoints, synth_seq.keypoints)
    assert s2.fps == synth_seq.fps
    assert s2.source_url == synth_seq.source_url
