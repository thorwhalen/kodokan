"""Tests for the dol-backed pose/segments stores."""

import numpy as np
import pytest


def test_pose_store_roundtrip(tmp_path, synth_seq):
    pytest.importorskip("pyarrow")
    pytest.importorskip("dol")
    from kodokan.store import pose_store

    ps = pose_store(tmp_path)
    ps["vid1"] = synth_seq
    assert "vid1" in list(ps)
    s2 = ps["vid1"]
    assert s2.keypoints.shape == synth_seq.keypoints.shape
    m = np.isfinite(synth_seq.keypoints[..., 0])
    assert np.allclose(s2.keypoints[m], synth_seq.keypoints[m], atol=1e-3)
    assert s2.source_url == synth_seq.source_url


def test_tidy_df_columns(synth_seq):
    pytest.importorskip("pandas")
    from kodokan.store import sequence_to_tidy_df

    df = sequence_to_tidy_df(synth_seq, video_id="vid1")
    for col in ("video_id", "frame", "t_sec", "person", "keypoint", "x", "y", "conf"):
        assert col in df.columns
    assert (df["person"] < synth_seq.n_persons).all()


def test_segments_store_roundtrip(tmp_path):
    pytest.importorskip("dol")
    from kodokan.store import segments_store

    ss = segments_store(tmp_path)
    ss["vid1"] = {"technique": "X", "demos": [{"index": 0, "start_s": 1.0, "end_s": 2.0}]}
    assert ss["vid1"]["technique"] == "X"
    assert "vid1" in list(ss)
