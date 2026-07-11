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
    ss["vid1"] = {
        "technique": "X",
        "demos": [{"index": 0, "start_s": 1.0, "end_s": 2.0}],
    }
    assert ss["vid1"]["technique"] == "X"
    assert "vid1" in list(ss)


def test_sequence_integrity_clean(synth_seq):
    pytest.importorskip("pandas")
    from kodokan.store import check_sequence_integrity

    rep = check_sequence_integrity(synth_seq)
    assert rep["ok"]
    assert rep["n_duplicate_rows"] == 0
    assert rep["n_oob_xy"] == 0 and rep["n_bad_conf"] == 0 and rep["n_nan_xy"] == 0


def test_tidy_integrity_flags_dup_oob_conf():
    pd = pytest.importorskip("pandas")
    from kodokan.store import check_tidy_integrity

    # one duplicate (fidx,person,keypoint), one out-of-bounds x, one bad conf
    df = pd.DataFrame(
        {
            "fidx": [0, 0, 0, 1],
            "person": [0, 0, 0, 0],
            "keypoint": [0, 1, 1, 0],  # rows 1 & 2 duplicate (0,0,1)
            "x": [10.0, 20.0, 20.0, 9999.0],  # last row out of bounds
            "y": [10.0, 20.0, 20.0, 30.0],
            "conf": [0.9, 0.9, 1.5, 0.9],  # 1.5 is invalid
        }
    )
    rep = check_tidy_integrity(df, width=1280, height=720)
    assert rep["n_duplicate_rows"] == 1
    assert rep["n_oob_xy"] == 1
    assert rep["n_bad_conf"] == 1
    assert not rep["ok"]


def test_tidy_integrity_handles_missing_columns_and_empty():
    pd = pytest.importorskip("pandas")
    from kodokan.store import check_tidy_integrity

    # missing x/y/conf columns must not raise (returns a clean, guarded report)
    rep = check_tidy_integrity(
        pd.DataFrame({"fidx": [0, 1], "person": [0, 0], "keypoint": [0, 0]})
    )
    assert rep["n_rows"] == 2 and rep["n_oob_xy"] == 0
    # empty df is fine too
    empty = check_tidy_integrity(
        pd.DataFrame(columns=["fidx", "person", "keypoint", "x", "y", "conf"])
    )
    assert empty["n_rows"] == 0 and empty["ok"]
