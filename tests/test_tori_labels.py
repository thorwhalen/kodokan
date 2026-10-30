"""Tests for the tori hand-labelling helpers (#49)."""

import numpy as np

from kodokan.tori_labels import (
    agreement_report,
    manifest_entry,
    person_boxes,
    stratified_sample,
    wilson_interval,
)
from test_recognize import _two_person_seq


def test_manifest_entry_boxes_and_hidden_answer():
    seq = _two_person_seq([200.0, 500.0])  # person 0 hips higher (smaller y) -> tori
    e = manifest_entry(throw="x", video_id="v", source="s", pose_seq=seq, start_s=0.0, end_s=1.0, clip_file="c.mp4")
    assert e["slots"] == [0, 1] and e["tori_slot"] == 0 and not e["abstained"]
    assert len(e["t"]) == len(e["boxes"]) and e["t"][0] == 0.0
    a, b = e["boxes"][0]
    assert 0 <= a[0] < a[2] <= 1 and 0 <= b[1] < b[3] <= 1
    assert a[2] < b[0]  # person 0 is left of person 1


def test_person_boxes_missing_slot_is_none():
    seq = _two_person_seq([200.0, 500.0])
    seq.keypoints[:, 1] = np.nan
    out = person_boxes(seq, 0.0, 1.0, slots=[0, 1])
    assert all(row[1] is None and row[0] is not None for row in out["boxes"])


def _entries():
    return [
        {"id": f"{t}{i}", "throw": t, "source": s, "abstained": ab, "slots": [0, 1], "tori_slot": 0}
        for t in "abcd" for i, (s, ab) in enumerate([("s1", False), ("s2", False), ("s1", True)])
    ]


def test_stratified_sample_guarantees_abstains_and_spreads_throws():
    out = stratified_sample(_entries(), 8, abstain_share=0.5, seed=1)
    assert len(out) == 8 and len({e["id"] for e in out}) == 8
    assert sum(e["abstained"] for e in out) == 4
    assert len({e["throw"] for e in out}) == 4


def test_stratified_sample_is_deterministic_and_small_pool_ok():
    assert stratified_sample(_entries(), 5, seed=3) == stratified_sample(_entries(), 5, seed=3)
    assert len(stratified_sample(_entries()[:3], 60)) == 3


def test_agreement_report():
    man = _entries()[:4]  # a0 a1 a2(abstained) b0
    labels = [
        {"id": "a0", "verdict": "A", "ts": 1},  # agree (slot 0 == tori)
        {"id": "a1", "verdict": "B", "ts": 1},  # disagree
        {"id": "a2", "verdict": "cant_tell", "ts": 1},
        {"id": "b0", "verdict": "B", "ts": 1},
        {"id": "b0", "verdict": "A", "ts": 2},  # re-labelled: latest wins -> agree
    ]
    r = agreement_report(man, labels)
    assert r["overall"]["decided"] == 3 and r["overall"]["agree"] == 2
    assert r["abstained"]["cant_tell_rate"] == 1.0 and r["confident"]["cant_tell_rate"] == 0.0
    assert r["unlabelled"] == 0


def test_wilson_interval_brackets_estimate():
    lo, hi = wilson_interval(45, 60)
    assert lo < 0.75 < hi and 0 < lo and hi < 1
