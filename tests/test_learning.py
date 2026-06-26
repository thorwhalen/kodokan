"""Tests for the adaptive learning engine (selection strategies)."""

import random
from datetime import datetime, timedelta, timezone

import pytest

from kodokan import learning as L

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)
SET = ["a", "b", "c", "d", "e", "f"]


def _rec(target, correct, *, chosen=None, days_ago=0.0, score=None, rt=2000):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return {
        "target_key": target, "correct": correct,
        "chosen_key": chosen if chosen is not None else (target if correct else "x"),
        "choice_keys": [target, "x", "y", "z"],
        "score": score if score is not None else (1.0 if correct else 0.0),
        "ts_answered": ts, "ts_presented": ts, "response_time_ms": rt,
        "timed_out": False, "session_id": "s1", "mode": "video_to_name",
    }


@pytest.mark.parametrize("key", list(L.STRATEGIES))
def test_each_strategy_returns_valid_selection(key):
    strat = L.make_strategy(key)
    hist = [_rec("a", False, chosen="b", days_ago=1), _rec("c", True, days_ago=1)]
    sel = strat.next_selection(hist, SET, now=NOW, rng=random.Random(0))
    assert sel.target_key in SET
    assert sel.target_key in sel.choice_keys
    assert len(sel.choice_keys) == 4
    assert len(set(sel.choice_keys)) == 4  # no dup options
    assert set(sel.choice_keys) <= set(SET)


def test_registry_and_default():
    keys = {s["key"] for s in L.list_strategies()}
    assert {"uniform_random", "leitner", "sm2", "confusion_weighted", "fsrs_lite"} <= keys
    assert L.DEFAULT_STRATEGY == "confusion_weighted"
    assert L.make_strategy("nonexistent").key == "confusion_weighted"  # falls back


def test_uniform_anti_repeat():
    strat = L.make_strategy("uniform_random")
    hist = [_rec("a", True)]
    targets = [strat.next_selection(hist, SET, now=NOW, rng=random.Random(i)).target_key for i in range(50)]
    assert "a" not in targets  # never repeats the immediately-previous target


def test_empirical_confusion_matrix():
    hist = [_rec("a", False, chosen="b", days_ago=1)] * 3 + [_rec("a", True, days_ago=1)]
    conf = L.empirical_confusion(hist, now=NOW)
    assert "a" in conf and conf["a"].get("b", 0) > 0
    assert "b" not in conf  # b was never a (correct) target that got missed


def test_confusion_weighted_targets_confused_items():
    strat = L.make_strategy("confusion_weighted", epsilon_explore=0.0)
    # "a" is repeatedly confused with "b"; others answered correctly
    hist = [_rec("a", False, chosen="b", days_ago=1) for _ in range(6)]
    hist += [_rec(k, True, days_ago=1) for k in ("c", "d", "e", "f") for _ in range(2)]
    picks = [strat.next_selection(hist, SET, now=NOW, rng=random.Random(i)).target_key for i in range(60)]
    assert picks.count("a") > picks.count("c")  # confused item surfaces more
    # and "b" (the confuser) should appear as a distractor when "a" is the target
    sels = [strat.next_selection(hist, SET, now=NOW, rng=random.Random(i)) for i in range(60)]
    a_problems = [s for s in sels if s.target_key == "a"]
    assert any("b" in s.choice_keys for s in a_problems)


def test_leitner_promotes_and_demotes():
    strat = L.make_strategy("leitner")
    # "a" always wrong (stays box 1), "b" always right (climbs)
    hist = [_rec("a", False, days_ago=0.01) for _ in range(3)]
    hist += [_rec("b", True, days_ago=0.01) for _ in range(3)]
    boxes, _ = strat._boxes(hist)
    assert boxes["a"] == 1 and boxes["b"] > boxes["a"]


def test_confusion_weighted_merges_pose_after_a_mistake():
    # regression: a logged confusion must NOT replace pose-similarity for other targets
    pose = {"c": {"d": 10.0}}  # c is pose-confusable with d
    strat = L.make_strategy("confusion_weighted", epsilon_explore=0.0)
    hist = [_rec("a", False, chosen="b", days_ago=1)]  # only a<->b confused so far
    sels = [strat.next_selection(hist, ["c", "d", "e", "f", "g"], similarity=pose, now=NOW, rng=random.Random(i))
            for i in range(40)]
    c_sels = [s for s in sels if s.target_key == "c"]
    assert c_sels and all("d" in s.choice_keys for s in c_sels)  # pose-confusable d still surfaces


def test_sm2_and_fsrs_do_not_degenerate_to_one_item():
    hist = [_rec(k, True, days_ago=2) for k in ("a", "b", "c", "d")]  # all seen, equal-ish state
    for key in ("sm2", "fsrs_lite"):
        strat = L.make_strategy(key)
        picks = {strat.next_selection(hist, ["a", "b", "c", "d"], now=NOW, rng=random.Random(i)).target_key
                 for i in range(30)}
        assert len(picks) > 1, f"{key} repeats a single item"


def test_sm2_grade_and_unseen_priority():
    strat = L.make_strategy("sm2")
    assert strat._grade(_rec("a", True, rt=1000)) == 5      # fast correct
    assert strat._grade(_rec("a", True, rt=12000)) == 3     # slow correct
    assert strat._grade({**_rec("a", False), "timed_out": True}) == 0
    # unseen items are prioritized over a freshly-seen one
    hist = [_rec("a", True, days_ago=0.01)]
    picks = {strat.next_selection(hist, ["a", "b", "c", "d"], now=NOW, rng=random.Random(i)).target_key for i in range(20)}
    assert "a" not in picks  # a was just seen; unseen b/c/d chosen
