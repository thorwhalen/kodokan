"""Tests for the quiz's learning engine: kodokan records adapted to learny's strategies."""

import pytest

pytest.importorskip("learny", reason="the quiz needs learny (extra: quiz)")
import random
from datetime import datetime, timedelta, timezone

import pytest

from kodokan import learning as L

from learny.distractors import confusion_weights

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


def _next(strat, hist, items=None, *, seed=0, **kw):
    """Run ``strat`` over kodokan record dicts, as the quiz does."""
    return strat.next_selection(
        L.to_responses(hist), items or SET, now=NOW.timestamp(), rng=random.Random(seed), **kw
    )


@pytest.mark.parametrize("key", [k for k in L.STRATEGIES if k != "weakest"])
def test_each_strategy_returns_valid_selection(key):
    strat = L.make_strategy(key)
    hist = [_rec("a", False, chosen="b", days_ago=1), _rec("c", True, days_ago=1)]
    sel = _next(strat, hist, SET, seed=0)
    assert sel.target in SET
    assert sel.target in sel.options
    assert len(sel.options) == 4
    assert len(set(sel.options)) == 4  # no dup options
    assert set(sel.options) <= set(SET)


def test_registry_and_default():
    keys = {s["key"] for s in L.list_strategies()}
    assert {"uniform_random", "leitner", "sm2", "confusion_weighted", "fsrs_lite"} <= keys
    assert L.DEFAULT_STRATEGY == "confusion_weighted"
    with pytest.raises(Exception):
        L.make_strategy("nonexistent")  # learny raises rather than falling back


def test_uniform_anti_repeat():
    strat = L.make_strategy("uniform_random")
    hist = [_rec("a", True)]
    targets = [_next(strat, hist, SET, seed=i).target for i in range(50)]
    assert "a" not in targets  # never repeats the immediately-previous target


def test_empirical_confusion_matrix():
    hist = [_rec("a", False, chosen="b", days_ago=1)] * 3 + [_rec("a", True, days_ago=1)]
    conf = confusion_weights(L.to_responses(hist), now=NOW.timestamp(), symmetric=False)
    assert "a" in conf and conf["a"].get("b", 0) > 0
    assert "b" not in conf  # b was never a (correct) target that got missed


def test_confusion_weighted_targets_confused_items():
    strat = L.make_strategy("confusion_weighted", epsilon_explore=0.0)
    # "a" is repeatedly confused with "b"; others answered correctly
    hist = [_rec("a", False, chosen="b", days_ago=1) for _ in range(6)]
    hist += [_rec(k, True, days_ago=1) for k in ("c", "d", "e", "f") for _ in range(2)]
    picks = [_next(strat, hist, SET, seed=i).target for i in range(60)]
    assert picks.count("a") > picks.count("c")  # confused item surfaces more
    # and "b" (the confuser) should appear as a distractor when "a" is the target
    sels = [_next(strat, hist, SET, seed=i) for i in range(60)]
    a_problems = [s for s in sels if s.target == "a"]
    assert any("b" in s.options for s in a_problems)


def test_leitner_promotes_and_demotes():
    strat = L.make_strategy("leitner")
    # "a" always wrong (stays box 1), "b" always right (climbs)
    hist = [_rec("a", False, days_ago=0.01) for _ in range(3)]
    hist += [_rec("b", True, days_ago=0.01) for _ in range(3)]
    boxes = {k: b for k, (b, _) in strat.boxes(L.to_responses(hist)).items()}
    assert boxes["a"] == 1 and boxes["b"] > boxes["a"]


def test_confusion_weighted_merges_pose_after_a_mistake():
    # regression: a logged confusion must NOT replace pose-similarity for other targets
    pose = {"c": {"d": 10.0}}  # c is pose-confusable with d
    strat = L.make_strategy("confusion_weighted", epsilon_explore=0.0)
    hist = [_rec("a", False, chosen="b", days_ago=1)]  # only a<->b confused so far
    sels = [_next(strat, hist, ["c", "d", "e", "f", "g"], seed=i, similarity=pose)
            for i in range(40)]
    c_sels = [s for s in sels if s.target == "c"]
    assert c_sels and all("d" in s.options for s in c_sels)  # pose-confusable d still surfaces


def test_sm2_and_fsrs_do_not_degenerate_to_one_item():
    hist = [_rec(k, True, days_ago=2) for k in ("a", "b", "c", "d")]  # all seen, equal-ish state
    for key in ("sm2", "fsrs_lite"):
        strat = L.make_strategy(key)
        picks = {_next(strat, hist, ["a", "b", "c", "d"], seed=i).target
                 for i in range(30)}
        assert len(picks) > 1, f"{key} repeats a single item"


def test_sm2_grade_and_unseen_priority():
    strat = L.make_strategy("sm2")
    assert strat.grade(L.to_response(_rec("a", True, rt=1000))) == 5      # fast correct
    assert strat.grade(L.to_response(_rec("a", True, rt=12000))) == 3     # slow correct
    assert strat.grade(L.to_response({**_rec("a", False), "timed_out": True})) == 0
    # unseen items are prioritized over a freshly-seen one
    hist = [_rec("a", True, days_ago=0.01)]
    picks = {_next(strat, hist, ["a", "b", "c", "d"], seed=i).target for i in range(20)}
    assert "a" not in picks  # a was just seen; unseen b/c/d chosen


# --- #22: generalized recording model (item_key / content_domain) ------------- #

def test_item_key_falls_back_to_target_key():
    """Legacy rows (no item_key) aggregate exactly as before under target_key."""
    legacy = _rec("a", True)  # no item_key field
    assert L.to_response(legacy).item == "a"
    assert L.item_accuracy([legacy])["a"]["n"] == 1


def test_events_keyed_on_item_key_for_word_rows():
    """Word rows aggregate by their item_key (the word slug), not target_key."""
    word_row = {**_rec("seoinage", True), "content_domain": "word", "item_key": "nage"}
    assert L.to_response(word_row).item == "nage"
    stats = L.item_accuracy([word_row])
    assert stats["nage"]["n"] == 1 and stats["nage"]["last_correct"] is True


def test_word_rows_do_not_disturb_throw_selection():
    """A strategy fed mixed history still selects only from the throw study set."""
    strat = L.make_strategy("confusion_weighted", epsilon_explore=0.0)
    word_noise = [{**_rec("w1", False, chosen="w2"), "content_domain": "word", "item_key": "w1"}]
    throw_hist = [_rec("a", False, chosen="b", days_ago=1)]
    sel = _next(strat, word_noise + throw_hist, SET, seed=0)
    assert sel.target in SET
    assert set(sel.options) <= set(SET)


def test_adapter_maps_record_fields_to_response():
    rec = {**_rec("a", False, chosen="b", score=0.4, rt=1500), "strategy_key": "sm2"}
    r = L.to_response({**rec, "user": "ann"})
    assert (r.student, r.item, r.outcome.value) == ("ann", "a", "wrong")
    assert r.at == pytest.approx(NOW.timestamp())
    assert r.meta["chosen"] == "b" and r.meta["rt_ms"] == 1500
    assert r.meta["credit"] == 0.4 and r.score is None  # credit is not the ordered score
    assert r.meta["shown"] == ["a", "x", "y", "z"]
    assert (r.meta["session"], r.meta["spec"], r.meta["pattern"]) == ("s1", "sm2", "M1")
