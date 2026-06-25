"""Tests for the flashcard learning-game logic (pure + storage)."""

import random

import pytest

from kodokan import flashcards as fc


def _catalog(n=5):
    return {
        f"t{i}": {"name": f"Technique {i}", "clips": [{"video_id": f"v{i}", "source": "s",
                                                       "demos": [{"start_s": 0.0, "end_s": 2.0}]}]}
        for i in range(n)
    }


def test_make_problem_video_to_name():
    cat = _catalog()
    p = fc.make_problem(cat, "t2", mode="video_to_name", n_choices=4, rng=random.Random(0))
    assert p.mode == "video_to_name"
    assert len(p.choices) == 4
    assert p.choices[p.correct_index]["key"] == "t2"
    assert "clip" in p.prompt and p.is_correct(p.correct_index)


def test_make_problem_name_to_video():
    cat = _catalog()
    p = fc.make_problem(cat, "t1", mode="name_to_video", n_choices=3, rng=random.Random(1))
    assert p.prompt["name"] == "Technique 1"
    assert len(p.choices) == 3 and all("clip" in c for c in p.choices)
    assert p.choices[p.correct_index]["key"] == "t1"


def test_confusable_distractors_prefers_similar():
    keys = [f"t{i}" for i in range(6)]
    sim = {"t0": {"t1": 0.95, "t2": 0.9, "t3": 0.05, "t4": 0.05, "t5": 0.05}}
    rng = random.Random(0)
    picks = [tuple(fc.confusable_distractors("t0", keys, n=2, similarity=sim, rng=rng)) for _ in range(200)]
    flat = [k for pair in picks for k in pair]
    # the highly-similar distractors should dominate
    assert flat.count("t1") + flat.count("t2") > flat.count("t3") + flat.count("t4") + flat.count("t5")


def test_score_response_confusion_weighted():
    sim = {"a": {"b": 0.9, "c": 0.0}}
    assert fc.score_response("a", "a", similarity=sim) == 1.0
    assert fc.score_response("a", "b", similarity=sim) > fc.score_response("a", "c", similarity=sim)
    assert fc.score_response("a", "c", similarity=sim) == 0.0


def test_next_target_focus_and_weakness():
    keys = [f"t{i}" for i in range(4)]
    # t0 always wrong, t1 always right
    history = [{"target_key": "t0", "correct": False}] * 5 + [{"target_key": "t1", "correct": True}] * 5
    rng = random.Random(0)
    picks = [fc.next_target(keys, history, focus={"t0", "t1"}, rng=rng) for _ in range(200)]
    assert set(picks) <= {"t0", "t1"}  # focus respected
    assert picks.count("t0") > picks.count("t1")  # weakness prioritized


def test_log_response_roundtrip(tmp_path):
    pytest.importorskip("dol")
    from dol import JsonFiles

    cat = _catalog()
    p = fc.make_problem(cat, "t0", mode="video_to_name", rng=random.Random(0))
    store = JsonFiles(str(tmp_path))
    rec = fc.log_response(p, p.correct_index, store=store, timestamp="2026-06-25T00:00:00Z")
    assert rec["correct"] and rec["score"] == 1.0
    assert rec["target_key"] == "t0" and rec["chosen_key"] == "t0"
    assert any(k.endswith(".json") for k in store)
