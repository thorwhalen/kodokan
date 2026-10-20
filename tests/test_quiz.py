"""The quiz service: issue, answer, history, adaptivity, and its stores (all offline, synthetic catalog)."""

import random
from datetime import datetime, timedelta, timezone

import pytest

from kodokan import learning
from kodokan.quiz import Quiz, quiz_stores


def _catalog(n=6):
    return {f"t{i}": {"name": f"Technique {i}", "clips": [{"video_id": f"v{i}"}]} for i in range(n)}


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=30)
        return self.t


def _quiz(**kw):
    return Quiz(_catalog(), responses={}, problems={}, now=Clock(), **kw)


def test_a_problem_is_issued_then_answered_once_into_the_record_schema():
    quiz = _quiz()
    problem = quiz.next_problem("ana", session_id="s1", rng=random.Random(1))
    assert len(quiz.problems) == 1 and quiz.history("ana") == []
    record = quiz.submit(problem.problem_id, problem.correct_index, response_time_ms=2500)
    assert set(learning.RECORD_FIELDS) <= set(record)
    assert record["correct"] and record["score"] == 1.0 and record["session_id"] == "s1"
    assert record["ts_presented"] < record["ts_answered"] and not quiz.problems
    assert quiz.history("ana") == [record]
    with pytest.raises(KeyError):
        quiz.submit(problem.problem_id, 0)  # answered once


def test_a_wrong_answer_and_a_timeout_are_recorded():
    quiz = _quiz()
    problem = quiz.next_problem("ana", rng=random.Random(2))
    wrong = (problem.correct_index + 1) % len(problem.choices)
    record = quiz.submit(problem.problem_id, wrong)
    assert not record["correct"] and record["chosen_key"] == problem.choices[wrong]["key"] and not record["timed_out"]
    late = quiz.submit(quiz.next_problem("ana").problem_id, None, timed_out=True)
    assert late["timed_out"] and late["chosen_key"] is None and late["score"] == 0.0 and not late["correct"]
    assert quiz.stats("ana")[problem.target_key]["wrong"] >= 1


def test_history_is_per_user():
    quiz = _quiz()
    for user in ("ana", "ana", "bo"):
        p = quiz.next_problem(user)
        quiz.submit(p.problem_id, p.correct_index)
    assert len(quiz.history("ana")) == 2 and len(quiz.history("bo")) == 1 and quiz.history("cy") == []


def test_the_chosen_strategy_and_study_set_are_honoured():
    quiz = _quiz()
    seen = set()
    for _ in range(12):
        p = quiz.next_problem("ana", strategy="leitner", study_set=["t0", "t1", "t2"], mode="name_to_video")
        seen.add(p.target_key)
        assert {c["key"] for c in p.choices} <= {"t0", "t1", "t2"} and all("clip" in c for c in p.choices)
        record = quiz.submit(p.problem_id, p.correct_index)
        assert record["strategy_key"] == "leitner"
    assert seen <= {"t0", "t1", "t2"}
    with pytest.raises(ValueError):
        quiz.next_problem("ana", study_set=["nope"])


@pytest.mark.parametrize("user", ["", "a/b", "../x", "a__b", "x" * 65, " ana"])
def test_user_ids_that_could_escape_the_store_are_refused(user):
    with pytest.raises(ValueError):
        _quiz().next_problem(user)


def test_a_bad_choice_index_is_refused_and_leaves_the_problem_open():
    quiz = _quiz()
    p = quiz.next_problem("ana")
    with pytest.raises(ValueError):
        quiz.submit(p.problem_id, 99)
    assert quiz.submit(p.problem_id, p.correct_index)["correct"]


def test_state_survives_a_new_quiz_over_the_same_directory(tmp_path):
    pytest.importorskip("dol")
    responses, problems = quiz_stores(tmp_path)
    first = Quiz(_catalog(), responses=responses, problems=problems)
    open_problem = first.next_problem("ana")
    done = first.next_problem("ana")
    first.submit(done.problem_id, done.correct_index)
    second = Quiz(_catalog(), responses=quiz_stores(tmp_path)[0], problems=quiz_stores(tmp_path)[1])
    assert len(second.history("ana")) == 1
    assert second.submit(open_problem.problem_id, open_problem.correct_index)["correct"]
    assert len(second.history("ana")) == 2


def test_strategies_are_listed_for_a_settings_screen():
    assert {s["key"] for s in Quiz.strategies()} >= {"leitner", "sm2"}
