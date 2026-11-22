"""The flashcard quiz's learning engine: kodokan's records, learny's strategies.

Which technique to ask next, and which distractors to show, is learny's job
(:mod:`learny.scheduling`, :mod:`learny.distractors`): Leitner, SM-2, FSRS-lite,
confusion-weighted and uniform-random strategies, replayed from a response log. This
module is the thin adapter that stays in kodokan: it knows kodokan's logged record
dicts (:data:`RECORD_FIELDS`) and turns them into :class:`learny.tracing.Response`
objects, so the quiz keeps storing plain JSON rows while learny reads its own type.

Record schema (one dict per answered problem)::

    response_id, problem_id, user, session_id, strategy_key,
    target_key, mode, choice_keys, chosen_key,
    correct, score, timed_out, response_time_ms,
    ts_presented, ts_answered            # ISO-8601 UTC

>>> rec = {"user": "ann", "target_key": "o-soto-gari", "correct": False,
...        "chosen_key": "o-uchi-gari", "score": 0.4,
...        "choice_keys": ["o-soto-gari", "o-uchi-gari"],
...        "ts_answered": "2026-06-25T12:00:00+00:00"}
>>> r = to_response(rec)
>>> r.item, r.outcome.value, r.meta["chosen"], r.meta["credit"]
('o-soto-gari', 'wrong', 'o-uchi-gari', 0.4)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from learny.scheduling import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    Selection,
    Strategy,
    list_strategies,
    make_strategy,
)
from learny.tracing import Response

__all__ = [
    "RECORD_FIELDS",
    "DEFAULT_STRATEGY",
    "STRATEGIES",
    "Selection",
    "Strategy",
    "list_strategies",
    "make_strategy",
    "item_accuracy",
    "to_response",
    "to_responses",
]

RECORD_FIELDS = (
    "response_id",
    "problem_id",
    "user",
    "session_id",
    "strategy_key",
    "target_key",
    "mode",
    "content_domain",  # "throw" | "word" (default "throw" for legacy rows)
    "item_key",  # generic per-item key (throw or word slug); == target_key for throws
    "choice_keys",
    "chosen_key",
    "correct",
    "score",
    "timed_out",
    "response_time_ms",
    "ts_presented",
    "ts_answered",
)

#: The ``meta['pattern']`` a kodokan quiz round is: pick the match among k options.
CHOICE_PATTERN = "M1"


def _posix(ts) -> float | None:
    """ISO-8601 text (``Z`` allowed) to POSIX seconds; ``None`` when absent or unparsable."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _item_key(record) -> str | None:
    """The throw or vocab word a row is about; ``target_key`` for legacy rows."""
    return record.get("item_key") or record.get("target_key")


def to_response(record: dict) -> Response:
    """Adapt one logged record dict to a :class:`learny.tracing.Response`.

    ``score`` (kodokan's float near-miss credit) goes to ``meta['credit']``, not to
    ``Response.score``, which is the learner model's ordered score.
    """
    meta = {
        key: value
        for key, value in {
            "shown": list(record["choice_keys"]) if record.get("choice_keys") else None,
            "chosen": record.get("chosen_key"),
            "rt_ms": record.get("response_time_ms"),
            "timed_out": record.get("timed_out"),
            "session": record.get("session_id"),
            "spec": record.get("strategy_key"),
            "credit": record.get("score"),
            "pattern": CHOICE_PATTERN,
        }.items()
        if value is not None
    }
    return Response(
        record.get("user") or "",
        _item_key(record) or "",
        bool(record.get("correct")),
        at=_posix(record.get("ts_answered") or record.get("ts_presented")),
        meta=meta,
    )


def to_responses(history: Iterable[dict]) -> list[Response]:
    """Adapt a list of record dicts, dropping rows about no item."""
    return [to_response(r) for r in history if _item_key(r)]


def item_accuracy(history) -> dict[str, dict]:
    """Per-item running stats: ``{key: {n, wrong, last_ts, last_correct}}``."""
    by: dict[str, list[dict]] = {}
    for r in history:
        if k := _item_key(r):
            by.setdefault(k, []).append(r)
    stats = {}
    for k, evs in by.items():
        evs.sort(key=lambda r: r.get("ts_answered") or r.get("ts_presented") or "")
        stats[k] = {
            "n": len(evs),
            "wrong": sum(1 for r in evs if not r.get("correct")),
            "last_ts": evs[-1].get("ts_answered"),
            "last_correct": bool(evs[-1].get("correct")),
        }
    return stats
