"""Quiz service: the UI-agnostic orchestrator of the technique-recognition game (app #6).

``flashcards`` makes and scores one problem, ``learning`` chooses the next one from a
learner's history. This module wires them to storage so that any surface (a web app, a
CLI, a test) needs only three calls, all taking and returning plain data::

    quiz = Quiz(catalog, similarity=similarity)          # or Quiz.from_library()
    problem = quiz.next_problem("ana", strategy="sm2")   # issued problems are remembered
    record = quiz.submit(problem.problem_id, chosen_index=2, response_time_ms=3100)
    quiz.stats("ana")                                    # per-technique {n, wrong, ...}

State lives in two dict-like stores (seams; default: ``dol`` JSON files under the data
root, one per user prefix): the **issued problems** not yet answered, and the **responses**
(one record per answered problem, the ``learning.RECORD_FIELDS`` schema). A surface can
therefore be stateless between "give me a problem" and "here is the answer".

>>> catalog = {k: {"name": k.title(), "clips": []} for k in ("seoinage", "ogoshi", "tomoenage", "uchimata")}
>>> quiz = Quiz(catalog, responses={}, problems={})
>>> problem = quiz.next_problem("ana", rng=__import__("random").Random(0))
>>> record = quiz.submit(problem.problem_id, problem.correct_index)
>>> record["correct"], record["user"], quiz.stats("ana")[problem.target_key]["n"]
(True, 'ana', 1)
"""

from __future__ import annotations

import re
import uuid
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kodokan import learning
from kodokan.flashcards import Problem, problem_from_keys

PathLike = str | Path
#: A user id is a filesystem-safe token; ``__`` is the separator of the stored keys.
USER_RE = re.compile(r"^(?!.*__)[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CONTENT_DOMAIN = "throw"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _check_user(user: str) -> str:
    if not USER_RE.match(user or ""):
        raise ValueError(
            f"user must be 1-64 letters, digits, '.', '-' or '_' (no '__'), not {user!r}"
        )
    return user


def quiz_stores(
    directory: PathLike | None = None,
) -> tuple[MutableMapping, MutableMapping]:
    """``(responses, problems)`` as ``dol`` JSON-file stores under the data root."""
    from dol import JsonFiles

    from kodokan.config import data_dir

    root = Path(directory) if directory else data_dir() / "quiz"
    stores = []
    for name in ("responses", "problems"):
        (root / name).mkdir(parents=True, exist_ok=True)
        stores.append(JsonFiles(str(root / name)))
    return tuple(stores)  # type: ignore[return-value]


@dataclass
class Quiz:
    """The game for one ``catalog`` (``technique_key -> {"name", "clips"}``).

    :param similarity: ``target -> {key: weight}``, e.g. ``flashcards.build_confusability()``;
        the learner's own confusion is merged on top by the strategy.
    :param responses: answered-problem records, keyed ``<user>__<ts>__<response_id>.json``.
    :param problems: issued, unanswered problems, keyed ``<problem_id>.json``.
        Both default to :func:`quiz_stores`.
    :param clip_for: ``key -> clip`` for the media of a technique (default: its first clip).
    """

    catalog: dict
    similarity: dict | None = None
    responses: MutableMapping = None  # type: ignore[assignment]
    problems: MutableMapping = None  # type: ignore[assignment]
    clip_for: object = None
    now: object = field(default=_now, repr=False)

    def __post_init__(self):
        if self.responses is None or self.problems is None:
            responses, problems = quiz_stores()
            self.responses = responses if self.responses is None else self.responses
            self.problems = problems if self.problems is None else self.problems

    @classmethod
    def from_library(cls, **kwargs) -> "Quiz":
        """A quiz over the clip library on disk, with pose-shape confusability (needs ``kodokan[analysis]``)."""
        from kodokan.flashcards import build_catalog, build_confusability

        catalog = kwargs.pop("catalog", None) or build_catalog()
        kwargs.setdefault("similarity", build_confusability(catalog))
        return cls(catalog, **kwargs)

    # ------------------------------------------------------------------ reading

    @staticmethod
    def strategies() -> list[dict]:
        """The selectable strategies (for a settings screen)."""
        return learning.list_strategies()

    def history(self, user: str) -> list[dict]:
        """The user's answered-problem records, oldest first."""
        prefix = f"{_check_user(user)}__"
        records = [
            self.responses[k] for k in list(self.responses) if k.startswith(prefix)
        ]
        return sorted(records, key=lambda r: r.get("ts_answered") or "")

    def stats(self, user: str) -> dict[str, dict]:
        """Per-technique ``{n, wrong, last_ts, last_correct}`` for the user."""
        return learning.item_accuracy(self.history(user))

    # ------------------------------------------------------------------ playing

    def next_problem(
        self,
        user: str,
        *,
        strategy: str | None = None,
        mode: str = "video_to_name",
        study_set=None,
        session_id: str | None = None,
        rng=None,
    ) -> Problem:
        """Choose the next problem for ``user`` from their history, and remember it as issued.

        :param strategy: a key of :meth:`strategies` (default: confusion-weighted).
        :param study_set: the technique keys to draw from (default: the whole catalog).
        :raises ValueError: for an unknown user format, mode, or an empty study set.
        """
        history = self.history(user)
        keys = [k for k in (study_set or self.catalog) if k in self.catalog]
        chosen = learning.make_strategy(strategy)
        selection = chosen.next_selection(
            history,
            keys,
            mode=mode,
            similarity=self.similarity,
            now=self.now(),
            rng=rng,
        )
        problem = problem_from_keys(
            self.catalog,
            selection.target_key,
            selection.choice_keys,
            mode=mode,
            clip_for=self.clip_for,
            problem_id=f"{mode}-{selection.target_key}-{uuid.uuid4().hex[:12]}",
        )
        self.problems[f"{problem.problem_id}.json"] = {
            "problem": asdict(problem),
            "user": user,
            "session_id": session_id,
            "strategy_key": chosen.key,
            "ts_presented": self.now().isoformat(),
        }
        return problem

    def submit(
        self,
        problem_id: str,
        chosen_index: int | None,
        *,
        timed_out: bool = False,
        response_time_ms: int | None = None,
    ) -> dict:
        """Score an issued problem, store the record, and return it.

        ``chosen_index`` is None for a timeout. A problem is answered once.

        :raises KeyError: if ``problem_id`` was never issued, or was already answered.
        :raises ValueError: if ``chosen_index`` is not one of the choices.
        """
        from kodokan.flashcards import score_response

        key = f"{problem_id}.json"
        if key not in self.problems:
            raise KeyError(
                f"no open problem {problem_id!r}: never issued, or already answered"
            )
        issued = self.problems[key]
        problem = Problem(**issued["problem"])
        if chosen_index is not None and not 0 <= chosen_index < len(problem.choices):
            raise ValueError(
                f"chosen_index must be in 0..{len(problem.choices) - 1}, not {chosen_index}"
            )
        chosen_key = (
            None if chosen_index is None else problem.choices[chosen_index]["key"]
        )
        ts = self.now().isoformat()
        record = {
            "response_id": uuid.uuid4().hex,
            "problem_id": problem.problem_id,
            "user": issued["user"],
            "session_id": issued.get("session_id"),
            "strategy_key": issued["strategy_key"],
            "target_key": problem.target_key,
            "mode": problem.mode,
            "content_domain": CONTENT_DOMAIN,
            "item_key": problem.target_key,
            "choice_keys": [c["key"] for c in problem.choices],
            "chosen_key": chosen_key,
            "correct": chosen_key == problem.target_key,
            "score": 0.0
            if chosen_key is None
            else score_response(
                problem.target_key, chosen_key, similarity=self.similarity
            ),
            "timed_out": bool(timed_out or chosen_index is None),
            "response_time_ms": response_time_ms,
            "ts_presented": issued["ts_presented"],
            "ts_answered": ts,
        }
        self.responses[
            f"{issued['user']}__{ts}__{record['response_id']}.json".replace(":", "-")
        ] = record
        del self.problems[key]
        return record
