"""Adaptive learning engine for the flashcard quiz — selectable strategies.

A :class:`Strategy` maps the recorded response **history** to the next **problem**
— which technique to ask (target) and which distractors to show — both drawn from
an active study *set*. Strategies are research-backed (spaced repetition + MCQ
learning science; see ``misc/docs/learning-research.md``) and the user picks one in
settings. The function from recorded data → next question is therefore a swappable
strategy (open/closed): add a class, register it, and it appears in the UI.

Design notes
------------
- History is a list of plain response dicts (the logged record schema below), so the
  engine has no storage dependency and is trivially testable.
- Per-item adaptive state (Leitner box, SM-2 ease, FSRS stability) is reconstructed by
  *replaying* the history — no mutable server state to corrupt.
- Spacing intervals are in **days**; within a single dense session most items aren't
  "due", so every scheduler falls back to an urgency ranking (most-overdue / lowest
  recall / box weight) instead of stalling.

Record schema (one dict per answered problem)::

    response_id, problem_id, user, session_id, strategy_key,
    target_key, mode, choice_keys, chosen_key,
    correct, score, timed_out, response_time_ms,
    ts_presented, ts_answered            # ISO-8601 UTC
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

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


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_since(ts, now: datetime) -> float:
    t = _parse_ts(ts)
    return (now - t).total_seconds() / 86400.0 if t else math.inf


@dataclass
class Selection:
    """A chosen target plus the full ordered option set (includes the target)."""

    target_key: str
    choice_keys: list[str]


# --------------------------------------------------------------------------- #
# History helpers (operate on the logged record dicts)
# --------------------------------------------------------------------------- #


def _item_key(r) -> str | None:
    """Generic per-item key: the throw or vocab word a row is about.

    Falls back to ``target_key`` for legacy rows logged before the word games existed,
    so existing throw history aggregates exactly as before.
    """
    return r.get("item_key") or r.get("target_key")


def _events_by_item(history) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in history:
        k = _item_key(r)
        if k:
            by[k].append(r)
    for evs in by.values():
        evs.sort(key=lambda r: r.get("ts_answered") or r.get("ts_presented") or "")
    return by


def empirical_confusion(
    history, *, halflife_days: float = 7.0, now: datetime | None = None
) -> dict:
    """Per-learner confusion matrix from wrong answers: ``{correct: {chosen_wrong: weight}}``.

    Each mistake contributes a recency-decayed, partial-credit-shortfall weight, so the
    throws a learner *actually* mixes up (recently) dominate.
    """
    now = now or datetime.now(timezone.utc)
    conf: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in history:
        if r.get("correct") or not r.get("chosen_key"):
            continue
        correct_k, chosen_k = r.get("target_key"), r["chosen_key"]
        if not correct_k or chosen_k == correct_k:
            continue
        days = _days_since(r.get("ts_answered"), now)
        decay = 0.5 ** (days / max(halflife_days, 1e-6)) if math.isfinite(days) else 1.0
        shortfall = 1.0 - float(r.get("score") or 0.0)
        conf[correct_k][chosen_k] += decay * max(shortfall, 0.25)
    return {k: dict(v) for k, v in conf.items()}


def _symmetric_confusion(history, **kw) -> dict:
    """Symmetric confusability view of the directional confusion matrix.

    Perceptual confusability is symmetric (if A is mistaken for B, B↔A are both hard to
    tell apart), so for *distractor* selection we fold the directional matrix together.
    """
    conf = empirical_confusion(history, **kw)
    sym: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for a, row in conf.items():
        for b, w in row.items():
            sym[a][b] += w
            sym[b][a] += w
    return {k: dict(v) for k, v in sym.items()}


def merge_similarity(
    base: dict | None, overlay: dict | None, *, boost: float = 2.0
) -> dict:
    """Additively merge two ``target -> {key: weight}`` matrices (overlay boosted).

    Keeps the dense ``base`` (pose-shape) for every target and *adds* the sparse
    ``overlay`` (per-learner confusion) on top — so confusable distractors are always
    available, with the learner's personal confusers weighted up rather than replacing
    pose-similarity for not-yet-confused targets.
    """
    out = {k: dict(v) for k, v in (base or {}).items()}
    for k, row in (overlay or {}).items():
        d = out.setdefault(k, {})
        for j, w in row.items():
            d[j] = d.get(j, 0.0) + boost * w
    return out


def item_accuracy(history) -> dict[str, dict]:
    """Per-item running stats: ``{key: {n, wrong, last_ts, last_correct}}``."""
    stats: dict[str, dict] = {}
    for k, evs in _events_by_item(history).items():
        wrong = sum(1 for r in evs if not r.get("correct"))
        stats[k] = {
            "n": len(evs),
            "wrong": wrong,
            "last_ts": evs[-1].get("ts_answered"),
            "last_correct": bool(evs[-1].get("correct")),
        }
    return stats


# --------------------------------------------------------------------------- #
# Distractor selection (shared)
# --------------------------------------------------------------------------- #


def _weighted_sample(items, weights, k, rng):
    items, weights = list(items), [max(float(w), 1e-9) for w in weights]
    out = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                out.append(items.pop(i))
                weights.pop(i)
                break
    return out


def pick_distractors(target, study_set, *, n, similarity=None, rng, top_up=True):
    """Pick ``n`` distractors from ``study_set``, preferring confusable ones.

    ``similarity`` maps ``target -> {key: weight}`` (pose-shape or empirical confusion).
    Falls back to uniform when absent; tops up from the rest of the set if the
    confusable pool is too small (so the answer set always has ``n`` options when possible).
    """
    pool = [k for k in study_set if k != target]
    if not pool:
        return []
    sims = (similarity or {}).get(target, {})
    weights = [sims.get(k, 0.0) for k in pool]
    chosen = _weighted_sample(pool, weights, n, rng) if any(weights) else []
    if top_up and len(chosen) < n:
        rest = [k for k in pool if k not in chosen]
        rng.shuffle(rest)
        chosen += rest[: n - len(chosen)]
    return chosen[:n]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


@dataclass
class Strategy:
    """Base strategy: target selection is overridden; distractors default to confusable."""

    key = "base"
    name = "Base"
    description = ""
    n_choices: int = 4

    def pick_target(self, history, study_set, *, now, rng) -> str:
        raise NotImplementedError

    def distractor_similarity(self, history, *, now):
        return None  # subclasses may return a similarity matrix; else caller's pose-sim

    def next_selection(
        self,
        history,
        study_set,
        *,
        mode="video_to_name",
        similarity=None,
        now=None,
        rng=None,
    ) -> Selection:
        now = now or datetime.now(timezone.utc)
        rng = rng or random.Random()
        study_set = [k for k in dict.fromkeys(study_set)]  # dedupe, keep order
        if not study_set:
            raise ValueError("empty study set")
        target = self.pick_target(history, study_set, now=now, rng=rng)
        # merge per-learner confusion ON TOP OF pose-similarity (never replace it), so
        # confusable distractors are available even for not-yet-confused targets.
        sim = merge_similarity(similarity, self.distractor_similarity(history, now=now))
        distractors = pick_distractors(
            target, study_set, n=self.n_choices - 1, similarity=sim, rng=rng
        )
        options = [*distractors, target]
        rng.shuffle(options)
        return Selection(target_key=target, choice_keys=options)


@dataclass
class UniformRandom(Strategy):
    key = "uniform_random"
    name = "Uniform random"
    description = "No memory — every throw in the set is equally likely. The baseline."
    anti_repeat: bool = True

    def pick_target(self, history, study_set, *, now, rng):
        last = history[-1]["target_key"] if history else None
        pool = (
            [k for k in study_set if k != last]
            if (self.anti_repeat and len(study_set) > 1)
            else list(study_set)
        )
        return rng.choice(pool)


@dataclass
class Leitner(Strategy):
    key = "leitner"
    name = "Leitner boxes"
    description = "Spaced repetition by boxes: a right answer moves a throw up a box (seen less often), a wrong answer drops it back."
    n_boxes: int = 5
    intervals: tuple = (1, 2, 4, 8, 16)  # days per box; auto-extended to n_boxes
    demotion: str = "reset"  # "reset" -> box 1, "step" -> box-1

    def __post_init__(self):
        iv = list(self.intervals)
        while (
            len(iv) < self.n_boxes
        ):  # keep box/interval invariant even if n_boxes overridden
            iv.append(iv[-1] * 2 if iv else 1)
        self.intervals = tuple(iv)

    def _boxes(self, history):
        by = _events_by_item(history)
        boxes, last = {}, {}
        for k, evs in by.items():
            b = 1
            for r in evs:
                if r.get("correct"):
                    b = min(b + 1, self.n_boxes)
                else:
                    b = 1 if self.demotion == "reset" else max(b - 1, 1)
            boxes[k], last[k] = b, evs[-1].get("ts_answered")
        return boxes, last

    def pick_target(self, history, study_set, *, now, rng):
        boxes, last = self._boxes(history)
        intervals = self.intervals[: self.n_boxes]
        due, urgency = [], []
        for k in study_set:
            b = boxes.get(k, 1)
            elapsed = _days_since(last.get(k), now) if k in last else math.inf
            if elapsed >= intervals[min(b, len(intervals)) - 1]:
                due.append(k)
                urgency.append(self.n_boxes - b + 1)  # lower boxes weigh more
        if due:
            return _weighted_sample(due, urgency, 1, rng)[0]
        # nothing due this session -> review the lowest box (least learned)
        cand = sorted(study_set, key=lambda k: boxes.get(k, 1))
        floor = boxes.get(cand[0], 1)
        return rng.choice([k for k in cand if boxes.get(k, 1) == floor])


@dataclass
class SM2(Strategy):
    key = "sm2"
    name = "SuperMemo SM-2"
    description = "Classic spaced repetition: each throw gets a personal ease factor; correct answers stretch the interval, mistakes reset it."
    ef_start: float = 2.5
    ef_min: float = 1.3
    i1: int = 1
    i2: int = 6
    fast_ms: int = 3500
    slow_ms: int = 9000

    def _grade(self, r) -> int:
        if r.get("timed_out"):
            return (
                3 if r.get("correct") else 0
            )  # correct-but-slow is a weak pass, not a lapse
        rt = r.get("response_time_ms")
        if r.get("correct"):
            if rt is not None and rt <= self.fast_ms:
                return 5
            if rt is not None and rt >= self.slow_ms:
                return 3
            return 4
        return (
            2 if float(r.get("score") or 0) > 0 else 1
        )  # confusable miss vs clear miss

    def _state(self, history):
        st = {}
        for k, evs in _events_by_item(history).items():
            ef, n, interval = self.ef_start, 0, 0
            for r in evs:
                q = self._grade(r)
                ef = max(self.ef_min, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
                if q >= 3:
                    interval = (
                        self.i1
                        if n == 0
                        else (self.i2 if n == 1 else round(interval * ef))
                    )
                    n += 1
                else:
                    n, interval = 0, self.i1
            st[k] = {
                "ef": ef,
                "n": n,
                "I": interval,
                "last": evs[-1].get("ts_answered"),
            }
        return st

    def pick_target(self, history, study_set, *, now, rng):
        st = self._state(history)
        unseen = [k for k in study_set if k not in st]
        if unseen:
            return rng.choice(unseen)
        # sample weighted by overdue-ness (not a hard argmax) so a dense session
        # interleaves rather than repeating one item; more overdue -> higher weight.
        overdue = {k: _days_since(st[k]["last"], now) - st[k]["I"] for k in study_set}
        lo = min(overdue.values())
        weights = [overdue[k] - lo + 0.1 for k in study_set]
        return _weighted_sample(study_set, weights, 1, rng)[0]


@dataclass
class ConfusionWeighted(Strategy):
    key = "confusion_weighted"
    name = "Confusion-weighted"
    description = "Focuses on the throws you actually mix up, and pits them against the very throws you confuse them with. Recommended."
    recency_halflife_days: float = 7.0
    epsilon_explore: float = 0.1
    prior: float = 1.0

    def _confusion_score(self, history, now):
        conf = empirical_confusion(
            history, halflife_days=self.recency_halflife_days, now=now
        )
        return {k: sum(v.values()) for k, v in conf.items()}

    def pick_target(self, history, study_set, *, now, rng):
        if rng.random() < self.epsilon_explore or not history:
            return rng.choice(study_set)  # explore / cold start
        cscore = self._confusion_score(history, now)
        seen = item_accuracy(history)
        weights = []
        for k in study_set:
            unseen_bonus = self.prior if k not in seen else 0.0
            weights.append(cscore.get(k, 0.0) + unseen_bonus + 0.05)
        return _weighted_sample(study_set, weights, 1, rng)[0]

    def distractor_similarity(self, history, *, now):
        # symmetric per-learner confusion pairs, merged over pose-sim by next_selection
        conf = _symmetric_confusion(
            history, halflife_days=self.recency_halflife_days, now=now
        )
        return conf or None


@dataclass
class FSRSLite(Strategy):
    key = "fsrs_lite"
    name = "FSRS-lite (memory model)"
    description = "Models how memory of each throw decays and reschedules it just before you'd forget (target ~90% recall)."
    target_retention: float = 0.9
    s_init: float = 1.0
    stability_growth: float = 2.0
    lapse_floor: float = 0.5

    def _state(self, history):
        st = {}
        for k, evs in _events_by_item(history).items():
            s = self.s_init
            for r in evs:
                if r.get("correct"):
                    s *= self.stability_growth
                else:
                    s = max(self.lapse_floor, s * 0.5)
            st[k] = {"S": s, "last": evs[-1].get("ts_answered")}
        return st

    def _recall(self, k, st, now):
        if k not in st:
            return 0.0
        return math.exp(-_days_since(st[k]["last"], now) / max(st[k]["S"], 1e-6))

    def pick_target(self, history, study_set, *, now, rng):
        st = self._state(history)
        unseen = [k for k in study_set if k not in st]
        if unseen:
            return rng.choice(unseen)
        # sample weighted by forgetting (1 - recall), not a hard argmin, to interleave
        weights = [1.0 - self._recall(k, st, now) + 0.05 for k in study_set]
        return _weighted_sample(study_set, weights, 1, rng)[0]

    def distractor_similarity(self, history, *, now):
        conf = _symmetric_confusion(history, now=now)
        return conf or None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_STRATEGY_CLASSES = (UniformRandom, Leitner, SM2, ConfusionWeighted, FSRSLite)
STRATEGIES = {c.key: c for c in _STRATEGY_CLASSES}
DEFAULT_STRATEGY = ConfusionWeighted.key


def make_strategy(key: str | None = None, **params) -> Strategy:
    """Instantiate a strategy by key (default = confusion-weighted); unknown params ignored."""
    from dataclasses import fields

    cls = STRATEGIES.get(key or DEFAULT_STRATEGY, STRATEGIES[DEFAULT_STRATEGY])
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in params.items() if k in valid})


def list_strategies() -> list[dict]:
    """Describe available strategies (for the settings UI)."""
    out = []
    for key, cls in STRATEGIES.items():
        out.append(
            {
                "key": key,
                "name": cls.name,
                "description": cls.description,
                "is_default": key == DEFAULT_STRATEGY,
            }
        )
    return out
