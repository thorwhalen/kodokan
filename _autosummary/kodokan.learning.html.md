# kodokan.learning

Adaptive learning engine for the flashcard quiz — selectable strategies.

A [`Strategy`](#kodokan.learning.Strategy) maps the recorded response **history** to the next **problem**
— which technique to ask (target) and which distractors to show — both drawn from
an active study *set*. Strategies are research-backed (spaced repetition + MCQ
learning science; see `misc/docs/learning-research.md`) and the user picks one in
settings. The function from recorded data → next question is therefore a swappable
strategy (open/closed): add a class, register it, and it appears in the UI.

## Design notes

- History is a list of plain response dicts (the logged record schema below), so the
  engine has no storage dependency and is trivially testable.
- Per-item adaptive state (Leitner box, SM-2 ease, FSRS stability) is reconstructed by
  *replaying* the history — no mutable server state to corrupt.
- Spacing intervals are in **days**; within a single dense session most items aren’t
  “due”, so every scheduler falls back to an urgency ranking (most-overdue / lowest
  recall / box weight) instead of stalling.

Record schema (one dict per answered problem):

```default
response_id, problem_id, user, session_id, strategy_key,
target_key, mode, choice_keys, chosen_key,
correct, score, timed_out, response_time_ms,
ts_presented, ts_answered            # ISO-8601 UTC
```

### Functions

| [`empirical_confusion`](#kodokan.learning.empirical_confusion)(history, \*[, ...])           | Per-learner confusion matrix from wrong answers: `{correct: {chosen_wrong: weight}}`.   |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`item_accuracy`](#kodokan.learning.item_accuracy)(history)                            | Per-item running stats: `{key: {n, wrong, last_ts, last_correct}}`.                     |
| [`list_strategies`](#kodokan.learning.list_strategies)()                                 | Describe available strategies (for the settings UI).                                    |
| [`make_strategy`](#kodokan.learning.make_strategy)([key])                              | Instantiate a strategy by key (default = confusion-weighted); unknown params ignored.   |
| [`merge_similarity`](#kodokan.learning.merge_similarity)(base, overlay, \*[, boost])      | Additively merge two `target -> {key: weight}` matrices (overlay boosted).              |
| [`pick_distractors`](#kodokan.learning.pick_distractors)(target, study_set, \*, n[, ...]) | Pick `n` distractors from `study_set`, preferring confusable ones.                      |

### Classes

| [`ConfusionWeighted`](#kodokan.learning.ConfusionWeighted)([n_choices, ...])             |                                                                                   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`FSRSLite`](#kodokan.learning.FSRSLite)([n_choices, target_retention, ...])    |                                                                                   |
| [`Leitner`](#kodokan.learning.Leitner)([n_choices, n_boxes, intervals, ...])   |                                                                                   |
| [`SM2`](#kodokan.learning.SM2)([n_choices, ef_start, ef_min, i1, i2, ...]) |                                                                                   |
| [`Selection`](#kodokan.learning.Selection)(target_key, choice_keys)              | A chosen target plus the full ordered option set (includes the target).           |
| [`Strategy`](#kodokan.learning.Strategy)([n_choices])                           | Base strategy: target selection is overridden; distractors default to confusable. |
| [`UniformRandom`](#kodokan.learning.UniformRandom)([n_choices, anti_repeat])         |                                                                                   |

### *class* kodokan.learning.ConfusionWeighted(n_choices=4, recency_halflife_days=7.0, epsilon_explore=0.1, prior=1.0)

Bases: [`Strategy`](#kodokan.learning.Strategy)

### *class* kodokan.learning.FSRSLite(n_choices=4, target_retention=0.9, s_init=1.0, stability_growth=2.0, lapse_floor=0.5)

Bases: [`Strategy`](#kodokan.learning.Strategy)

### *class* kodokan.learning.Leitner(n_choices=4, n_boxes=5, intervals=(1, 2, 4, 8, 16), demotion='reset')

Bases: [`Strategy`](#kodokan.learning.Strategy)

### *class* kodokan.learning.SM2(n_choices=4, ef_start=2.5, ef_min=1.3, i1=1, i2=6, fast_ms=3500, slow_ms=9000)

Bases: [`Strategy`](#kodokan.learning.Strategy)

### *class* kodokan.learning.Selection(target_key, choice_keys)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A chosen target plus the full ordered option set (includes the target).

### *class* kodokan.learning.Strategy(n_choices=4)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base strategy: target selection is overridden; distractors default to confusable.

### *class* kodokan.learning.UniformRandom(n_choices=4, anti_repeat=True)

Bases: [`Strategy`](#kodokan.learning.Strategy)

### kodokan.learning.empirical_confusion(history, , halflife_days=7.0, now=None)

Per-learner confusion matrix from wrong answers: `{correct: {chosen_wrong: weight}}`.

Each mistake contributes a recency-decayed, partial-credit-shortfall weight, so the
throws a learner *actually* mixes up (recently) dominate.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.learning.item_accuracy(history)

Per-item running stats: `{key: {n, wrong, last_ts, last_correct}}`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.learning.list_strategies()

Describe available strategies (for the settings UI).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.learning.make_strategy(key=None, \*\*params)

Instantiate a strategy by key (default = confusion-weighted); unknown params ignored.

* **Return type:**
  [`Strategy`](#kodokan.learning.Strategy)

### kodokan.learning.merge_similarity(base, overlay, , boost=2.0)

Additively merge two `target -> {key: weight}` matrices (overlay boosted).

Keeps the dense `base` (pose-shape) for every target and *adds* the sparse
`overlay` (per-learner confusion) on top — so confusable distractors are always
available, with the learner’s personal confusers weighted up rather than replacing
pose-similarity for not-yet-confused targets.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.learning.pick_distractors(target, study_set, , n, similarity=None, rng, top_up=True)

Pick `n` distractors from `study_set`, preferring confusable ones.

`similarity` maps `target -> {key: weight}` (pose-shape or empirical confusion).
Falls back to uniform when absent; tops up from the rest of the set if the
confusable pool is too small (so the answer set always has `n` options when possible).
