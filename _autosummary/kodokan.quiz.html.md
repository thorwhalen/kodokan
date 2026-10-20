# kodokan.quiz

Quiz service: the UI-agnostic orchestrator of the technique-recognition game (app #6).

`flashcards` makes and scores one problem, `learning` chooses the next one from a
learner’s history. This module wires them to storage so that any surface (a web app, a
CLI, a test) needs only three calls, all taking and returning plain data:

```default
quiz = Quiz(catalog, similarity=similarity)          # or Quiz.from_library()
problem = quiz.next_problem("ana", strategy="sm2")   # issued problems are remembered
record = quiz.submit(problem.problem_id, chosen_index=2, response_time_ms=3100)
quiz.stats("ana")                                    # per-technique {n, wrong, ...}
```

State lives in two dict-like stores (seams; default: `dol` JSON files under the data
root, one per user prefix): the **issued problems** not yet answered, and the **responses**
(one record per answered problem, the `learning.RECORD_FIELDS` schema). A surface can
therefore be stateless between “give me a problem” and “here is the answer”.

```pycon
>>> catalog = {k: {"name": k.title(), "clips": []} for k in ("seoinage", "ogoshi", "tomoenage", "uchimata")}
>>> quiz = Quiz(catalog, responses={}, problems={})
>>> problem = quiz.next_problem("ana", rng=__import__("random").Random(0))
>>> record = quiz.submit(problem.problem_id, problem.correct_index)
>>> record["correct"], record["user"], quiz.stats("ana")[problem.target_key]["n"]
(True, 'ana', 1)
```

### Module Attributes

| [`USER_RE`](#kodokan.quiz.USER_RE)   | A user id is a filesystem-safe token; `__` is the separator of the stored keys.   |
|------------------------------------------------------------|-----------------------------------------------------------------------------------|

### Functions

| [`quiz_stores`](#kodokan.quiz.quiz_stores)([directory])   | `(responses, problems)` as `dol` JSON-file stores under the data root.   |
|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|

### Classes

| [`Quiz`](#kodokan.quiz.Quiz)(catalog[, similarity, responses, ...])   | The game for one `catalog` (`technique_key -> {"name", "clips"}`).   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|

### *class* kodokan.quiz.Quiz(catalog, similarity=None, responses=None, problems=None, clip_for=None, now=<function \_now>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The game for one `catalog` (`technique_key -> {"name", "clips"}`).

* **Parameters:**
  * **similarity** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `target -> {key: weight}`, e.g. `flashcards.build_confusability()`;
    the learner’s own confusion is merged on top by the strategy.
  * **responses** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – answered-problem records, keyed `<user>__<ts>__<response_id>.json`.
  * **problems** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – issued, unanswered problems, keyed `<problem_id>.json`.
    Both default to [`quiz_stores()`](#kodokan.quiz.quiz_stores).
  * **clip_for** ([`object`](https://docs.python.org/3/builtins/functions.html#object)) – `key -> clip` for the media of a technique (default: its first clip).

#### *classmethod* from_library(\*\*kwargs)

A quiz over the clip library on disk (needs `kodokan[analysis]`).

`confusability`: `"pose"` (default, pose-shape proxy) or `"model"` (the
recognizer’s leave-one-clip-out confusions; weak until cross-source recognition
beats chance, #7). An explicit `similarity=` mapping wins over both.

* **Return type:**
  [`Quiz`](#kodokan.quiz.Quiz)

#### history(user)

The user’s answered-problem records, oldest first.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### next_problem(user, , strategy=None, mode='video_to_name', study_set=None, session_id=None, rng=None)

Choose the next problem for `user` from their history, and remember it as issued.

* **Parameters:**
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – a key of [`strategies()`](#kodokan.quiz.Quiz.strategies) (default: confusion-weighted).
  * **study_set** – the technique keys to draw from (default: the whole catalog).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unknown user format, mode, or an empty study set.
* **Return type:**
  [`Problem`](kodokan.flashcards.html.md#kodokan.flashcards.Problem)

#### stats(user)

Per-technique `{n, wrong, last_ts, last_correct}` for the user.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### *static* strategies()

The selectable strategies (for a settings screen).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### submit(problem_id, chosen_index, , timed_out=False, response_time_ms=None)

Score an issued problem, store the record, and return it.

`chosen_index` is None for a timeout. A problem is answered once.

* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `problem_id` was never issued, or was already answered.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `chosen_index` is not one of the choices.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.quiz.USER_RE *= re.compile('^(?!.\*_\_)[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')*

A user id is a filesystem-safe token; `__` is the separator of the stored keys.

### kodokan.quiz.quiz_stores(directory=None)

`(responses, problems)` as `dol` JSON-file stores under the data root.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)]
