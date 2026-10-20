# kodokan.flashcards

Technique-recognition learning game (app #6) — logic layer.

Help students learn to recognize/name throws. No throw-detection is needed (ground
truth is known): we generate multiple-choice problems from the clip library —
**name -> pick the video** or **video -> pick the name** — with *confusable*
distractors, log every response (timestamped, with the exact problem) for adaptive
learning, and score with confusion-aware weighting.

“Confusability” between techniques is a **pose-shape similarity** (closeness of their
mean pooled descriptors) — an honest proxy for visual confusability that does NOT
require a working recognizer (the cross-source recognizer is currently at chance). It
can be swapped for the model’s confusion matrix once recognition generalizes.

This module is the (UI-agnostic) logic + storage; a web UI plugs into it later.

### Functions

| [`build_catalog`](#kodokan.flashcards.build_catalog)(\*[, min_two_person_frac, ...])      | Build `{technique_key: {"name", "clips":[...]}}` from the pose/segments stores.                                 |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`build_confusability`](#kodokan.flashcards.build_confusability)([catalog, feature])            | Technique×technique similarity (0..1) from mean pooled pose descriptors.                                        |
| [`clip_filename`](#kodokan.flashcards.clip_filename)(video_id, start_s, dur_s)            | The web-app clip file for one demo segment, named by the segment's IDENTITY.                                    |
| [`confusable_distractors`](#kodokan.flashcards.confusable_distractors)(target_key, keys, \*, n)    | Pick `n` distractor keys, preferring those most confusable with the target.                                     |
| [`log_response`](#kodokan.flashcards.log_response)(problem, chosen_index, \*[, ...])     | Record a response (datetime + exact problem + choice + correctness + score).                                    |
| [`make_problem`](#kodokan.flashcards.make_problem)(catalog, target_key, \*[, mode, ...]) | Build a multiple-choice problem for `target_key` from a technique `catalog`.                                    |
| [`next_target`](#kodokan.flashcards.next_target)(keys, history, \*[, focus, rng])       | Pick the next technique to quiz: unseen and often-missed (within `focus`) first.                                |
| [`problem_from_keys`](#kodokan.flashcards.problem_from_keys)(catalog, target_key, ...)        | Assemble a [`Problem`](#kodokan.flashcards.Problem) from an already-chosen, ordered option set. |
| [`responses_store`](#kodokan.flashcards.responses_store)([directory])                       | A dict-like JSON store of logged responses (per-user, timestamped).                                             |
| [`score_response`](#kodokan.flashcards.score_response)(correct_key, chosen_key, \*[, ...]) | Confusion-weighted credit in [0, 1].                                                                            |

### Classes

| [`Problem`](#kodokan.flashcards.Problem)(problem_id, mode, target_key, ...)   | One multiple-choice flashcard problem.   |
|-----------------------------------------------------------------------------------------------|------------------------------------------|

### *class* kodokan.flashcards.Problem(problem_id, mode, target_key, prompt, choices, correct_index)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One multiple-choice flashcard problem.

### kodokan.flashcards.build_catalog(, min_two_person_frac=None, min_demo_s=1.0, max_demos=None, segments=None)

Build `{technique_key: {"name", "clips":[...]}}` from the pose/segments stores.

Aggregates all sources: each clip contributes its demo intervals (for looping media),
so a technique present in several sources offers varied multiple-choice media.

`max_demos` caps the demos kept per source video (`None`, the default, keeps every
demo that passes the quality gate — #37: the store has 6–8 reps per video and all of
them are worth showing). `segments` is the `{video_id: record}` mapping to read
(default: [`kodokan.store.segments_store()`](kodokan.store.html.md#kodokan.store.segments_store)).

### kodokan.flashcards.build_confusability(catalog=None, , feature='tori_angles_pos')

Technique×technique similarity (0..1) from mean pooled pose descriptors.

`similarity[a][b]` high ⇒ techniques a, b look alike in pose-space (likely
confusable for a learner). An honest, recognizer-free proxy; swap for the model’s
confusion matrix once cross-source recognition works.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.flashcards.clip_filename(video_id, start_s, dur_s)

The web-app clip file for one demo segment, named by the segment’s IDENTITY.

`{video_id}_{start_ms}_{dur_ms}.mp4`: the name says which time span of which source it
shows, so re-segmenting can never make an existing name show a different cut (#45) — a
changed segment simply gets a new name. Positional names (`{vid}_3.mp4`) could.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.flashcards.confusable_distractors(target_key, keys, , n, similarity=None, rng=None)

Pick `n` distractor keys, preferring those most confusable with the target.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### kodokan.flashcards.log_response(problem, chosen_index, , user='default', store=None, similarity=None, timestamp=None)

Record a response (datetime + exact problem + choice + correctness + score).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.flashcards.make_problem(catalog, target_key, , mode='video_to_name', n_choices=4, similarity=None, rng=None, clip_for=None)

Build a multiple-choice problem for `target_key` from a technique `catalog`.

`catalog` maps `technique_key -> {"name": str, "clips": [clip dicts]}`.
`clip_for(key)` returns a looping clip for a key (default: first clip); for
`name_to_video` the choices are clips, for `video_to_name` they are names.

* **Return type:**
  [`Problem`](#kodokan.flashcards.Problem)

### kodokan.flashcards.next_target(keys, history, , focus=None, rng=None)

Pick the next technique to quiz: unseen and often-missed (within `focus`) first.

Spaced-repetition-ish weight = (wrongs + 1) / (seen + 1): unseen -> 1, frequently
wrong -> high, mastered -> low. `history` is a list of response dicts.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.flashcards.problem_from_keys(catalog, target_key, option_keys, , mode='video_to_name', clip_for=None, problem_id)

Assemble a [`Problem`](#kodokan.flashcards.Problem) from an already-chosen, ordered option set.

The seam between *choosing* (`make_problem`, or a `learning.Strategy`) and
*presenting*: whoever picks the options, the problem looks the same.

* **Return type:**
  [`Problem`](#kodokan.flashcards.Problem)

### kodokan.flashcards.responses_store(directory=None)

A dict-like JSON store of logged responses (per-user, timestamped).

### kodokan.flashcards.score_response(correct_key, chosen_key, , similarity=None, max_partial=0.5)

Confusion-weighted credit in [0, 1].

Correct = 1.0. A wrong answer earns *partial* credit proportional to how confusable
the chosen technique is with the correct one (an honest mistake between look-alike
throws is penalized less than confusing two obviously different throws).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
