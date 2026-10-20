# kodokan.recognizer

Technique recognition for applications: top-k guesses with inspectable scores (#3).

[`kodokan.recognize`](kodokan.recognize.html.md#module-kodokan.recognize) holds the research tools (features, LOO evaluation, metrics).
This module is the application facade over them:

- [`TechniqueRecognizer`](#kodokan.recognizer.TechniqueRecognizer) — fit on pooled demo descriptors, then
  [`recognize()`](#kodokan.recognizer.TechniqueRecognizer.recognize) returns a [`Recognition`](#kodokan.recognizer.Recognition): the top-k
  [`Guess`](#kodokan.recognizer.Guess) es *and* the score of every technique (the intermediate vector a UI
  > can show).
- [`recognize_segments()`](#kodokan.recognizer.recognize_segments) — offline video: detect the throws in a whole pose
  sequence and recognize each.
- [`ThrowDetector`](#kodokan.recognizer.ThrowDetector) — a rolling pose buffer that reports each throw once its
  segment has closed; [`LiveRecognizer`](#kodokan.recognizer.LiveRecognizer) recognizes them (live stream mode).

Scores are a softmax over negative distances to each technique’s centroid, in the
train-standardized (and, when scikit-learn is installed, LDA-projected) descriptor
space, with distances in units of the typical within-technique spread (measured
leave-one-out, as a held-out demo would see it). They rank techniques and sum to 1;
they are not calibrated probabilities. The LDA projection is still fit on every
training demo, so with few demos per technique the spread stays optimistic.

```pycon
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(c, 0.3, size=(5, 4)) for c in (0.0, 3.0, 6.0)])
>>> y = np.repeat(["seoi-nage", "o-goshi", "uchi-mata"], 5)
>>> rec = TechniqueRecognizer(method="centroid").fit(X, y)
>>> r = rec.recognize(np.full(4, 3.1), k=2)
>>> r.guesses[0].technique, len(r.guesses), round(sum(r.scores.values()), 6)
('o-goshi', 2, 1.0)
```

### Module Attributes

| [`Featurizer`](#kodokan.recognizer.Featurizer)             | `(pose_seq, start_s, end_s) -> fixed-length descriptor` of one throw window.                                                                                          |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_MIN_PEAK_ACTIVITY`](#kodokan.recognizer.DFLT_MIN_PEAK_ACTIVITY) | Smoothed motion energy (keypoint speed, px per analysed frame) a throw must reach; below it a "segment" is pose jitter, which quantile thresholds alone would accept. |

### Functions

| [`motion_segments`](#kodokan.recognizer.motion_segments)(pose_seq, \*[, min_peak_activity])   | Throw windows by motion energy ([`kodokan.segment.find_segments()`](kodokan.segment.html.md#kodokan.segment.find_segments)), keeping only those whose peak energy reaches `min_peak_activity` (an absolute floor: the hysteresis thresholds are quantiles, so an idle buffer still has "segments").   |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`pooled_demo_descriptor`](#kodokan.recognizer.pooled_demo_descriptor)(pose_seq, start_s, ...)       | The default [`Featurizer`](#kodokan.recognizer.Featurizer): per-frame demo features, temporal-pyramid pooled.                                                                                                                                                                                                |
| [`recognize_segments`](#kodokan.recognizer.recognize_segments)(pose_seq, recognizer, \*)         | Offline video mode: every detected throw in `pose_seq`, recognized.                                                                                                                                                                                                                                                       |

### Classes

| [`Guess`](#kodokan.recognizer.Guess)(technique, score)                        | One candidate technique and its score (higher is more likely).                                                   |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`LiveRecognizer`](#kodokan.recognizer.LiveRecognizer)(recognizer, \*, fps[, k, ...])  | Live mode: each throw a [`ThrowDetector`](#kodokan.recognizer.ThrowDetector) reports, recognized once. |
| [`Recognition`](#kodokan.recognizer.Recognition)(guesses[, scores])                 | The top-k guesses, best first, and every technique's score.                                                      |
| [`TechniqueRecognizer`](#kodokan.recognizer.TechniqueRecognizer)(\*[, method, temperature]) | Fit on pooled descriptors; recognize with top-k guesses and the full score vector.                               |
| [`ThrowDetector`](#kodokan.recognizer.ThrowDetector)(\*, fps[, width, height, ...])   | A rolling pose buffer that reports each throw once, after its segment closes.                                    |

### kodokan.recognizer.DFLT_MIN_PEAK_ACTIVITY *= 3.0*

Smoothed motion energy (keypoint speed, px per analysed frame) a throw must reach;
below it a “segment” is pose jitter, which quantile thresholds alone would accept.

### kodokan.recognizer.Featurizer

`(pose_seq, start_s, end_s) -> fixed-length descriptor` of one throw window.

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], `ndarray`]

### *class* kodokan.recognizer.Guess(technique, score)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One candidate technique and its score (higher is more likely).

### *class* kodokan.recognizer.LiveRecognizer(recognizer, \*, fps, k=3, featurize=<function pooled_demo_descriptor>, \*\*detector_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Live mode: each throw a [`ThrowDetector`](#kodokan.recognizer.ThrowDetector) reports, recognized once.

[`push()`](#kodokan.recognizer.LiveRecognizer.push) takes one frame (`(P, 17, 3)` keypoints) and returns
`(segment, recognition)` for the throws that closed with it. Detector options
(`buffer_s`, `close_after_s`, `segmenter`, `width`, `height`) pass through.

#### push(keypoints, frame_index=None)

Add one frame; return the throws that closed with it, recognized.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment), [`Recognition`](#kodokan.recognizer.Recognition)]]

### *class* kodokan.recognizer.Recognition(guesses, scores=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The top-k guesses, best first, and every technique’s score.

#### score_of(technique)

The score of `technique` (0 for a technique the recognizer never saw).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* kodokan.recognizer.TechniqueRecognizer(, method='lda', temperature=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Fit on pooled descriptors; recognize with top-k guesses and the full score vector.

* **Parameters:**
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"lda"` (LDA projection when scikit-learn is installed, else plain
    standardized space) or `"centroid"` (standardized space only).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – softmax temperature; higher spreads the scores out.

#### fit(X, y)

Fit on pooled descriptors `X` (one row per demo) and technique labels `y`.

* **Return type:**
  [`TechniqueRecognizer`](#kodokan.recognizer.TechniqueRecognizer)

#### fit_demos(feature_seqs, labels, , levels=(1, 2, 4))

Fit on per-frame feature sequences (e.g. [`kodokan.recognize.demo_feature()`](kodokan.recognize.html.md#kodokan.recognize.demo_feature)).

* **Return type:**
  [`TechniqueRecognizer`](#kodokan.recognizer.TechniqueRecognizer)

#### recognize(x, , k=3)

Top-`k` guesses (best first) and the full score vector for one descriptor.

* **Return type:**
  [`Recognition`](#kodokan.recognizer.Recognition)

#### recognize_clip(pose_seq, start_s, end_s, \*, k=3, featurize=<function pooled_demo_descriptor>)

Recognize the throw in `pose_seq` between `start_s` and `end_s`.

* **Return type:**
  [`Recognition`](#kodokan.recognizer.Recognition)

#### scores(x)

Every technique’s score for one pooled descriptor (they sum to 1).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### *class* kodokan.recognizer.ThrowDetector(\*, fps, width=0, height=0, buffer_s=20.0, close_after_s=1.0, segmenter=<function motion_segments>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A rolling pose buffer that reports each throw once, after its segment closes.

Push one frame at a time (`(P, 17, 3)` keypoints, as in `PoseSequence`).
[`push()`](#kodokan.recognizer.ThrowDetector.push) returns `(buffer, segment)` for the throws that closed with that
frame. A segment counts as closed when it ends at least `close_after_s` before
the newest frame; the buffer keeps the last `buffer_s` seconds of pushed frames
(at `fps` frames a second: push every analysed frame, or pass the analysed rate
as `fps`). A segment that mostly overlaps a throw already reported is that throw.

#### buffer()

The buffered frames as a `PoseSequence` (people padded with NaN).

* **Return type:**
  [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)

#### push(keypoints, frame_index=None)

Add one frame; return `(buffer, segment)` for each throw it closed.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence), [`Segment`](kodokan.segment.html.md#kodokan.segment.Segment)]]

### kodokan.recognizer.motion_segments(pose_seq, , min_peak_activity=3.0, \*\*segment_kwargs)

Throw windows by motion energy ([`kodokan.segment.find_segments()`](kodokan.segment.html.md#kodokan.segment.find_segments)), keeping
only those whose peak energy reaches `min_peak_activity` (an absolute floor:
the hysteresis thresholds are quantiles, so an idle buffer still has “segments”).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment)]

### kodokan.recognizer.pooled_demo_descriptor(pose_seq, start_s, end_s, , mode='tori_angles_pos', levels=(1, 2, 4))

The default [`Featurizer`](#kodokan.recognizer.Featurizer): per-frame demo features, temporal-pyramid pooled.

* **Return type:**
  `ndarray`

### kodokan.recognizer.recognize_segments(pose_seq, recognizer, \*, k=3, featurize=<function pooled_demo_descriptor>, segmenter=<function motion_segments>)

Offline video mode: every detected throw in `pose_seq`, recognized.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment), [`Recognition`](#kodokan.recognizer.Recognition)]]
