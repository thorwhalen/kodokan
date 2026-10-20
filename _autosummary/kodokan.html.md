# kodokan

kodokan — study Kodokan Judo throws from video via body-pose analysis.

A pipeline over the official *Kodokan 100 Techniques* YouTube playlist: acquire
clips + metadata, estimate per-frame two-person skeletons, segment each clip into
its repeated demonstrations, visualize skeletons (overlay-on-video and on a blank
canvas), and — later — recognize, compare, and score throws.

Quick start:

```default
from kodokan.acquire import download_techniques
from kodokan.pose import estimate_poses
from kodokan.viz import render_skeleton_video

res = download_techniques(playlist_items="2")          # Seoi-nage
seq = estimate_poses(res[0].path, source_url=res[0].info["webpage_url"])
render_skeleton_video(seq, out_path="overlay.mp4", source_video=res[0].path)
render_skeleton_video(seq, out_path="skeleton.mp4", blank_canvas=True)
```

See `misc/docs/research-architecture.md` for the tool/architecture rationale.

### Functions

| [`estimate_poses`](#kodokan.estimate_poses)(video_path, \*[, backend, ...])     | Estimate per-frame, per-person COCO-17 keypoints for a video clip.                 |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`estimate_poses_tracked`](#kodokan.estimate_poses_tracked)(video_path, \*[, ...])      | Estimate per-frame keypoints with persistent tori/uke identity.                    |
| [`render_skeleton_video`](#kodokan.render_skeleton_video)(pose_seq, \*, out_path)      | Render skeletons to an MP4.                                                        |
| [`log_to_rerun`](#kodokan.log_to_rerun)(pose_seq, \*[, source_video, ...])    | Log frames + 2D skeletons to Rerun (overlay-on-video and skeleton-only views).     |
| [`segment_demonstrations`](#kodokan.segment_demonstrations)(pose_seq, \*[, ...])        | Find demonstrations, annotate each with its two-person coverage, and gate.         |
| [`compare`](#kodokan.compare)(features_a, features_b)                    | DTW-align two angle-feature sequences.                                             |
| [`angle_features`](#kodokan.angle_features)(pose_seq, \*[, person, ...])        | Per-frame joint-angle features `(F, n_angles)` for one person.                     |
| [`distance_matrix`](#kodokan.distance_matrix)(feature_seqs)                      | Pairwise normalized-DTW distance matrix over a list of feature sequences.          |
| [`pose_store`](#kodokan.pose_store)([directory])                            | A dict-like `{video_id: PoseSequence}` store backed by per-clip Parquet.           |
| [`segments_store`](#kodokan.segments_store)([directory])                        | A dict-like `{video_id: <json>}` store for demo segments (and metadata).           |
| [`load_all_tidy`](#kodokan.load_all_tidy)([store])                             | Concatenate every clip's tidy table (with a `video_id` column) into one DataFrame. |
| [`build_reference`](#kodokan.build_reference)(feature_list)                      | Pick the medoid demo as reference and record the baseline distance spread.         |
| [`score`](#kodokan.score)(query_features, reference)                   | Score a query against a reference, calibrated to the genuine-demo spread.          |
| [`feedback`](#kodokan.feedback)(query_features, reference, \*[, ...])     | Interpretable difference: per joint-angle and per phase (degrees).                 |
| [`demo_features`](#kodokan.demo_features)(pose_seq, start_s, end_s, \*[, ...]) | Cleaned joint-angle features for one demo window (primary person by default).      |
| [`recognize_segments`](#kodokan.recognize_segments)(pose_seq, recognizer, \*)       | Offline video mode: every detected throw in `pose_seq`, recognized.                |
| [`make_problem`](#kodokan.make_problem)(catalog, target_key, \*[, mode, ...]) | Build a multiple-choice problem for `target_key` from a technique `catalog`.       |
| [`build_catalog`](#kodokan.build_catalog)(\*[, min_two_person_frac, ...])      | Build `{technique_key: {"name", "clips":[...]}}` from the pose/segments stores.    |
| [`clip_filename`](#kodokan.clip_filename)(video_id, start_s, dur_s)            | The web-app clip file for one demo segment, named by the segment's IDENTITY.       |
| [`build_confusability`](#kodokan.build_confusability)([catalog, feature])            | Technique×technique similarity (0..1) from mean pooled pose descriptors.           |
| [`score_response`](#kodokan.score_response)(correct_key, chosen_key, \*[, ...]) | Confusion-weighted credit in [0, 1].                                               |
| [`next_target`](#kodokan.next_target)(keys, history, \*[, focus, rng])       | Pick the next technique to quiz: unseen and often-missed (within `focus`) first.   |
| [`log_response`](#kodokan.log_response)(problem, chosen_index, \*[, ...])     | Record a response (datetime + exact problem + choice + correctness + score).       |

### Classes

| [`PoseSequence`](#kodokan.PoseSequence)(keypoints, frame_indices, fps, ...)   | A per-frame, per-person sequence of COCO-17 keypoints for one clip.                 |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`Segment`](#kodokan.Segment)(index, start_s, end_s, start_frame, ...)   | One detected demonstration interval.                                                |
| [`TechniqueRecognizer`](#kodokan.TechniqueRecognizer)(\*[, method, temperature])     | Fit on pooled descriptors; recognize with top-k guesses and the full score vector.  |
| [`Recognition`](#kodokan.Recognition)(guesses[, scores])                     | The top-k guesses, best first, and every technique's score.                         |
| [`Guess`](#kodokan.Guess)(technique, score)                            | One candidate technique and its score (higher is more likely).                      |
| [`LiveRecognizer`](#kodokan.LiveRecognizer)(recognizer, \*, fps[, width, ...])  | Live mode: a rolling pose buffer; each throw is recognized once its segment closes. |

### *class* kodokan.Guess(technique, score)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One candidate technique and its score (higher is more likely).

### *class* kodokan.LiveRecognizer(recognizer, \*, fps, width=0, height=0, k=3, buffer_s=20.0, close_after_s=1.0, featurize=<function pooled_demo_descriptor>, segmenter=<function motion_segments>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Live mode: a rolling pose buffer; each throw is recognized once its segment closes.

Push one frame at a time (`(P, 17, 3)` keypoints, as in [`PoseSequence`](#kodokan.PoseSequence)).
[`push()`](#kodokan.LiveRecognizer.push) returns the throws that closed with that frame, each recognized once.
A segment counts as closed when it ends at least `close_after_s` before the newest
frame; the buffer keeps the last `buffer_s` seconds of pushed frames (at `fps`
frames a second: push every analysed frame, or pass the analysed rate as `fps`).
A segment that mostly overlaps a throw already recognized is that throw.

#### buffer()

The buffered frames as a [`PoseSequence`](#kodokan.PoseSequence) (people padded with NaN).

* **Return type:**
  [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)

#### push(keypoints, frame_index=None)

Add one frame; return the throws that closed with it, recognized.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment), [`Recognition`](kodokan.recognizer.html.md#kodokan.recognizer.Recognition)]]

### *class* kodokan.PoseSequence(keypoints, frame_indices, fps, width, height, backend, video_path, source_url=None, keypoint_names=('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'), skeleton=((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)))

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A per-frame, per-person sequence of COCO-17 keypoints for one clip.

#### keypoints

`(F, P, 17, 3)` float array of (x_px, y_px, confidence);
`NaN` for empty person slots.

* **Type:**
  `ndarray`

#### frame_indices

`(F,)` int array of source frame indices analyzed.

* **Type:**
  `ndarray`

#### fps

Source frames-per-second.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### width, height

Source frame dimensions in pixels.

#### backend

The estimator backend used.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### video_path

Source clip path (provenance).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### source_url

Canonical YouTube URL, if known (provenance; required by project).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### save_npz(path)

Persist as a compressed `.npz` (the dense numeric cache).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

#### times()

Timestamp (seconds) of each analyzed frame.

* **Return type:**
  `ndarray`

### *class* kodokan.Recognition(guesses, scores=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The top-k guesses, best first, and every technique’s score.

#### score_of(technique)

The score of `technique` (0 for a technique the recognizer never saw).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* kodokan.Segment(index, start_s, end_s, start_frame, end_frame, peak_activity, two_person_frac=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One detected demonstration interval.

### *class* kodokan.TechniqueRecognizer(, method='lda', temperature=1.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Fit on pooled descriptors; recognize with top-k guesses and the full score vector.

* **Parameters:**
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"lda"` (LDA projection when scikit-learn is installed, else plain
    standardized space) or `"centroid"` (standardized space only).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – softmax temperature; higher spreads the scores out.

#### fit(X, y)

Fit on pooled descriptors `X` (one row per demo) and technique labels `y`.

* **Return type:**
  [`TechniqueRecognizer`](kodokan.recognizer.html.md#kodokan.recognizer.TechniqueRecognizer)

#### fit_demos(feature_seqs, labels, , levels=(1, 2, 4))

Fit on per-frame feature sequences (e.g. [`kodokan.recognize.demo_feature()`](kodokan.recognize.html.md#kodokan.recognize.demo_feature)).

* **Return type:**
  [`TechniqueRecognizer`](kodokan.recognizer.html.md#kodokan.recognizer.TechniqueRecognizer)

#### recognize(x, , k=3)

Top-`k` guesses (best first) and the full score vector for one descriptor.

* **Return type:**
  [`Recognition`](kodokan.recognizer.html.md#kodokan.recognizer.Recognition)

#### recognize_clip(pose_seq, start_s, end_s, \*, k=3, featurize=<function pooled_demo_descriptor>)

Recognize the throw in `pose_seq` between `start_s` and `end_s`.

* **Return type:**
  [`Recognition`](kodokan.recognizer.html.md#kodokan.recognizer.Recognition)

#### scores(x)

Every technique’s score for one pooled descriptor (they sum to 1).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### kodokan.angle_features(pose_seq, , person=0, frame_range=None)

Per-frame joint-angle features `(F, n_angles)` for one person.

* **Return type:**
  `ndarray`

### kodokan.build_catalog(, min_two_person_frac=None, min_demo_s=1.0, max_demos=None, segments=None)

Build `{technique_key: {"name", "clips":[...]}}` from the pose/segments stores.

Aggregates all sources: each clip contributes its demo intervals (for looping media),
so a technique present in several sources offers varied multiple-choice media.

`max_demos` caps the demos kept per source video (`None`, the default, keeps every
demo that passes the quality gate — #37: the store has 6–8 reps per video and all of
them are worth showing). `segments` is the `{video_id: record}` mapping to read
(default: [`kodokan.store.segments_store()`](kodokan.store.html.md#kodokan.store.segments_store)).

### kodokan.build_confusability(catalog=None, , feature='tori_angles_pos')

Technique×technique similarity (0..1) from mean pooled pose descriptors.

`similarity[a][b]` high ⇒ techniques a, b look alike in pose-space (likely
confusable for a learner). An honest, recognizer-free proxy; swap for the model’s
confusion matrix once cross-source recognition works.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.build_reference(feature_list)

Pick the medoid demo as reference and record the baseline distance spread.

Returns `{medoid, reference, baseline, distance_matrix}` where `baseline` is
the array of normalized-DTW distances from every other demo to the medoid.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.clip_filename(video_id, start_s, dur_s)

The web-app clip file for one demo segment, named by the segment’s IDENTITY.

`{video_id}_{start_ms}_{dur_ms}.mp4`: the name says which time span of which source it
shows, so re-segmenting can never make an existing name show a different cut (#45) — a
changed segment simply gets a new name. Positional names (`{vid}_3.mp4`) could.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.compare(features_a, features_b)

DTW-align two angle-feature sequences.

Returns a dict with `distance` (total DTW cost), `normalized` (cost per
aligned step — comparable across different-length demos), the warping `path`,
and the cleaned input arrays `a`/`b`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.demo_features(pose_seq, start_s, end_s, , person=None)

Cleaned joint-angle features for one demo window (primary person by default).

* **Return type:**
  `ndarray`

### kodokan.distance_matrix(feature_seqs)

Pairwise normalized-DTW distance matrix over a list of feature sequences.

* **Return type:**
  `ndarray`

### kodokan.estimate_poses(video_path, , backend='rtmlib', n_persons=2, conf_thresh=0.3, frame_step=1, frame_range=None, device=None, source_url=None, progress=True, \*\*backend_kwargs)

Estimate per-frame, per-person COCO-17 keypoints for a video clip.

Simplest use: `estimate_poses("clip.mp4")` → a [`PoseSequence`](#kodokan.PoseSequence) with the
two highest-confidence people per frame (left→right ordered) via RTMPose.

* **Parameters:**
  * **video_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to the video clip.
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"rtmlib"` (RTMPose top-down, CPU/ONNX; default, and the
    only one installed by `kodokan[pose]`) or `"ultralytics"`
    (YOLO11-pose, MPS), which needs `kodokan[track]` — an
    AGPL-3.0-or-later dependency, see the README’s
    “Licensing of extras”.
  * **n_persons** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of person slots to keep per frame (2 for tori+uke).
  * **conf_thresh** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum mean per-person confidence to keep a detection.
  * **frame_step** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analyze every `frame_step`-th frame (1 = every frame).
  * **frame_range** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional `(start, stop)` frame index window.
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Backend device (`"cpu"`/`"mps"`); backend default if `None`.
  * **source_url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Canonical source URL to record as provenance.
  * **progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print a progress line periodically.
  * **\*\*backend_kwargs** – Forwarded to the backend (e.g. `mode="performance"`
    for rtmlib, or `model_name="yolo11m-pose.pt"` for ultralytics).
* **Return type:**
  [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)
* **Returns:**
  A [`PoseSequence`](#kodokan.PoseSequence) of shape `(F, n_persons, 17, 3)`.

### kodokan.estimate_poses_tracked(video_path, , n_persons=2, tracker='botsort.yaml', conf_thresh=0.3, max_gap_frac=0.15, stale_after=12, frame_step=1, frame_range=None, device='mps', model_name='yolo11n-pose.pt', source_url=None, progress=True)

Estimate per-frame keypoints with persistent tori/uke identity.

Returns a [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence) whose person slots are *stable*
across the clip (slot 0 = the track that is, on average, further left).

* **Parameters:**
  * **video_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Path to the clip.
  * **n_persons** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of stable identity slots to keep (2 for tori+uke).
  * **tracker** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Ultralytics tracker config (`"botsort.yaml"` or `"bytetrack.yaml"`).
  * **conf_thresh** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum mean per-person confidence to count a detection.
  * **frame_step** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Analyze every n-th frame.
  * **frame_range** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional `(start, stop)` frame window.
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Torch device (`"mps"`/`"cpu"`).
  * **model_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – YOLO-pose weights (resolved under the data models dir).
  * **source_url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Provenance URL.
  * **progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print progress.
* **Return type:**
  [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)

### kodokan.feedback(query_features, reference, , n_phases=3)

Interpretable difference: per joint-angle and per phase (degrees).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.load_all_tidy(store=None)

Concatenate every clip’s tidy table (with a `video_id` column) into one DataFrame.

### kodokan.log_response(problem, chosen_index, , user='default', store=None, similarity=None, timestamp=None)

Record a response (datetime + exact problem + choice + correctness + score).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.log_to_rerun(pose_seq, , source_video=None, save=None, spawn=False, blank_canvas=True, frame_scale=0.5, conf_thresh=0.3, entity_prefix='')

Log frames + 2D skeletons to Rerun (overlay-on-video and skeleton-only views).

* **Parameters:**
  * **pose_seq** ([`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)) – The sequence to log.
  * **source_video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If given, downscaled frames are logged under `video/` as a backdrop.
  * **save** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Write a standalone `.rrd` recording to this path.
  * **spawn** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Launch the Rerun viewer.
  * **blank_canvas** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Also log a skeleton-only view under `skeleton/`.
  * **frame_scale** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Downscale factor for logged video frames (keypoints scaled to match).
  * **conf_thresh** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Hide keypoints below this confidence.
  * **entity_prefix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prefix all entity paths (use distinct prefixes to compare two
    sequences side-by-side in one recording, e.g. `"demoA/"`, `"demoB/"`).
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### kodokan.make_problem(catalog, target_key, , mode='video_to_name', n_choices=4, similarity=None, rng=None, clip_for=None)

Build a multiple-choice problem for `target_key` from a technique `catalog`.

`catalog` maps `technique_key -> {"name": str, "clips": [clip dicts]}`.
`clip_for(key)` returns a looping clip for a key (default: first clip); for
`name_to_video` the choices are clips, for `video_to_name` they are names.

* **Return type:**
  [`Problem`](kodokan.flashcards.html.md#kodokan.flashcards.Problem)

### kodokan.next_target(keys, history, , focus=None, rng=None)

Pick the next technique to quiz: unseen and often-missed (within `focus`) first.

Spaced-repetition-ish weight = (wrongs + 1) / (seen + 1): unseen -> 1, frequently
wrong -> high, mastered -> low. `history` is a list of response dicts.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.pose_store(directory=None)

A dict-like `{video_id: PoseSequence}` store backed by per-clip Parquet.

### kodokan.recognize_segments(pose_seq, recognizer, \*, k=3, featurize=<function pooled_demo_descriptor>, segmenter=<function motion_segments>)

Offline video mode: every detected throw in `pose_seq`, recognized.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment), [`Recognition`](kodokan.recognizer.html.md#kodokan.recognizer.Recognition)]]

### kodokan.render_skeleton_video(pose_seq, , out_path, source_video=None, blank_canvas=False, conf_thresh=0.3, fps=None, person_colors=((0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255)), background=(0, 0, 0), thickness=2, radius=4)

Render skeletons to an MP4.

* **Parameters:**
  * **pose_seq** ([`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)) – The pose sequence to draw.
  * **out_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – Output `.mp4` path.
  * **source_video** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Source clip; required (and used as backdrop) unless
    `blank_canvas=True`.
  * **blank_canvas** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, draw on a solid `background` instead of the video.
  * **conf_thresh** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Hide keypoints/edges below this confidence.
  * **fps** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output fps (defaults to the sequence’s source fps).
  * **person_colors** – BGR colors per person slot.
  * **background** – BGR background color for the blank canvas.
  * **thickness** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Edge thickness / joint radius in pixels.
  * **radius** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Edge thickness / joint radius in pixels.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The output path.

### kodokan.score(query_features, reference)

Score a query against a reference, calibrated to the genuine-demo spread.

`score` is 0–100: 100 ≈ as close as the closest genuine demo, 0 ≈ as far as the
90th-percentile (or worse). `closer_than_pct` is the percent of genuine demos
the query is closer-to-reference than.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.score_response(correct_key, chosen_key, , similarity=None, max_partial=0.5)

Confusion-weighted credit in [0, 1].

Correct = 1.0. A wrong answer earns *partial* credit proportional to how confusable
the chosen technique is with the correct one (an honest mistake between look-alike
throws is penalized less than confusing two obviously different throws).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### kodokan.segment_demonstrations(pose_seq, , video_path=None, use_optical_flow=False, min_two_person_frac=0.0, \*\*find_kwargs)

Find demonstrations, annotate each with its two-person coverage, and gate.

Uses pose motion energy by default (fuse optical flow with `use_optical_flow`).
Segments whose two-person coverage is below `min_two_person_frac` are dropped
(a throw demonstration needs both tori and uke), then re-indexed in time order.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment)]

### kodokan.segments_store(directory=None)

A dict-like `{video_id: <json>}` store for demo segments (and metadata).

### Modules

| [`acquire`](kodokan.acquire.html.md#module-kodokan.acquire)         | Acquire Kodokan technique clips from YouTube (thin wrapper over `yb.download`).                      |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`canon3d`](kodokan.canon3d.html.md#module-kodokan.canon3d)         | Pelvis-frame (Procrustes) canonicalization of 3D world landmarks.                                    |
| [`config`](kodokan.config.html.md#module-kodokan.config)           | Where kodokan keeps its data (downloaded clips + derived artifacts).                                 |
| [`descriptors`](kodokan.descriptors.html.md#module-kodokan.descriptors) | Experimental per-demo feature descriptors (to test what makes throws separable).                     |
| [`flashcards`](kodokan.flashcards.html.md#module-kodokan.flashcards)   | Technique-recognition learning game (app #6) — logic layer.                                          |
| [`learning`](kodokan.learning.html.md#module-kodokan.learning)       | Adaptive learning engine for the flashcard quiz — selectable strategies.                             |
| [`pose`](kodokan.pose.html.md#module-kodokan.pose)               | Body-pose estimation: a video → a sequence of per-person skeletal keypoints.                         |
| [`quiz`](kodokan.quiz.html.md#module-kodokan.quiz)               | Quiz service: the UI-agnostic orchestrator of the technique-recognition game (app #6).               |
| [`recognize`](kodokan.recognize.html.md#module-kodokan.recognize)     | Learned-vs-DTW technique recognition experiments (data-appropriate, no GPU).                         |
| [`recognizer`](kodokan.recognizer.html.md#module-kodokan.recognizer)   | Technique recognition for applications: top-k guesses with inspectable scores (#3).                  |
| [`segment`](kodokan.segment.html.md#module-kodokan.segment)         | Segment a clip into its repeated demonstrations of one throw.                                        |
| [`store`](kodokan.store.html.md#module-kodokan.store)             | dol-backed stores for the pipeline's artifacts (the analysis SSOT).                                  |
| [`track`](kodokan.track.html.md#module-kodokan.track)             | Stable tori/uke identity tracking (BoT-SORT / ByteTrack).                                            |
| [`viz`](kodokan.viz.html.md#module-kodokan.viz)                 | Visualize a [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence). |
