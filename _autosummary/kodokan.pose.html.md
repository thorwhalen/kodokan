# kodokan.pose

Body-pose estimation: a video → a sequence of per-person skeletal keypoints.

A single [`estimate_poses()`](#kodokan.pose.estimate_poses) facade with a pluggable, multi-person backend
(`rtmlib` = RTMPose top-down, default; or `ultralytics` = YOLO11-pose). Both
emit the **COCO-17** keypoint layout, so downstream code is backend-agnostic.

Judo throws involve two people in close contact, so the default keeps the
`n_persons` highest-confidence detections per frame and orders them left→right
(a crude tori/uke proxy that is good enough for visualization and for the first
“compare two demos” experiments; true identity tracking through the occluded
apex of a throw is a separate, harder problem — see `misc/docs/research-architecture.md`).

Output is a [`PoseSequence`](#kodokan.pose.PoseSequence): an `(F, P, 17, 3)` array (x, y, confidence;
`NaN` where a person slot is empty in that frame), plus the frame indices, fps,
and source provenance.

### Module Attributes

| [`COCO17_KEYPOINTS`](#kodokan.pose.COCO17_KEYPOINTS)   | COCO-17 keypoint names, in index order.                                     |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`COCO17_SKELETON`](#kodokan.pose.COCO17_SKELETON)    | COCO-17 skeleton edges (pairs of keypoint indices) for drawing/connections. |

### Functions

| [`estimate_poses`](#kodokan.pose.estimate_poses)(video_path, \*[, backend, ...])   | Estimate per-frame, per-person COCO-17 keypoints for a video clip.   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|

### Classes

| [`PoseSequence`](#kodokan.pose.PoseSequence)(keypoints, frame_indices, fps, ...)   | A per-frame, per-person sequence of COCO-17 keypoints for one clip.   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|

### kodokan.pose.COCO17_KEYPOINTS *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]* *= ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')*

COCO-17 keypoint names, in index order.

### kodokan.pose.COCO17_SKELETON *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)], ...]* *= ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6))*

COCO-17 skeleton edges (pairs of keypoint indices) for drawing/connections.

### *class* kodokan.pose.PoseSequence(keypoints, frame_indices, fps, width, height, backend, video_path, source_url=None, keypoint_names=('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'), skeleton=((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)))

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

### kodokan.pose.estimate_poses(video_path, , backend='rtmlib', n_persons=2, conf_thresh=0.3, frame_step=1, frame_range=None, device=None, source_url=None, progress=True, \*\*backend_kwargs)

Estimate per-frame, per-person COCO-17 keypoints for a video clip.

Simplest use: `estimate_poses("clip.mp4")` → a [`PoseSequence`](#kodokan.pose.PoseSequence) with the
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
  [`PoseSequence`](#kodokan.pose.PoseSequence)
* **Returns:**
  A [`PoseSequence`](#kodokan.pose.PoseSequence) of shape `(F, n_persons, 17, 3)`.
