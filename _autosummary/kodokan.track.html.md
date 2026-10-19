# kodokan.track

Stable tori/uke identity tracking (BoT-SORT / ByteTrack).

The plain [`estimate_poses()`](kodokan.pose.html.md#kodokan.pose.estimate_poses) keeps the top-2 detections and
orders them left→right *per frame*, so the two slots swap whenever tori and uke
cross. This module instead runs a multi-object tracker (Ultralytics’ built-in
BoT-SORT by default — appearance ReID + camera-motion compensation) so each
person keeps a persistent `track_id` across frames; we then bind the two most
persistent tracks to fixed slots for the whole clip (ordered left→right by their
clip-average x), giving a stable identity that does not swap mid-throw.

Limitation: under the heavy mutual occlusion at a throw’s apex a track can
fragment (a person reappears with a new id). Top-2-by-presence captures the
dominant fragments; fragment-merging/ReID-stitching is a later refinement (see
`misc/docs/research-architecture.md` §4).

### Functions

| [`estimate_poses_tracked`](#kodokan.track.estimate_poses_tracked)(video_path, \*[, ...])   | Estimate per-frame keypoints with persistent tori/uke identity.                  |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`identity_swap_rate`](#kodokan.track.identity_swap_rate)(pose_seq)                    | Ground-truth-free track-identity discontinuity rate (a tracking-quality metric). |

### kodokan.track.estimate_poses_tracked(video_path, , n_persons=2, tracker='botsort.yaml', conf_thresh=0.3, max_gap_frac=0.15, stale_after=12, frame_step=1, frame_range=None, device='mps', model_name='yolo11n-pose.pt', source_url=None, progress=True)

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

### kodokan.track.identity_swap_rate(pose_seq)

Ground-truth-free track-identity discontinuity rate (a tracking-quality metric).

For each consecutive frame pair in which *all* person slots are present, we
solve the minimum-cost assignment between the two frames’ person centroids.
When the optimal assignment is not the identity (slot *i* → slot *i*), the
slots’ spatial positions are better explained by a **label swap** — a proxy
for a tori/uke identity swap that the review asked us to instrument. A stable
tracker trends toward `0`.

Returns `{n_pairs, n_swaps, swap_rate}`.

Caveat (honest): this also fires on a *genuine* physical crossing that the
tracker correctly follows through, so read `swap_rate` as a relative
diagnostic (compare trackers/clips), not an absolute error count.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
