# kodokan.viz

Visualize a [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence).

Two outputs, both requested by the project:

- [`render_skeleton_video()`](#kodokan.viz.render_skeleton_video) — bake a shareable MP4 with the skeleton drawn
  either *on the original video* (overlay) or *on a blank canvas* (skeleton only).
- [`log_to_rerun()`](#kodokan.viz.log_to_rerun) — log frames + 2D skeletons to a Rerun recording for
  interactive, scrubbable inspection (and side-by-side compare of two sequences).

Persons are drawn in distinct colors (slot 0 green, slot 1 magenta) so tori/uke
are visually separable.

### Module Attributes

| [`DEFAULT_PERSON_COLORS`](#kodokan.viz.DEFAULT_PERSON_COLORS)   | BGR colors per person slot (green, magenta, cyan, yellow…).   |
|--------------------------------------------------------------------------|---------------------------------------------------------------|

### Functions

| [`log_to_rerun`](#kodokan.viz.log_to_rerun)(pose_seq, \*[, source_video, ...])   | Log frames + 2D skeletons to Rerun (overlay-on-video and skeleton-only views).   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`render_skeleton_video`](#kodokan.viz.render_skeleton_video)(pose_seq, \*, out_path)     | Render skeletons to an MP4.                                                      |

### kodokan.viz.DEFAULT_PERSON_COLORS *= ((0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255))*

BGR colors per person slot (green, magenta, cyan, yellow…).

### kodokan.viz.log_to_rerun(pose_seq, , source_video=None, save=None, spawn=False, blank_canvas=True, frame_scale=0.5, conf_thresh=0.3, entity_prefix='')

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

### kodokan.viz.render_skeleton_video(pose_seq, , out_path, source_video=None, blank_canvas=False, conf_thresh=0.3, fps=None, person_colors=((0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255)), background=(0, 0, 0), thickness=2, radius=4)

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
