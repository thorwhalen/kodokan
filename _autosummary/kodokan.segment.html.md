# kodokan.segment

Segment a clip into its repeated demonstrations of one throw.

A Kodokan demo clip shows the *same* throw several times (varying speed/angle/
position), each demonstration a burst of whole-body motion bracketed by a
low-motion “reset” (walk back, re-grip, bow). Scene-cut detection finds nothing
in a continuous take, so we segment by **rhythm**: build a 1-D motion-energy
signal and take the high-motion runs as demonstrations (research §5).

Robustness pieces:

- **Hysteresis thresholding** — a demo turns *on* above `high_quantile` and stays
  on until energy drops below `low_quantile`. This stops *slow-motion* reps (whose
  absolute energy is low) from being split, while rests still separate demos.
- **Two-person gate** — a throw demonstration needs both tori and uke, so segments
  are filtered by their two-person coverage (drops single-person intro/close-up spans).
- **Self-similarity matrix + autocorrelation period** — a dependency-free,
  RepNet-style cross-check on the rep structure/count (soft, since reps vary in speed).

Energy sources (fusable): [`pose_motion_energy()`](#kodokan.segment.pose_motion_energy) (confidence-weighted total
keypoint speed; pose-only) and [`optical_flow_energy()`](#kodokan.segment.optical_flow_energy) (dense Farneback; pose-free).

### Functions

| [`estimate_period`](#kodokan.segment.estimate_period)(energy, fps, \*[, ...])         | Soft rep-period/count cross-check via autocorrelation of the energy signal.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`find_segments`](#kodokan.segment.find_segments)(energy, frame_indices, fps, \*)   | Turn a 1-D motion-energy signal into demonstration intervals via hysteresis.  |
| [`optical_flow_energy`](#kodokan.segment.optical_flow_energy)(video_path, \*[, ...])      | Per-frame mean dense optical-flow magnitude (Farneback), pose-free.           |
| [`pose_motion_energy`](#kodokan.segment.pose_motion_energy)(pose_seq, \*[, conf_thresh]) | Per-frame confidence-weighted total keypoint speed, summed over all people.   |
| [`segment_demonstrations`](#kodokan.segment.segment_demonstrations)(pose_seq, \*[, ...])     | Find demonstrations, annotate each with its two-person coverage, and gate.    |
| [`self_similarity_matrix`](#kodokan.segment.self_similarity_matrix)(pose_seq, \*[, ...])     | Frame×frame self-similarity on joint-angle features (for rep-structure viz).  |

### Classes

| [`Segment`](#kodokan.segment.Segment)(index, start_s, end_s, start_frame, ...)   | One detected demonstration interval.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------|

### *class* kodokan.segment.Segment(index, start_s, end_s, start_frame, end_frame, peak_activity, two_person_frac=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One detected demonstration interval.

### kodokan.segment.estimate_period(energy, fps, , smooth_sigma=4.0, min_period_s=1.5, max_period_s=10.0)

Soft rep-period/count cross-check via autocorrelation of the energy signal.

Returns `{period_s, count_est, strength}`. `strength` (0–1) is the
autocorrelation at the chosen lag — low values mean weak periodicity (expected
when reps vary a lot in speed), so treat the count as approximate.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.segment.find_segments(energy, frame_indices, fps, , smooth_sigma=4.0, low_quantile=0.25, high_quantile=0.5, min_duration_s=1.0, merge_gap_s=0.4)

Turn a 1-D motion-energy signal into demonstration intervals via hysteresis.

A demonstration starts when smoothed energy rises above the `high_quantile`
threshold and ends when it falls below the `low_quantile` threshold — so a
slow-motion dip in the middle of a rep does not split it, while the low-motion
rest between reps (below `low_quantile`) does separate them.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#kodokan.segment.Segment)]

### kodokan.segment.optical_flow_energy(video_path, , frame_range=None, step=1, scale=0.4)

Per-frame mean dense optical-flow magnitude (Farneback), pose-free.

Returns `(energy, frame_indices)`. Frames are downscaled by `scale` for speed.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]

### kodokan.segment.pose_motion_energy(pose_seq, , conf_thresh=0.2)

Per-frame confidence-weighted total keypoint speed, summed over all people.

Returns an array aligned to `pose_seq.frame_indices` (first sample is 0).
Robust to missing people/keypoints (NaNs are ignored).

* **Return type:**
  `ndarray`

### kodokan.segment.segment_demonstrations(pose_seq, , video_path=None, use_optical_flow=False, min_two_person_frac=0.0, \*\*find_kwargs)

Find demonstrations, annotate each with its two-person coverage, and gate.

Uses pose motion energy by default (fuse optical flow with `use_optical_flow`).
Segments whose two-person coverage is below `min_two_person_frac` are dropped
(a throw demonstration needs both tori and uke), then re-indexed in time order.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](#kodokan.segment.Segment)]

### kodokan.segment.self_similarity_matrix(pose_seq, , person=None, max_frames=300)

Frame×frame self-similarity on joint-angle features (for rep-structure viz).

Returns `(sim, frame_indices_sub)`; repeated demonstrations appear as
off-diagonal bright blocks. Subsamples to `max_frames` for speed/readability.
