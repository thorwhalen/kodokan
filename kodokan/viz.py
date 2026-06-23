"""Visualize a :class:`~kodokan.pose.PoseSequence`.

Two outputs, both requested by the project:

- :func:`render_skeleton_video` — bake a shareable MP4 with the skeleton drawn
  either *on the original video* (overlay) or *on a blank canvas* (skeleton only).
- :func:`log_to_rerun` — log frames + 2D skeletons to a Rerun recording for
  interactive, scrubbable inspection (and side-by-side compare of two sequences).

Persons are drawn in distinct colors (slot 0 green, slot 1 magenta) so tori/uke
are visually separable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kodokan.pose import PoseSequence

PathLike = str | Path

#: BGR colors per person slot (green, magenta, cyan, yellow…).
DEFAULT_PERSON_COLORS = ((0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255))


def _draw_persons(canvas, persons, skeleton, colors, conf_thresh, thickness, radius):
    import cv2

    for p in range(persons.shape[0]):
        kp = persons[p]
        if np.all(np.isnan(kp)):
            continue
        color = colors[p % len(colors)]
        for a, b in skeleton:
            xa, ya, ca = kp[a]
            xb, yb, cb = kp[b]
            if ca >= conf_thresh and cb >= conf_thresh and not (np.isnan(xa) or np.isnan(xb)):
                cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), color, thickness, cv2.LINE_AA)
        for j in range(kp.shape[0]):
            x, y, c = kp[j]
            if c >= conf_thresh and not np.isnan(x):
                cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def render_skeleton_video(
    pose_seq: PoseSequence,
    *,
    out_path: PathLike,
    source_video: PathLike | None = None,
    blank_canvas: bool = False,
    conf_thresh: float = 0.3,
    fps: float | None = None,
    person_colors=DEFAULT_PERSON_COLORS,
    background=(0, 0, 0),
    thickness: int = 2,
    radius: int = 4,
) -> Path:
    """Render skeletons to an MP4.

    Args:
        pose_seq: The pose sequence to draw.
        out_path: Output ``.mp4`` path.
        source_video: Source clip; required (and used as backdrop) unless
            ``blank_canvas=True``.
        blank_canvas: If True, draw on a solid ``background`` instead of the video.
        conf_thresh: Hide keypoints/edges below this confidence.
        fps: Output fps (defaults to the sequence's source fps).
        person_colors: BGR colors per person slot.
        background: BGR background color for the blank canvas.
        thickness, radius: Edge thickness / joint radius in pixels.

    Returns:
        The output path.
    """
    import cv2

    out_path = Path(out_path)
    W, H = int(pose_seq.width), int(pose_seq.height)
    fps = float(fps or pose_seq.fps)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    use_backdrop = source_video is not None and not blank_canvas
    cap = cv2.VideoCapture(str(source_video)) if use_backdrop else None
    src_idx = -1

    for k, fi in enumerate(int(i) for i in pose_seq.frame_indices):
        if cap is not None:
            frame = None
            while src_idx < fi:
                ok, frame = cap.read()
                src_idx += 1
                if not ok:
                    frame = None
                    break
            canvas = frame.copy() if frame is not None else np.full((H, W, 3), background, np.uint8)
        else:
            canvas = np.full((H, W, 3), background, np.uint8)
        _draw_persons(canvas, pose_seq.keypoints[k], pose_seq.skeleton, person_colors, conf_thresh, thickness, radius)
        writer.write(canvas)

    writer.release()
    if cap is not None:
        cap.release()
    return out_path


def _set_time(rr, seconds: float) -> None:
    """Set the Rerun timeline (compatible with 0.33's ``set_time`` and older API)."""
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("time", duration=float(seconds))
            return
        except TypeError:
            pass
    rr.set_time_seconds("time", float(seconds))  # older Rerun


def log_to_rerun(
    pose_seq: PoseSequence,
    *,
    source_video: PathLike | None = None,
    save: PathLike | None = None,
    spawn: bool = False,
    blank_canvas: bool = True,
    frame_scale: float = 0.5,
    conf_thresh: float = 0.3,
    entity_prefix: str = "",
) -> None:
    """Log frames + 2D skeletons to Rerun (overlay-on-video and skeleton-only views).

    Args:
        pose_seq: The sequence to log.
        source_video: If given, downscaled frames are logged under ``video/`` as a backdrop.
        save: Write a standalone ``.rrd`` recording to this path.
        spawn: Launch the Rerun viewer.
        blank_canvas: Also log a skeleton-only view under ``skeleton/``.
        frame_scale: Downscale factor for logged video frames (keypoints scaled to match).
        conf_thresh: Hide keypoints below this confidence.
        entity_prefix: Prefix all entity paths (use distinct prefixes to compare two
            sequences side-by-side in one recording, e.g. ``"demoA/"``, ``"demoB/"``).
    """
    import cv2
    import rerun as rr

    rr.init("kodokan", spawn=spawn)
    if save:
        rr.save(str(save))

    ann = rr.AnnotationContext(
        [rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, label="person"),
            keypoint_connections=pose_seq.skeleton,
        )]
    )
    rr.log(f"{entity_prefix}", ann, static=True)

    cap = cv2.VideoCapture(str(source_video)) if source_video else None
    src_idx = -1

    for k, fi in enumerate(int(i) for i in pose_seq.frame_indices):
        _set_time(rr, fi / float(pose_seq.fps))
        if cap is not None:
            frame = None
            while src_idx < fi:
                ok, frame = cap.read()
                src_idx += 1
                if not ok:
                    frame = None
                    break
            if frame is not None:
                small = cv2.resize(frame, None, fx=frame_scale, fy=frame_scale)
                rr.log(f"{entity_prefix}video/image", rr.Image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

        for p in range(pose_seq.n_persons):
            kp = pose_seq.keypoints[k, p]
            conf = kp[:, 2]
            mask = conf >= conf_thresh
            if not mask.any():
                continue
            ids = np.nonzero(mask)[0]
            pts = kp[mask, :2]
            if cap is not None:
                rr.log(
                    f"{entity_prefix}video/person{p}",
                    rr.Points2D(pts * frame_scale, keypoint_ids=ids, class_ids=0),
                )
            if blank_canvas:
                rr.log(
                    f"{entity_prefix}skeleton/person{p}",
                    rr.Points2D(pts, keypoint_ids=ids, class_ids=0),
                )

    if cap is not None:
        cap.release()
