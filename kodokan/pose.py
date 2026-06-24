"""Body-pose estimation: a video → a sequence of per-person skeletal keypoints.

A single :func:`estimate_poses` facade with a pluggable, multi-person backend
(``rtmlib`` = RTMPose top-down, default; or ``ultralytics`` = YOLO11-pose). Both
emit the **COCO-17** keypoint layout, so downstream code is backend-agnostic.

Judo throws involve two people in close contact, so the default keeps the
``n_persons`` highest-confidence detections per frame and orders them left→right
(a crude tori/uke proxy that is good enough for visualization and for the first
"compare two demos" experiments; true identity tracking through the occluded
apex of a throw is a separate, harder problem — see ``misc/docs/research-architecture.md``).

Output is a :class:`PoseSequence`: an ``(F, P, 17, 3)`` array (x, y, confidence;
``NaN`` where a person slot is empty in that frame), plus the frame indices, fps,
and source provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PathLike = str | Path

#: COCO-17 keypoint names, in index order.
COCO17_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: COCO-17 skeleton edges (pairs of keypoint indices) for drawing/connections.
COCO17_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
)


@dataclass
class PoseSequence:
    """A per-frame, per-person sequence of COCO-17 keypoints for one clip.

    Attributes:
        keypoints: ``(F, P, 17, 3)`` float array of (x_px, y_px, confidence);
            ``NaN`` for empty person slots.
        frame_indices: ``(F,)`` int array of source frame indices analyzed.
        fps: Source frames-per-second.
        width, height: Source frame dimensions in pixels.
        backend: The estimator backend used.
        video_path: Source clip path (provenance).
        source_url: Canonical YouTube URL, if known (provenance; required by project).
    """

    keypoints: np.ndarray
    frame_indices: np.ndarray
    fps: float
    width: int
    height: int
    backend: str
    video_path: str
    source_url: str | None = None
    keypoint_names: tuple[str, ...] = field(default=COCO17_KEYPOINTS)
    skeleton: tuple[tuple[int, int], ...] = field(default=COCO17_SKELETON)

    @property
    def n_frames(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def n_persons(self) -> int:
        return int(self.keypoints.shape[1])

    def times(self) -> np.ndarray:
        """Timestamp (seconds) of each analyzed frame."""
        return self.frame_indices / float(self.fps)

    def save_npz(self, path: PathLike) -> Path:
        """Persist as a compressed ``.npz`` (the dense numeric cache)."""
        path = Path(path)
        np.savez_compressed(
            path,
            keypoints=self.keypoints,
            frame_indices=self.frame_indices,
            fps=self.fps,
            width=self.width,
            height=self.height,
            backend=self.backend,
            video_path=self.video_path,
            source_url=self.source_url or "",
        )
        return path

    @classmethod
    def load_npz(cls, path: PathLike) -> "PoseSequence":
        z = np.load(path, allow_pickle=False)
        return cls(
            keypoints=z["keypoints"],
            frame_indices=z["frame_indices"],
            fps=float(z["fps"]),
            width=int(z["width"]),
            height=int(z["height"]),
            backend=str(z["backend"]),
            video_path=str(z["video_path"]),
            source_url=str(z["source_url"]) or None,
        )


# --------------------------------------------------------------------------- #
# Backends (strategy): each returns estimate(frame_bgr) -> (n, 17, 3) array
# --------------------------------------------------------------------------- #


def _make_rtmlib_estimator(
    *, device: str | None, mode: str
) -> Callable[[np.ndarray], np.ndarray]:
    from rtmlib import Body

    body = Body(mode=mode, backend="onnxruntime", device=device or "cpu")

    def estimate(frame_bgr: np.ndarray) -> np.ndarray:
        kpts, scores = body(frame_bgr)  # (n,17,2), (n,17)
        if kpts is None or len(kpts) == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return np.concatenate([kpts, scores[..., None]], axis=-1).astype(np.float32)

    return estimate


def _make_yolo_estimator(
    *, device: str | None, model_name: str
) -> Callable[[np.ndarray], np.ndarray]:
    from ultralytics import YOLO

    from kodokan.config import models_dir

    # Keep weights out of the cwd/repo: resolve bare names under the data models dir.
    weight = Path(model_name)
    if not weight.is_absolute() and weight.parent == Path("."):
        weight = models_dir() / model_name
    model = YOLO(str(weight))

    def estimate(frame_bgr: np.ndarray) -> np.ndarray:
        r = model(frame_bgr, verbose=False, device=device or "mps")[0]
        if (
            r.keypoints is None
            or r.keypoints.data is None
            or len(r.keypoints.data) == 0
        ):
            return np.zeros((0, 17, 3), dtype=np.float32)
        return r.keypoints.data.cpu().numpy().astype(np.float32)  # (n,17,3)

    return estimate


_BACKENDS = {
    "rtmlib": _make_rtmlib_estimator,
    "ultralytics": _make_yolo_estimator,
}

_DEFAULT_BACKEND_KW = {
    "rtmlib": {"mode": "balanced"},
    "ultralytics": {"model_name": "yolo11n-pose.pt"},
}


def _select_persons(dets: np.ndarray, n_persons: int, conf_thresh: float) -> np.ndarray:
    """Keep the ``n_persons`` highest-confidence detections, ordered left→right.

    Returns an ``(n_persons, 17, 3)`` array, ``NaN``-padded for empty slots.
    """
    out = np.full((n_persons, 17, 3), np.nan, dtype=np.float32)
    if dets is None or len(dets) == 0:
        return out
    person_score = np.nanmean(dets[..., 2], axis=1)
    keep = np.argsort(person_score)[::-1]
    keep = [i for i in keep if person_score[i] >= conf_thresh][:n_persons]
    if not keep:
        return out
    chosen = dets[keep]
    mean_x = np.nanmean(chosen[..., 0], axis=1)
    chosen = chosen[np.argsort(mean_x)]  # left→right
    out[: len(chosen)] = chosen
    return out


def _video_meta(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, n_frames, width, height) via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return float(fps), n, w, h


def estimate_poses(
    video_path: PathLike,
    *,
    backend: str = "rtmlib",
    n_persons: int = 2,
    conf_thresh: float = 0.3,
    frame_step: int = 1,
    frame_range: tuple[int, int] | None = None,
    device: str | None = None,
    source_url: str | None = None,
    progress: bool = True,
    **backend_kwargs,
) -> PoseSequence:
    """Estimate per-frame, per-person COCO-17 keypoints for a video clip.

    Simplest use: ``estimate_poses("clip.mp4")`` → a :class:`PoseSequence` with the
    two highest-confidence people per frame (left→right ordered) via RTMPose.

    Args:
        video_path: Path to the video clip.
        backend: ``"rtmlib"`` (RTMPose top-down, CPU/ONNX; default) or
            ``"ultralytics"`` (YOLO11-pose, MPS).
        n_persons: Number of person slots to keep per frame (2 for tori+uke).
        conf_thresh: Minimum mean per-person confidence to keep a detection.
        frame_step: Analyze every ``frame_step``-th frame (1 = every frame).
        frame_range: Optional ``(start, stop)`` frame index window.
        device: Backend device (``"cpu"``/``"mps"``); backend default if ``None``.
        source_url: Canonical source URL to record as provenance.
        progress: Print a progress line periodically.
        **backend_kwargs: Forwarded to the backend (e.g. ``mode="performance"``
            for rtmlib, or ``model_name="yolo11m-pose.pt"`` for ultralytics).

    Returns:
        A :class:`PoseSequence` of shape ``(F, n_persons, 17, 3)``.
    """
    import cv2

    video_path = str(video_path)
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {sorted(_BACKENDS)}"
        )

    fps, n_total, width, height = _video_meta(video_path)
    start, stop = frame_range or (0, n_total or 10**9)

    kw = {**_DEFAULT_BACKEND_KW[backend], **backend_kwargs}
    estimate = _BACKENDS[backend](device=device, **kw)

    cap = cv2.VideoCapture(video_path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    per_frame: list[np.ndarray] = []
    indices: list[int] = []
    idx = start
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % frame_step == 0:
            dets = estimate(frame)
            per_frame.append(_select_persons(dets, n_persons, conf_thresh))
            indices.append(idx)
            if progress and len(indices) % 50 == 0:
                print(
                    f"  [{backend}] analyzed {len(indices)} frames (frame {idx})",
                    flush=True,
                )
        idx += 1
    cap.release()

    keypoints = (
        np.stack(per_frame, axis=0)
        if per_frame
        else np.empty((0, n_persons, 17, 3), dtype=np.float32)
    )
    if progress:
        print(f"  [{backend}] done: {len(indices)} frames", flush=True)
    return PoseSequence(
        keypoints=keypoints,
        frame_indices=np.asarray(indices, dtype=int),
        fps=fps,
        width=width or (keypoints.shape and 0),
        height=height,
        backend=backend,
        video_path=video_path,
        source_url=source_url,
    )
