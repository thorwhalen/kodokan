"""Segment a clip into its repeated demonstrations of one throw.

A Kodokan demo clip shows the *same* throw several times (varying speed/angle/
position), each demonstration a burst of whole-body motion bracketed by a
low-motion "reset" (walk back, re-grip, bow). Scene-cut detection finds nothing
in a continuous take, so we segment by **rhythm**: build a 1-D motion-energy
signal, then take the high-motion runs (separated by low-motion valleys) as the
demonstrations. This is training-free, speed/angle/position-invariant, and
degrades gracefully under the two-person occlusion that makes per-joint pose
noisy. See ``misc/docs/research-architecture.md`` §5.

Two energy sources, fusable:
- :func:`pose_motion_energy` — confidence-weighted total keypoint velocity
  (summed across people); needs only the :class:`~kodokan.pose.PoseSequence`.
- :func:`optical_flow_energy` — dense Farneback flow magnitude; needs the video
  but no pose, so it is robust where pose fails.

:func:`find_segments` turns an energy signal into ``(start_s, end_s)`` intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kodokan.pose import PoseSequence

PathLike = str | Path


@dataclass
class Segment:
    """One detected demonstration interval."""

    index: int
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    peak_activity: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def pose_motion_energy(pose_seq: PoseSequence, *, conf_thresh: float = 0.2) -> np.ndarray:
    """Per-frame confidence-weighted total keypoint speed, summed over all people.

    Returns an array aligned to ``pose_seq.frame_indices`` (first sample is 0).
    Robust to missing people/keypoints (NaNs are ignored).
    """
    kp = pose_seq.keypoints  # (F, P, K, 3)
    xy = kp[..., :2]
    conf = kp[..., 2]
    F = xy.shape[0]
    energy = np.zeros(F, dtype=np.float64)
    if F < 2:
        return energy
    disp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)  # (F-1, P, K)
    w = np.minimum(conf[1:], conf[:-1])
    w = np.where(w >= conf_thresh, w, 0.0)
    num = np.nansum(np.where(np.isfinite(disp), disp * w, 0.0), axis=(1, 2))  # (F-1,)
    den = np.nansum(w, axis=(1, 2)) + 1e-9
    energy[1:] = num / den
    return energy


def optical_flow_energy(
    video_path: PathLike,
    *,
    frame_range: tuple[int, int] | None = None,
    step: int = 1,
    scale: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean dense optical-flow magnitude (Farneback), pose-free.

    Returns ``(energy, frame_indices)``. Frames are downscaled by ``scale`` for speed.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start, stop = frame_range or (0, n_total or 10**9)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    prev = None
    energy: list[float] = []
    indices: list[int] = []
    idx = start
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            if prev is None:
                energy.append(0.0)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                energy.append(float(mag.mean()))
            prev = gray
            indices.append(idx)
        idx += 1
    cap.release()
    return np.asarray(energy), np.asarray(indices, dtype=int)


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(x, sigma)
    except Exception:  # pragma: no cover - scipy fallback
        k = max(1, int(sigma))
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        return np.convolve(x, kernel, mode="same")


def find_segments(
    energy: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    *,
    smooth_sigma: float = 4.0,
    active_quantile: float = 0.4,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.4,
) -> list[Segment]:
    """Turn a 1-D motion-energy signal into demonstration intervals.

    High-motion runs (energy above the ``active_quantile`` threshold) become
    demonstrations; short runs are dropped and near-adjacent runs merged.

    Args:
        energy: 1-D activity signal.
        frame_indices: Source frame index of each energy sample (same length).
        fps: Source frames-per-second.
        smooth_sigma: Gaussian smoothing (in samples) applied before thresholding.
        active_quantile: Energy quantile used as the active/rest threshold.
        min_duration_s: Drop demonstrations shorter than this.
        merge_gap_s: Merge demonstrations separated by less than this gap.

    Returns:
        A list of :class:`Segment` in time order.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.size == 0:
        return []
    sm = _smooth(energy, smooth_sigma)
    thr = np.quantile(sm, active_quantile)
    active = sm > thr

    # contiguous active runs -> (start_i, end_i) in sample space
    runs: list[list[int]] = []
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1

    def i2t(i: int) -> float:
        return float(frame_indices[i]) / float(fps)

    # merge near-adjacent runs
    merged: list[list[int]] = []
    for r in runs:
        if merged and i2t(r[0]) - i2t(merged[-1][1]) <= merge_gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    segments: list[Segment] = []
    for a, b in merged:
        start_s, end_s = i2t(a), i2t(b)
        if end_s - start_s < min_duration_s:
            continue
        segments.append(
            Segment(
                index=len(segments),
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                start_frame=int(frame_indices[a]),
                end_frame=int(frame_indices[b]),
                peak_activity=float(sm[a : b + 1].max()),
            )
        )
    return segments


def segment_demonstrations(
    pose_seq: PoseSequence,
    *,
    video_path: PathLike | None = None,
    use_optical_flow: bool = False,
    **find_kwargs,
) -> list[Segment]:
    """Convenience: build the motion-energy signal and find demonstration segments.

    Uses pose motion energy by default; set ``use_optical_flow=True`` (and pass
    ``video_path``) to fuse in flow energy (averaged with the pose signal).
    """
    pose_e = pose_motion_energy(pose_seq)
    energy = pose_e
    if use_optical_flow:
        if video_path is None:
            raise ValueError("use_optical_flow=True requires video_path")
        start = int(pose_seq.frame_indices[0])
        stop = int(pose_seq.frame_indices[-1]) + 1
        flow_e, _ = optical_flow_energy(video_path, frame_range=(start, stop))
        m = min(len(flow_e), len(pose_e))

        def _norm(a):
            a = a[:m]
            rng = a.max() - a.min()
            return (a - a.min()) / (rng + 1e-9)

        energy = (_norm(pose_e) + _norm(flow_e)) / 2.0
    return find_segments(energy, pose_seq.frame_indices[: len(energy)], pose_seq.fps, **find_kwargs)
