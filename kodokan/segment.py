"""Segment a clip into its repeated demonstrations of one throw.

A Kodokan demo clip shows the *same* throw several times (varying speed/angle/
position), each demonstration a burst of whole-body motion bracketed by a
low-motion "reset" (walk back, re-grip, bow). Scene-cut detection finds nothing
in a continuous take, so we segment by **rhythm**: build a 1-D motion-energy
signal and take the high-motion runs as demonstrations (research §5).

Robustness pieces:
- **Hysteresis thresholding** — a demo turns *on* above ``high_quantile`` and stays
  on until energy drops below ``low_quantile``. This stops *slow-motion* reps (whose
  absolute energy is low) from being split, while rests still separate demos.
- **Two-person gate** — a throw demonstration needs both tori and uke, so segments
  are filtered by their two-person coverage (drops single-person intro/close-up spans).
- **Self-similarity matrix + autocorrelation period** — a dependency-free,
  RepNet-style cross-check on the rep structure/count (soft, since reps vary in speed).

Energy sources (fusable): :func:`pose_motion_energy` (confidence-weighted total
keypoint speed; pose-only) and :func:`optical_flow_energy` (dense Farneback; pose-free).
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
    two_person_frac: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def pose_motion_energy(pose_seq: PoseSequence, *, conf_thresh: float = 0.2) -> np.ndarray:
    """Per-frame confidence-weighted total keypoint speed, summed over all people.

    Returns an array aligned to ``pose_seq.frame_indices`` (first sample is 0).
    Robust to missing people/keypoints (NaNs are ignored).
    """
    kp = pose_seq.keypoints
    xy, conf = kp[..., :2], kp[..., 2]
    F = xy.shape[0]
    energy = np.zeros(F, dtype=np.float64)
    if F < 2:
        return energy
    disp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)  # (F-1, P, K)
    w = np.minimum(conf[1:], conf[:-1])
    w = np.where(w >= conf_thresh, w, 0.0)
    num = np.nansum(np.where(np.isfinite(disp), disp * w, 0.0), axis=(1, 2))
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
                energy.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
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
    except Exception:  # pragma: no cover
        k = max(1, int(sigma))
        return np.convolve(x, np.ones(2 * k + 1) / (2 * k + 1), mode="same")


def find_segments(
    energy: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    *,
    smooth_sigma: float = 4.0,
    low_quantile: float = 0.25,
    high_quantile: float = 0.5,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.4,
) -> list[Segment]:
    """Turn a 1-D motion-energy signal into demonstration intervals via hysteresis.

    A demonstration starts when smoothed energy rises above the ``high_quantile``
    threshold and ends when it falls below the ``low_quantile`` threshold — so a
    slow-motion dip in the middle of a rep does not split it, while the low-motion
    rest between reps (below ``low_quantile``) does separate them.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.size == 0:
        return []
    sm = _smooth(energy, smooth_sigma)
    t_low, t_high = np.quantile(sm, low_quantile), np.quantile(sm, high_quantile)

    active = np.zeros(len(sm), dtype=bool)
    on = False
    for i, v in enumerate(sm):
        if not on and v >= t_high:
            on = True
        elif on and v < t_low:
            on = False
        active[i] = on

    runs: list[list[int]] = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1

    def i2t(k: int) -> float:
        return float(frame_indices[k]) / float(fps)

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


def self_similarity_matrix(
    pose_seq: PoseSequence, *, person: int | None = None, max_frames: int = 300
):
    """Frame×frame self-similarity on joint-angle features (for rep-structure viz).

    Returns ``(sim, frame_indices_sub)``; repeated demonstrations appear as
    off-diagonal bright blocks. Subsamples to ``max_frames`` for speed/readability.
    """
    from scipy.spatial.distance import cdist

    from kodokan.compare import angle_features, primary_person

    p = person if person is not None else primary_person(pose_seq)
    feats = angle_features(pose_seq, person=p)  # (F, D) with NaN where absent
    F = len(feats)
    if F == 0:
        return np.zeros((0, 0)), np.asarray([], dtype=int)
    step = max(1, F // max_frames)
    idx = np.arange(0, F, step)
    sub = feats[idx]
    col_mean = np.nanmean(np.where(np.isfinite(sub), sub, np.nan), axis=0)
    filled = np.where(np.isnan(sub), col_mean, sub)
    d = cdist(filled, filled)
    sim = np.exp(-d / (np.median(d) + 1e-9))
    return sim, pose_seq.frame_indices[idx]


def estimate_period(
    energy: np.ndarray,
    fps: float,
    *,
    smooth_sigma: float = 4.0,
    min_period_s: float = 1.5,
    max_period_s: float = 10.0,
) -> dict:
    """Soft rep-period/count cross-check via autocorrelation of the energy signal.

    Returns ``{period_s, count_est, strength}``. ``strength`` (0–1) is the
    autocorrelation at the chosen lag — low values mean weak periodicity (expected
    when reps vary a lot in speed), so treat the count as approximate.
    """
    sm = _smooth(np.asarray(energy, float), smooth_sigma)
    sm = sm - sm.mean()
    if sm.size < 4 or not np.any(sm):
        return {"period_s": None, "count_est": None, "strength": 0.0}
    ac = np.correlate(sm, sm, "full")[sm.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(min_period_s * fps), min(int(max_period_s * fps), len(ac) - 1)
    if hi <= lo:
        return {"period_s": None, "count_est": None, "strength": 0.0}
    lag = lo + int(np.argmax(ac[lo:hi]))
    period_s = lag / fps
    return {
        "period_s": round(period_s, 2),
        "count_est": round((len(sm) / fps) / period_s, 1),
        "strength": round(float(ac[lag]), 3),
    }


def segment_demonstrations(
    pose_seq: PoseSequence,
    *,
    video_path: PathLike | None = None,
    use_optical_flow: bool = False,
    min_two_person_frac: float = 0.0,
    **find_kwargs,
) -> list[Segment]:
    """Find demonstrations, annotate each with its two-person coverage, and gate.

    Uses pose motion energy by default (fuse optical flow with ``use_optical_flow``).
    Segments whose two-person coverage is below ``min_two_person_frac`` are dropped
    (a throw demonstration needs both tori and uke), then re-indexed in time order.
    """
    pose_e = pose_motion_energy(pose_seq)
    energy = pose_e
    if use_optical_flow:
        if video_path is None:
            raise ValueError("use_optical_flow=True requires video_path")
        start, stop = int(pose_seq.frame_indices[0]), int(pose_seq.frame_indices[-1]) + 1
        flow_e, _ = optical_flow_energy(video_path, frame_range=(start, stop))
        m = min(len(flow_e), len(pose_e))

        def _norm(a):
            a = a[:m]
            return (a - a.min()) / (a.max() - a.min() + 1e-9)

        energy = (_norm(pose_e) + _norm(flow_e)) / 2.0

    segs = find_segments(energy, pose_seq.frame_indices[: len(energy)], pose_seq.fps, **find_kwargs)

    present = ~np.all(np.isnan(pose_seq.keypoints[..., 0]), axis=2)  # (F, P)
    counts = present.sum(axis=1)
    fi = pose_seq.frame_indices
    kept: list[Segment] = []
    for s in segs:
        m = (fi >= s.start_frame) & (fi <= s.end_frame)
        frac = float((counts[m] == pose_seq.n_persons).mean()) if m.any() else 0.0
        s.two_person_frac = round(frac, 2)
        if frac >= min_two_person_frac:
            kept.append(s)
    for i, s in enumerate(kept):
        s.index = i
    return kept
