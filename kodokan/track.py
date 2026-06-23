"""Stable tori/uke identity tracking (BoT-SORT / ByteTrack).

The plain :func:`~kodokan.pose.estimate_poses` keeps the top-2 detections and
orders them left→right *per frame*, so the two slots swap whenever tori and uke
cross. This module instead runs a multi-object tracker (Ultralytics' built-in
BoT-SORT by default — appearance ReID + camera-motion compensation) so each
person keeps a persistent ``track_id`` across frames; we then bind the two most
persistent tracks to fixed slots for the whole clip (ordered left→right by their
clip-average x), giving a stable identity that does not swap mid-throw.

Limitation: under the heavy mutual occlusion at a throw's apex a track can
fragment (a person reappears with a new id). Top-2-by-presence captures the
dominant fragments; fragment-merging/ReID-stitching is a later refinement (see
``misc/docs/research-architecture.md`` §4).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kodokan.pose import PoseSequence, _video_meta

PathLike = str | Path


def estimate_poses_tracked(
    video_path: PathLike,
    *,
    n_persons: int = 2,
    tracker: str = "botsort.yaml",
    conf_thresh: float = 0.3,
    max_gap_frac: float = 0.15,
    frame_step: int = 1,
    frame_range: tuple[int, int] | None = None,
    device: str | None = "mps",
    model_name: str = "yolo11n-pose.pt",
    source_url: str | None = None,
    progress: bool = True,
) -> PoseSequence:
    """Estimate per-frame keypoints with persistent tori/uke identity.

    Returns a :class:`~kodokan.pose.PoseSequence` whose person slots are *stable*
    across the clip (slot 0 = the track that is, on average, further left).

    Args:
        video_path: Path to the clip.
        n_persons: Number of stable identity slots to keep (2 for tori+uke).
        tracker: Ultralytics tracker config (``"botsort.yaml"`` or ``"bytetrack.yaml"``).
        conf_thresh: Minimum mean per-person confidence to count a detection.
        frame_step: Analyze every n-th frame.
        frame_range: Optional ``(start, stop)`` frame window.
        device: Torch device (``"mps"``/``"cpu"``).
        model_name: YOLO-pose weights (resolved under the data models dir).
        source_url: Provenance URL.
        progress: Print progress.
    """
    import cv2
    from ultralytics import YOLO

    from kodokan.config import models_dir

    weight = Path(model_name)
    if not weight.is_absolute() and weight.parent == Path("."):
        weight = models_dir() / model_name
    model = YOLO(str(weight))

    fps, n_total, width, height = _video_meta(str(video_path))
    start, stop = frame_range or (0, n_total or 10**9)

    cap = cv2.VideoCapture(str(video_path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # pass 1: per-frame {track_id: (17,3)}
    per_frame: list[dict[int, np.ndarray]] = []
    indices: list[int] = []
    idx = start
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % frame_step == 0:
            r = model.track(frame, persist=True, tracker=tracker, verbose=False, device=device)[0]
            d: dict[int, np.ndarray] = {}
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                ids = r.boxes.id.int().cpu().numpy()
                kk = r.keypoints.data.cpu().numpy()  # (n,17,3)
                for tid, kp in zip(ids, kk):
                    d[int(tid)] = kp.astype(np.float32)
            per_frame.append(d)
            indices.append(idx)
            if progress and len(indices) % 50 == 0:
                print(f"  [track] {len(indices)} frames (frame {idx})", flush=True)
        idx += 1
    cap.release()

    # choose the n_persons most-persistent tracks; order left->right by clip-mean x
    presence: Counter[int] = Counter()
    xs: dict[int, list[float]] = defaultdict(list)
    for d in per_frame:
        for tid, kp in d.items():
            if np.nanmean(kp[:, 2]) >= conf_thresh:
                presence[tid] += 1
                xs[tid].append(float(np.nanmean(kp[:, 0])))
    chosen = [tid for tid, _ in presence.most_common(n_persons)]
    chosen.sort(key=lambda t: np.mean(xs[t]) if xs[t] else 1e9)

    F = len(per_frame)
    out = np.full((F, n_persons, 17, 3), np.nan, dtype=np.float32)
    max_gap_px = max_gap_frac * float(width or 1920)

    def _centroid(kp: np.ndarray) -> np.ndarray:
        return np.nanmean(kp[:, :2], axis=0)

    last: list[np.ndarray | None] = [None] * n_persons
    for f, d in enumerate(per_frame):
        used: set[int] = set()
        # 1) place each bound track into its stable slot
        for slot, tid in enumerate(chosen):
            if tid in d and np.nanmean(d[tid][:, 2]) >= conf_thresh:
                out[f, slot] = d[tid]
                used.add(tid)
                last[slot] = _centroid(d[tid])
        # 2) gap-fill empty slots from the nearest unused detection (identity by continuity)
        avail = [
            (tid, kp) for tid, kp in d.items()
            if tid not in used and np.nanmean(kp[:, 2]) >= conf_thresh
        ]
        for slot in range(n_persons):
            if not np.all(np.isnan(out[f, slot])) or last[slot] is None or not avail:
                continue
            dists = [float(np.linalg.norm(_centroid(kp) - last[slot])) for _, kp in avail]
            k = int(np.argmin(dists))
            if dists[k] <= max_gap_px:
                tid, kp = avail.pop(k)
                out[f, slot] = kp
                used.add(tid)
                last[slot] = _centroid(kp)

    if progress:
        print(f"  [track] bound tracks {chosen} "
              f"(presence {[presence[t] for t in chosen]}/{F})", flush=True)
    return PoseSequence(
        keypoints=out,
        frame_indices=np.asarray(indices, dtype=int),
        fps=fps,
        width=width,
        height=height,
        backend=f"ultralytics+track:{tracker}",
        video_path=str(video_path),
        source_url=source_url,
    )
