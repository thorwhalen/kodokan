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

import warnings
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
    stale_after: int = 12,
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
            r = model.track(
                frame, persist=True, tracker=tracker, verbose=False, device=device
            )[0]
            d: dict[int, np.ndarray] = {}
            if (
                r.boxes is not None
                and r.boxes.id is not None
                and r.keypoints is not None
            ):
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

    # Per-frame assignment into n_persons stable slots, fusing two cues:
    #   (1) BoT-SORT track-id continuity (robust through crossings), then
    #   (2) spatial nearest-centroid continuity (survives track fragmentation),
    #   (3) lazy left->right initialization of still-empty slots.
    # This avoids the "two temporally-disjoint dominant tracks" failure that a
    # global top-2-by-presence binding hits on long, fragment-heavy clips.
    F = len(per_frame)
    out = np.full((F, n_persons, 17, 3), np.nan, dtype=np.float32)
    max_gap_px = max_gap_frac * float(width or 1920)

    def _centroid(kp: np.ndarray) -> np.ndarray:
        return np.nanmean(kp[:, :2], axis=0)

    slot_centroid: list[np.ndarray | None] = [None] * n_persons
    slot_tid: list[int | None] = [None] * n_persons
    slot_missing: list[int] = [10**9] * n_persons
    n_recover = 0

    for f, d in enumerate(per_frame):
        dets = [
            (tid, kp) for tid, kp in d.items() if np.nanmean(kp[:, 2]) >= conf_thresh
        ]
        det_c = [_centroid(kp) for _, kp in dets]
        used_slot: set[int] = set()
        used_det: set[int] = set()

        # (1) identity continuity: a bound track id reappears
        for si in range(n_persons):
            if slot_tid[si] is None:
                continue
            for di, (tid, _) in enumerate(dets):
                if di in used_det or tid != slot_tid[si]:
                    continue
                out[f, si] = dets[di][1]
                slot_centroid[si] = det_c[di]
                used_slot.add(si)
                used_det.add(di)
                break

        # (2) spatial continuity: nearest unused det to each *fresh* initialized empty slot (gated)
        fresh = [
            si
            for si in range(n_persons)
            if si not in used_slot
            and slot_centroid[si] is not None
            and slot_missing[si] < stale_after
        ]
        cand = sorted(
            (float(np.linalg.norm(slot_centroid[si] - det_c[di])), si, di)
            for si in fresh
            for di in range(len(dets))
            if di not in used_det
        )
        for dist_val, si, di in cand:
            if si in used_slot or di in used_det or dist_val > max_gap_px:
                continue
            out[f, si] = dets[di][1]
            slot_centroid[si] = det_c[di]
            slot_tid[si] = dets[di][0]  # re-bind: id may have changed after a fragment
            used_slot.add(si)
            used_det.add(di)
            n_recover += 1

        # (3) re-acquire / initialize: stale or never-initialized empty slots grab leftover dets
        for si in range(n_persons):
            if (
                si not in used_slot
                and slot_centroid[si] is not None
                and slot_missing[si] >= stale_after
            ):
                slot_centroid[si] = (
                    None  # forget a long-lost target so the slot can re-acquire
                )
        uninit = sorted(
            si
            for si in range(n_persons)
            if si not in used_slot and slot_centroid[si] is None
        )
        leftover = sorted(
            (di for di in range(len(dets)) if di not in used_det),
            key=lambda di: det_c[di][0],
        )
        for si, di in zip(uninit, leftover):
            out[f, si] = dets[di][1]
            slot_centroid[si] = det_c[di]
            slot_tid[si] = dets[di][0]
            used_slot.add(si)
            used_det.add(di)

        # (4) update miss counters
        for si in range(n_persons):
            slot_missing[si] = 0 if si in used_slot else slot_missing[si] + 1

    if progress:
        present = ~np.all(np.isnan(out[..., 0]), axis=2)
        print(
            f"  [track] both-present {float((present.sum(1) == n_persons).mean()):.0%}"
            f"  (spatial recoveries: {n_recover})",
            flush=True,
        )
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


def identity_swap_rate(pose_seq: PoseSequence) -> dict:
    """Ground-truth-free track-identity discontinuity rate (a tracking-quality metric).

    For each consecutive frame pair in which *all* person slots are present, we
    solve the minimum-cost assignment between the two frames' person centroids.
    When the optimal assignment is not the identity (slot *i* → slot *i*), the
    slots' spatial positions are better explained by a **label swap** — a proxy
    for a tori/uke identity swap that the review asked us to instrument. A stable
    tracker trends toward ``0``.

    Returns ``{n_pairs, n_swaps, swap_rate}``.

    Caveat (honest): this also fires on a *genuine* physical crossing that the
    tracker correctly follows through, so read ``swap_rate`` as a relative
    diagnostic (compare trackers/clips), not an absolute error count.
    """
    from scipy.optimize import linear_sum_assignment

    with warnings.catch_warnings():  # all-NaN person slots are expected (guarded below)
        warnings.simplefilter("ignore", RuntimeWarning)
        cent = np.nanmean(pose_seq.keypoints[..., :2], axis=2)  # (F, P, 2)
    present = np.all(np.isfinite(cent), axis=2)  # (F, P)
    P = cent.shape[1]
    ident = np.arange(P)
    n_pairs = n_swaps = 0
    for f in range(len(cent) - 1):
        if not (present[f].all() and present[f + 1].all()):
            continue
        cost = np.linalg.norm(cent[f, :, None, :] - cent[f + 1, None, :, :], axis=2)
        _, col = linear_sum_assignment(cost)
        n_pairs += 1
        n_swaps += not np.array_equal(col, ident)
    return {
        "n_pairs": n_pairs,
        "n_swaps": int(n_swaps),
        "swap_rate": round(n_swaps / n_pairs, 4) if n_pairs else 0.0,
    }
