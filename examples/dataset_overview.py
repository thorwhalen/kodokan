"""Summarize the built dataset: stats + a contact sheet (one annotated frame/technique).

Reads the dol pose + segments stores, prints corpus stats, and writes a grid PNG
with one representative annotated frame per technique.

Usage::
    PYTHONPATH=<repo> python examples/dataset_overview.py
"""

import math

import cv2
import numpy as np

from kodokan import config, store
from kodokan.pose import COCO17_SKELETON
from kodokan.viz import DEFAULT_PERSON_COLORS, _draw_persons


def _find_clip(vid: str):
    g = list(config.clips_dir().glob(f"*({vid}).mp4"))
    return g[0] if g else None


def _representative_k(seq, segs) -> int:
    """Index (into analyzed frames) of a representative 2-person frame."""
    demos = (segs or {}).get("demos") or []
    if demos:
        mid_t = (demos[0]["start_s"] + demos[0]["end_s"]) / 2
        target = int(mid_t * seq.fps)
    else:
        target = int(seq.frame_indices[len(seq.frame_indices) // 2])
    return int(np.argmin(np.abs(seq.frame_indices - target)))


def _tile(vid, seq, segs, size=(480, 270)):
    k = _representative_k(seq, segs)
    clip = _find_clip(vid)
    frame = None
    if clip is not None:
        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seq.frame_indices[k]))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None
    if frame is None:
        frame = np.zeros((seq.height or 1080, seq.width or 1920, 3), np.uint8)
    _draw_persons(frame, seq.keypoints[k], COCO17_SKELETON, DEFAULT_PERSON_COLORS, 0.3, 3, 6)
    tile = cv2.resize(frame, size)
    label = (segs or {}).get("technique", vid)
    cv2.rectangle(tile, (0, 0), (size[0], 24), (0, 0, 0), -1)
    cv2.putText(tile, label[:48], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main():
    ps, ss = store.pose_store(), store.segments_store()
    vids = sorted(ps)
    print(f"clips in pose_store: {len(vids)}")
    tiles, total_demos, total_frames = [], 0, 0
    print(f"\n{'video_id':12} {'frames':>6} {'2p%':>4} {'demos':>5}  technique")
    for vid in vids:
        seq = ps[vid]
        segs = ss[vid] if vid in ss else {}
        present = ~np.all(np.isnan(seq.keypoints[..., 0]), axis=2)
        both = f"{(present.sum(1) == 2).mean():.0%}"
        nd = segs.get("n_demos", 0)
        total_demos += nd
        total_frames += seq.n_frames
        print(f"{vid:12} {seq.n_frames:>6} {both:>4} {nd:>5}  {segs.get('technique', '?')}")
        tiles.append(_tile(vid, seq, segs))

    print(f"\nTOTAL: {len(vids)} techniques, {total_demos} demonstrations, {total_frames} analyzed frames")

    if tiles:
        cols = min(4, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        h, w = tiles[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        out = config.viz_dir() / "dataset_overview.png"
        cv2.imwrite(str(out), sheet)
        print("saved", out)


if __name__ == "__main__":
    main()
