"""Batch the pipeline over several techniques into the dol stores.

Downloads a range of playlist techniques (idempotent via a yt-dlp download
archive), then for each clip runs tracked two-person pose and demo segmentation,
writing results to the Parquet pose store and JSON segments store. Prints a
summary table.

Usage::
    PYTHONPATH=<repo> python examples/batch_pipeline.py --items 2:11
"""

import argparse
from pathlib import Path

import numpy as np

from kodokan import config, store
from kodokan.acquire import download_techniques
from kodokan.segment import segment_demonstrations
from kodokan.track import estimate_poses_tracked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="2:11", help="yt-dlp 1-based playlist selector (#002-#011)")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    archive = config.clips_dir() / ".download_archive.txt"
    print(f"[acquire] playlist items {args.items} (archive: {archive.name}) ...", flush=True)
    results = download_techniques(playlist_items=args.items, download_archive=str(archive))
    print(f"[acquire] {len(results)} clips", flush=True)

    ps = store.pose_store()
    ss = store.segments_store()
    summary = []

    for r in results:
        vid, title, url = r.info.get("id"), r.info.get("title"), r.info.get("webpage_url")
        if not r.path or not Path(r.path).exists():
            print(f"[skip] {vid} {title!r}: file missing", flush=True)
            continue
        print(f"\n=== {vid}  {title} ===", flush=True)
        seq = estimate_poses_tracked(
            r.path, frame_step=args.frame_step, device=args.device, source_url=url, progress=True
        )
        ps[vid] = seq
        segs = segment_demonstrations(
            seq, active_quantile=0.55, smooth_sigma=5, min_duration_s=1.5, merge_gap_s=0.6
        )
        ss[vid] = {
            "video_id": vid,
            "technique": title,
            "source_url": url,
            "fps": seq.fps,
            "n_demos": len(segs),
            "demos": [
                dict(index=s.index, start_s=s.start_s, end_s=s.end_s, duration_s=round(s.duration_s, 2))
                for s in segs
            ],
        }
        present = ~np.all(np.isnan(seq.keypoints[..., 0]), axis=2)
        both = f"{(present.sum(1) == 2).mean():.0%}"
        summary.append((vid, title, seq.n_frames, both, len(segs)))
        print(f"  stored {seq.keypoints.shape}  both-present={both}  demos={len(segs)}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'video_id':12} {'frames':>6} {'2p%':>4} {'demos':>5}  technique", flush=True)
    for vid, title, nf, p2, nd in summary:
        print(f"{vid:12} {nf:>6} {p2:>4} {nd:>5}  {title}", flush=True)
    print(f"\npose_store: {len(ps)} clips | segments_store: {len(ss)} clips", flush=True)


if __name__ == "__main__":
    main()
