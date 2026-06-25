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
from kodokan.acquire import (
    canonical_technique_key,
    download_source,
    download_techniques,
    local_clips,
    source_clips_dir,
)
from kodokan.segment import segment_demonstrations
from kodokan.track import estimate_poses_tracked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="2:11", help="yt-dlp 1-based playlist selector (#002-#011)")
    ap.add_argument("--source", default="kodokan_ijf", help="registered source key (e.g. efficient_judo)")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--force", action="store_true", help="reprocess clips already in the pose store")
    ap.add_argument("--download", action="store_true", help="(re)download before processing")
    args = ap.parse_args()

    cdir = source_clips_dir(args.source)
    if args.download:
        archive = cdir / ".download_archive.txt"
        print(f"[acquire] {args.source} items {args.items} ...", flush=True)
        if args.source == "kodokan_ijf":
            download_techniques(playlist_items=args.items, download_dir=cdir, download_archive=str(archive))
        else:
            download_source(args.source, playlist_items=args.items, download_dir=cdir, download_archive=str(archive))
    clips = local_clips(cdir)  # process whatever is on disk for this source
    print(f"[acquire] {len(clips)} {args.source} clips on disk", flush=True)

    ps = store.pose_store()
    ss = store.segments_store()
    summary = []

    for c in clips:
        vid, title, url, path = c["id"], c["title"], c["webpage_url"], c["path"]
        if not path or not Path(path).exists():
            print(f"[skip] {vid} {title!r}: file missing", flush=True)
            continue
        if not args.force and vid in ps:
            print(f"[skip] {vid} {title!r}: already in pose store", flush=True)
            continue
        print(f"\n=== {vid}  {title} ===", flush=True)
        try:
            seq = estimate_poses_tracked(
                path, frame_step=args.frame_step, device=args.device, source_url=url, progress=True
            )
            ps[vid] = seq  # cached incrementally -> the run is resumable on re-launch
            segs = segment_demonstrations(
                seq, smooth_sigma=5, low_quantile=0.25, high_quantile=0.5,
                min_duration_s=1.5, merge_gap_s=0.6, min_two_person_frac=0.3,
            )
            ss[vid] = {
                "video_id": vid,
                "technique": title,
                "technique_key": canonical_technique_key(title),
                "source": args.source,
                "source_url": url,
                "fps": seq.fps,
                "n_demos": len(segs),
                "demos": [
                    dict(index=s.index, start_s=s.start_s, end_s=s.end_s,
                         duration_s=round(s.duration_s, 2), two_person_frac=s.two_person_frac)
                    for s in segs
                ],
            }
            present = ~np.all(np.isnan(seq.keypoints[..., 0]), axis=2)
            both = f"{(present.sum(1) == 2).mean():.0%}"
            summary.append((vid, title, seq.n_frames, both, len(segs)))
            print(f"  stored {seq.keypoints.shape}  both-present={both}  demos={len(segs)}", flush=True)
        except Exception as e:  # one bad clip must not kill a multi-hour run
            print(f"[error] {vid} {title!r}: {type(e).__name__}: {e}", flush=True)
            continue

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'video_id':12} {'frames':>6} {'2p%':>4} {'demos':>5}  technique", flush=True)
    for vid, title, nf, p2, nd in summary:
        print(f"{vid:12} {nf:>6} {p2:>4} {nd:>5}  {title}", flush=True)
    print(f"\npose_store: {len(ps)} clips | segments_store: {len(ss)} clips", flush=True)


if __name__ == "__main__":
    main()
