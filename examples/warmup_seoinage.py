"""Warm-up: Seoi-nage pose bake-off + overlay/blank-canvas render.

Extracts two-person COCO-17 keypoints for a window of the Seoi-nage clip with a
chosen backend, saves an NPZ, and bakes two MP4s (overlay-on-video and
skeleton-on-blank-canvas).

Usage::

    PYTHONPATH=<kodokan-repo> python examples/warmup_seoinage.py --backend ultralytics --device mps
    PYTHONPATH=<kodokan-repo> python examples/warmup_seoinage.py --backend rtmlib --device cpu --start 300 --stop 700
"""

import argparse
from pathlib import Path

import numpy as np

from kodokan import config
from kodokan.pose import estimate_poses
from kodokan.viz import render_skeleton_video

SEOI_URL = "https://www.youtube.com/watch?v=zIq0xI0ogxk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "rtmlib"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    mp4 = next((config.clips_dir()).glob("*Seoi-nage*.mp4"))
    frame_range = (args.start, args.stop) if args.start is not None else None
    tag = args.tag or f"{args.backend}{f'_{args.start}-{args.stop}' if frame_range else '_full'}"

    print(f"clip: {mp4.name}  backend={args.backend} device={args.device} range={frame_range}")
    seq = estimate_poses(
        mp4, backend=args.backend, n_persons=2, frame_range=frame_range,
        device=args.device, source_url=SEOI_URL,
    )
    valid = ~np.all(np.isnan(seq.keypoints[..., 0]), axis=2)  # (F,P)
    print(f"seq {seq.keypoints.shape}  2-person frames: {int((valid.sum(1)==2).sum())}/{seq.n_frames}")

    npz = seq.save_npz(config.pose_dir() / f"seoi-nage_{tag}.npz")
    print("saved", npz.name)

    vd = config.viz_dir()
    ov = render_skeleton_video(seq, out_path=vd / f"seoi-nage_overlay_{tag}.mp4", source_video=mp4)
    bk = render_skeleton_video(seq, out_path=vd / f"seoi-nage_skeleton_{tag}.mp4", blank_canvas=True)
    for p in (ov, bk):
        print("rendered", p.name, round(p.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
