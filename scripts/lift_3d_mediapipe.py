"""Lift tracked 2D person tracks to 3D world landmarks via MediaPipe (per-crop).

Runs in the isolated MediaPipe venv (~/.kodokan_mp), NOT the main env. For each
bridge file written by examples/export_2d_bridge.py, it crops each tracked person's
bounding box per frame and runs MediaPipe Pose on the crop to get metric 3D world
landmarks (33 joints). 3D joint *angles* computed from these are viewpoint-invariant.

Output: ``{video_id}_3d.npz`` with ``world`` (F, P, 33, 4)=(x,y,z,visibility),
``frame_indices``, ``fps``.

Usage (venv python):
    ~/.kodokan_mp/bin/python scripts/lift_3d_mediapipe.py [--stride 2]
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose

BRIDGE = Path.home() / ".kodokan_mp_bridge"


def _bbox(kp17, W, H, margin=0.25, conf_thresh=0.3):
    xy, conf = kp17[:, :2], kp17[:, 2]
    v = xy[conf >= conf_thresh]
    if len(v) < 3:
        return None
    x0, y0 = v.min(0)
    x1, y1 = v.max(0)
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, int(x0 - margin * w)), max(0, int(y0 - margin * h))
    x1, y1 = min(W, int(x1 + margin * w)), min(H, int(y1 + margin * h))
    return (x0, y0, x1, y1) if (x1 - x0 > 10 and y1 - y0 > 10) else None


def lift_one(npz_path, poser, stride):
    d = np.load(npz_path, allow_pickle=False)
    kps, fi = d["keypoints"], d["frame_indices"]
    W, H, vp = int(d["width"]), int(d["height"]), str(d["video_path"])
    keep = np.arange(0, len(fi), stride)
    fi_sub = fi[keep]
    F, P = len(fi_sub), kps.shape[1]
    out = np.full((F, P, 33, 4), np.nan, dtype=np.float32)

    cap = cv2.VideoCapture(vp)
    src = -1
    for f, frame_idx in enumerate(fi_sub):
        frame = None
        while src < frame_idx:
            ok, frame = cap.read()
            src += 1
            if not ok:
                frame = None
                break
        if frame is None:
            continue
        kf = kps[keep[f]]  # (P, 17, 3)
        for p in range(P):
            bb = _bbox(kf[p], W, H)
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            res = poser.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            wl = res.pose_world_landmarks
            if wl is None:
                continue
            out[f, p] = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in wl.landmark], dtype=np.float32)
    cap.release()
    out_path = str(npz_path).replace(".npz", "_3d.npz")
    np.savez(out_path, world=out, frame_indices=fi_sub, fps=float(d["fps"]))
    present = ~np.all(np.isnan(out[..., 0]), axis=2)
    print(f"lifted {Path(npz_path).stem}: {F} frames, both-present {float((present.sum(1)==2).mean()):.0%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    poser = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)
    files = [f for f in sorted(glob.glob(str(BRIDGE / "*.npz"))) if not f.endswith("_3d.npz")]
    print(f"lifting {len(files)} clips (stride {args.stride})", flush=True)
    for npz in files:
        lift_one(npz, poser, args.stride)
    poser.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
