"""Export stored 2D pose + clip paths to a numpy bridge dir for the MediaPipe venv.

MediaPipe is ABI-incompatible in the main env, so the 3D lift runs in a separate
venv that only has numpy/opencv/mediapipe. This writes one ``{video_id}.npz`` per
clip (keypoints, frame_indices, fps, dims, video_path) the lifter can read without
importing kodokan/dol.

Usage::
    PYTHONPATH=<repo> python examples/export_2d_bridge.py
"""

from pathlib import Path

import numpy as np

from kodokan import config, store

BRIDGE = Path.home() / ".kodokan_mp_bridge"


def main():
    BRIDGE.mkdir(exist_ok=True)
    ps = store.pose_store()
    for vid in sorted(ps):
        seq = ps[vid]
        clip = next(config.clips_dir().glob(f"*({vid}).mp4"), None)
        if clip is None:
            print("no clip for", vid)
            continue
        np.savez(
            BRIDGE / f"{vid}.npz",
            keypoints=seq.keypoints,
            frame_indices=seq.frame_indices,
            fps=seq.fps,
            width=seq.width,
            height=seq.height,
            video_path=str(clip),
        )
        print("exported", vid, seq.keypoints.shape)
    print("bridge:", BRIDGE)


if __name__ == "__main__":
    main()
