"""kodokan — study Kodokan Judo throws from video via body-pose analysis.

A pipeline over the official *Kodokan 100 Techniques* YouTube playlist: acquire
clips + metadata, estimate per-frame two-person skeletons, segment each clip into
its repeated demonstrations, visualize skeletons (overlay-on-video and on a blank
canvas), and — later — recognize, compare, and score throws.

Quick start::

    from kodokan.acquire import download_techniques
    from kodokan.pose import estimate_poses
    from kodokan.viz import render_skeleton_video

    res = download_techniques(playlist_items="2")          # Seoi-nage
    seq = estimate_poses(res[0].path, source_url=res[0].info["webpage_url"])
    render_skeleton_video(seq, out_path="overlay.mp4", source_video=res[0].path)
    render_skeleton_video(seq, out_path="skeleton.mp4", blank_canvas=True)

See ``misc/docs/research-architecture.md`` for the tool/architecture rationale.
"""

from kodokan.pose import estimate_poses, PoseSequence, COCO17_KEYPOINTS, COCO17_SKELETON
from kodokan.viz import render_skeleton_video, log_to_rerun

__all__ = [
    "estimate_poses",
    "PoseSequence",
    "COCO17_KEYPOINTS",
    "COCO17_SKELETON",
    "render_skeleton_video",
    "log_to_rerun",
]
