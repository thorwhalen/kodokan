"""Review/nudge aid: visualize demo segmentation for one clip.

Plots the motion-energy signal with the hysteresis thresholds and detected
demonstration spans, plus the joint-angle self-similarity matrix (repeated demos
show as off-diagonal blocks) and the autocorrelation period estimate. Edit the
segments JSON in the segments store to nudge boundaries by hand.

Usage::
    PYTHONPATH=<repo> python examples/segment_review.py [VIDEO_ID]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kodokan import config, store
from kodokan.segment import (
    _smooth,
    estimate_period,
    pose_motion_energy,
    segment_demonstrations,
    self_similarity_matrix,
)

SEOI = "zIq0xI0ogxk"


def main():
    vid = sys.argv[1] if len(sys.argv) > 1 else SEOI
    seq = store.pose_store()[vid]
    fps = seq.fps
    t = seq.times()

    e = _smooth(pose_motion_energy(seq), 5)
    t_low, t_high = np.quantile(e, 0.25), np.quantile(e, 0.5)
    segs = segment_demonstrations(
        seq, smooth_sigma=5, low_quantile=0.25, high_quantile=0.5,
        min_duration_s=1.5, merge_gap_s=0.6,
        min_two_person_frac=config.SEGMENT_MIN_TWO_PERSON_FRAC,
    )
    period = estimate_period(pose_motion_energy(seq), fps)
    sim, sub_fi = self_similarity_matrix(seq, max_frames=300)

    seg_store = store.segments_store()
    title = seg_store[vid]["technique"] if vid in seg_store else vid
    title = title.split("/")[-1].strip()  # romaji only (DejaVu lacks CJK glyphs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(t, e, color="#2c3e50", lw=1, label="motion energy (smoothed)")
    ax1.axhline(t_high, color="#27ae60", ls="--", lw=1, label="t_high (on)")
    ax1.axhline(t_low, color="#e67e22", ls="--", lw=1, label="t_low (off)")
    for s in segs:
        ax1.axvspan(s.start_s, s.end_s, color="#3498db", alpha=0.18)
        ax1.text((s.start_s + s.end_s) / 2, e.max() * 0.95, str(s.index),
                 ha="center", va="top", fontsize=8, color="#2980b9")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("energy")
    pstr = f"period≈{period['period_s']}s, count≈{period['count_est']} (strength {period['strength']})"
    ax1.set_title(f"{title} — {len(segs)} demonstrations (hysteresis + 2-person gate)\nautocorr {pstr}")
    ax1.legend(fontsize=8, loc="upper right")

    if sim.size:
        extent = [t[0], t[-1], t[-1], t[0]]
        im = ax2.imshow(sim, cmap="magma", extent=extent, aspect="auto")
        fig.colorbar(im, ax=ax2, shrink=0.8, label="joint-angle similarity")
    ax2.set_title("Self-similarity matrix (repeated demos = off-diagonal blocks)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("time (s)")

    fig.tight_layout()
    out = config.viz_dir() / f"segment_review_{vid}.png"
    fig.savefig(out, dpi=130)
    print(f"{title}: {len(segs)} demos, {pstr}")
    for s in segs:
        print(f"  demo {s.index}: {s.start_s:5.1f}-{s.end_s:5.1f}s ({s.duration_s:4.1f}s)  2p={s.two_person_frac:.0%}")
    print("saved", out)


if __name__ == "__main__":
    main()
