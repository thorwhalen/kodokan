"""Stage 1 demo: compare demonstrations of Seoi-nage via joint-angle DTW.

Loads the full pose sequence + detected demo segments, then:
  1. proves DTW speed-invariance (a demo vs a 1.5x time-stretched copy of itself),
  2. builds a demo x demo normalized-DTW distance matrix (heatmap PNG),
  3. plots per-angle deviation for a chosen pair (PNG).

Usage::
    PYTHONPATH=<repo> python examples/compare_demos.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kodokan import compare as C
from kodokan import config
from kodokan.pose import PoseSequence


def main():
    seq = PoseSequence.load_npz(config.pose_dir() / "seoi-nage_ultralytics_full.npz")
    segs = json.loads((config.pose_dir() / "seoi-nage_segments.json").read_text())["demos"]

    # angle features per demo (primary person within the demo window)
    feats, labels = [], []
    for d in segs:
        fr = (d["start_frame"], d["end_frame"])
        person = C.primary_person(seq, frame_range=fr)
        f = C._clean(C.angle_features(seq, person=person, frame_range=fr))
        feats.append(f)
        labels.append(f"{d['start_s']:.0f}-{d['end_s']:.0f}s")
        print(f"demo {d['index']:2d} {labels[-1]:>10}  person{person}  clean_frames={len(f)}")

    # --- 1) speed-invariance: demo vs 1.5x time-stretched self, vs a different demo ---
    # choose two reasonably long clean demos
    long_idx = sorted(range(len(feats)), key=lambda i: -len(feats[i]))[:2]
    i, j = long_idx[0], long_idx[1]
    self_d = C.compare(feats[i], feats[i])["normalized"]
    stretch_d = C.compare(feats[i], C.time_stretch(feats[i], 1.5))["normalized"]
    cross_d = C.compare(feats[i], feats[j])["normalized"]
    print("\n--- DTW speed-invariance check (normalized distance) ---")
    print(f"  demo {labels[i]} vs itself          : {self_d:.4f}")
    print(f"  demo {labels[i]} vs 1.5x-stretched   : {stretch_d:.4f}   (should stay ~as low as self)")
    print(f"  demo {labels[i]} vs demo {labels[j]} : {cross_d:.4f}   (a genuinely different rep)")

    # --- 2) distance matrix heatmap ---
    D = C.distance_matrix(feats)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Seoi-nage demos — pairwise normalized-DTW distance\n(joint-angle features; darker = more similar)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="DTW distance / step (rad)")
    fig.tight_layout()
    p1 = config.viz_dir() / "seoi-nage_demo_distance_matrix.png"
    fig.savefig(p1, dpi=130)
    print("\nsaved", p1.name)

    # --- 3) per-angle deviation for the (i, j) pair ---
    res = C.compare(feats[i], feats[j])
    dev = np.degrees(C.per_angle_deviation(res))  # to degrees for readability
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8, 7))
    # representative angle (right elbow) along the DTW alignment
    a, b, path = res["a"], res["b"], res["path"]
    re = C.ANGLE_NAMES.index("r_elbow")
    axA.plot(np.degrees([a[p][re] for p, _ in path]), label=f"demo {labels[i]}")
    axA.plot(np.degrees([b[q][re] for _, q in path]), label=f"demo {labels[j]}")
    axA.set_title("Right-elbow angle along the DTW alignment")
    axA.set_xlabel("aligned step")
    axA.set_ylabel("angle (deg)")
    axA.legend(fontsize=8)
    axB.bar(C.ANGLE_NAMES, dev, color="#c0392b")
    axB.set_title(f"Mean per-joint-angle deviation: demo {labels[i]} vs {labels[j]}")
    axB.set_ylabel("deg")
    axB.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p2 = config.viz_dir() / "seoi-nage_demo_pair_deviation.png"
    fig.savefig(p2, dpi=130)
    print("saved", p2.name)


if __name__ == "__main__":
    main()
