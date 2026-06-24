"""Stage 5 demo: reference-based scoring + a technique-ID confusion preview.

Builds a medoid reference per technique from the dataset, then:
  1. technique x technique mean-DTW confusion (nearest-reference identification),
  2. genuine-vs-impostor score distribution for one technique,
  3. per-joint / per-phase feedback for one query.

Usage::
    PYTHONPATH=<repo> python examples/score_demos.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from kodokan import store
from kodokan.compare import ANGLE_NAMES, compare
from kodokan.config import viz_dir
from kodokan.score import build_reference, demo_features, feedback, score

SEOI = "zIq0xI0ogxk"
MIN_2P = 0.4  # only score demos where both judoka are reasonably visible


def load_techniques(ps, ss):
    techs = []
    for vid in sorted(ps):
        seq = ps[vid]
        rec = ss[vid] if vid in ss else {}
        demos = [d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P]
        feats = [demo_features(seq, d["start_s"], d["end_s"]) for d in demos]
        feats = [f for f in feats if len(f) >= 4]
        if len(feats) >= 3:
            techs.append(
                {
                    "vid": vid,
                    "label": rec.get("technique", vid).split("/")[-1].strip(),
                    "feats": feats,
                    "ref": build_reference(feats),
                }
            )
    return techs


def main():
    ps, ss = store.pose_store(), store.segments_store()
    techs = load_techniques(ps, ss)
    n = len(techs)
    labels = [t["label"] for t in techs]
    print(f"techniques with >=3 clean demos: {n}")

    # (1) confusion: mean normalized-DTW distance from each technique's demos to each reference
    M = np.zeros((n, n))
    for i, ti in enumerate(techs):
        for j, tj in enumerate(techs):
            M[i, j] = float(np.mean([compare(f, tj["ref"]["reference"])["normalized"] for f in ti["feats"]]))
    correct = [int(np.argmin(M[i]) == i) for i in range(n)]
    acc = float(np.mean(correct))
    print(f"technique-ID accuracy (demos -> nearest reference): {acc:.0%}  ({sum(correct)}/{n})")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(n), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("scored against reference")
    ax.set_ylabel("demonstrations of")
    ax.set_title(f"Technique x reference mean DTW distance\nnearest-reference ID accuracy = {acc:.0%}")
    for i in range(n):
        j = int(np.argmin(M[i]))
        ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean DTW distance (lower = more similar)")
    fig.tight_layout()
    p1 = viz_dir() / "score_confusion_matrix.png"
    fig.savefig(p1, dpi=130)
    print("saved", p1.name)

    # (2)+(3) scoring + feedback for one technique (Seoi-nage)
    si = next((k for k, t in enumerate(techs) if t["vid"] == SEOI), 0)
    ref = techs[si]["ref"]
    genuine = [score(f, ref)["score"] for f in techs[si]["feats"]]
    impostor = [score(f, ref)["score"] for k, t in enumerate(techs) if k != si for f in t["feats"]]
    print(f"\n{labels[si]} reference — genuine scores: {sorted(round(g) for g in genuine)}")
    print(f"  genuine median {np.median(genuine):.0f} vs impostor median {np.median(impostor):.0f}")

    fb = feedback(techs[si]["feats"][0], ref)
    print(f"  feedback (demo 0): worst joint = {fb['worst_joint']}, per-phase deg = {fb['per_phase_deg']}")

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.5, 8))
    axA.hist(impostor, bins=12, alpha=0.6, label=f"other techniques (n={len(impostor)})", color="#c0392b")
    axA.hist(genuine, bins=8, alpha=0.8, label=f"{labels[si]} demos (n={len(genuine)})", color="#27ae60")
    axA.set_title(f"Scores vs the {labels[si]} reference — genuine vs impostor")
    axA.set_xlabel("score (0-100, calibrated to genuine spread)")
    axA.set_ylabel("count")
    axA.legend(fontsize=8)
    dev = list(fb["per_angle_deg"].values())
    axB.bar(ANGLE_NAMES, dev, color="#2980b9")
    axB.set_title(f"Per-joint-angle deviation vs reference ({labels[si]} demo 0)  |  phases(deg)={fb['per_phase_deg']}")
    axB.set_ylabel("deg")
    axB.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p2 = viz_dir() / "score_feedback.png"
    fig.savefig(p2, dpi=130)
    print("saved", p2.name)


if __name__ == "__main__":
    main()
