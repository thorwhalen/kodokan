"""Evaluate viewpoint-invariant 3D joint-angle features (does 3D beat 2D?).

Reads the MediaPipe 3D world-landmark lifts (``{video_id}_3d.npz`` in the bridge
dir), computes 3D joint angles (viewpoint-invariant) per demo for the primary
person, and runs the SAME medoid-reference / technique-ID / separation-AUC harness
as examples/eval_features.py — so 3D is directly comparable to the 2D baseline.

Usage::
    PYTHONPATH=<repo> python examples/eval_features_3d.py
"""

import json
from pathlib import Path

import numpy as np

from kodokan import store
from kodokan.compare import compare
from kodokan.score import build_reference

BRIDGE = Path.home() / ".kodokan_mp_bridge"
MIN_2P = 0.4

# BlazePose-33 angle triples (name, a, vertex, c) — same joints as the 2D baseline.
ANGLES_3D = (
    ("l_elbow", 11, 13, 15),
    ("r_elbow", 12, 14, 16),
    ("l_shoulder", 13, 11, 23),
    ("r_shoulder", 14, 12, 24),
    ("l_hip", 11, 23, 25),
    ("r_hip", 12, 24, 26),
    ("l_knee", 23, 25, 27),
    ("r_knee", 24, 26, 28),
)


def _angles3d(lm33):
    xyz = lm33[:, :3]
    out = np.empty(len(ANGLES_3D))
    for i, (_, a, b, c) in enumerate(ANGLES_3D):
        ba, bc = xyz[a] - xyz[b], xyz[c] - xyz[b]
        nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
        out[i] = np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1)) if nba > 1e-6 and nbc > 1e-6 else np.nan
    return out


def _demo_feats(world, fi, fps, start_s, end_s):
    m = (fi >= int(start_s * fps)) & (fi <= int(end_s * fps))
    w = world[m]  # (f, P, 33, 4)
    if len(w) == 0:
        return np.empty((0, len(ANGLES_3D)))
    present = ~np.all(np.isnan(w[..., 0]), axis=2)  # (f, P)
    p = int(np.argmax(present.mean(0)))
    feat = np.stack([_angles3d(w[k, p]) for k in range(len(w))])
    return feat[~np.any(np.isnan(feat), axis=1)]


def main():
    ss = store.segments_store()
    techs = []
    for npz in sorted(BRIDGE.glob("*_3d.npz")):
        vid = npz.name[: -len("_3d.npz")]
        d = np.load(npz, allow_pickle=False)
        world, fi, fps = d["world"], d["frame_indices"], float(d["fps"])
        rec = ss[vid] if vid in ss else {}
        demos = [dm for dm in rec.get("demos", []) if dm.get("two_person_frac", 0) >= MIN_2P]
        feats = [_demo_feats(world, fi, fps, dm["start_s"], dm["end_s"]) for dm in demos]
        feats = [f for f in feats if len(f) >= 4]
        if len(feats) >= 3:
            techs.append({"vid": vid, "feats": feats, "ref": build_reference(feats)})

    n = len(techs)
    M = np.zeros((n, n))
    gen, imp = [], []
    for i, ti in enumerate(techs):
        for j, tj in enumerate(techs):
            ds = [compare(f, tj["ref"]["reference"])["normalized"] for f in ti["feats"]]
            M[i, j] = float(np.mean(ds))
            (gen if i == j else imp).extend(ds)
    acc = float(np.mean([int(np.argmin(M[i]) == i) for i in range(n)])) if n else 0.0
    gen = np.array([g for g in gen if g > 1e-9])
    imp = np.array(imp)
    auc = float(np.mean(imp[:, None] > gen[None, :])) if len(gen) and len(imp) else float("nan")

    print(json.dumps({
        "mode": "angles3d_mediapipe",
        "n_tech": n,
        "feat_dim": len(ANGLES_3D),
        "accuracy": round(acc, 3),
        "genuine_median": round(float(np.median(gen)), 4) if len(gen) else None,
        "impostor_median": round(float(np.median(imp)), 4) if len(imp) else None,
        "auc": round(auc, 3),
    }))


if __name__ == "__main__":
    main()
