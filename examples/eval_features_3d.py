"""Evaluate 3D MediaPipe features (does 3D beat 2D?) with pelvis-frame canonicalization.

Reads the MediaPipe 3D world-landmark lifts (``{video_id}_3d.npz`` in the bridge
dir) and runs the SAME leak-free leave-one-demo-out medoid separability harness as
``examples/eval_features.py`` (``--feat angles`` | ``pos`` | ``both``).

Scope note (honest — this is only a *partial* answer to the review's 3D item):

- ``kodokan.canon3d.canonicalize_pose`` removes camera rotation, translation, and
  isotropic torso-scale. **Joint angles are already invariant to all three**, so
  canonicalization is a mathematical no-op for ``--feat angles`` (``--no-canon``
  gives identical angle numbers). It changes results **only** for ``--feat pos``/
  ``both`` (position features genuinely depend on the frame) — that is where the
  canon/raw contrast is meaningful.
- ``--lift square`` reads the letterboxed lifts (``scripts/lift_3d_mediapipe.py
  --letterbox``): square crops with real frame context around the person. (MediaPipe
  letterboxes internally, so the raw bbox crop is not stretched; what differs is how much
  real context the model sees — see that script's docstring.)
- ``--matched-2d`` is the fair 2D-vs-3D comparison (#10): the SAME clips, the SAME
  stride-subsampled frames (the 3D lift's ``frame_indices``), the SAME demos (kept only
  if both modalities have ≥4 clean frames) and the SAME person slot (2D
  ``primary_person`` rule over the demo window, applied to both — the lift keeps the
  2D track's slot). Only the modality differs, so the two numbers are comparable.

Usage::
    PYTHONPATH=<repo> python examples/eval_features_3d.py --feat pos [--no-canon] [--lift square]
    PYTHONPATH=<repo> python examples/eval_features_3d.py --matched-2d [--lift square]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kodokan import store
from kodokan.canon3d import BLAZEPOSE33, canonicalize_pose
from kodokan.compare import compare, joint_angles
from kodokan.config import EVAL_MIN_TWO_PERSON_FRAC as MIN_2P
from kodokan.recognize import loo_medoid_separability, paired_bootstrap_separability

BRIDGE = Path.home() / ".kodokan_mp_bridge"

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
        out[i] = (
            np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1))
            if nba > 1e-6 and nbc > 1e-6
            else np.nan
        )
    return out


_L_HIP, _R_HIP = BLAZEPOSE33.l_hip, BLAZEPOSE33.r_hip


def _frame_feature(lm33, *, feat, canon):
    """Per-frame feature vector for one person's BlazePose-33 landmarks."""
    parts = []
    if feat in ("angles", "both"):
        # Angles are invariant to canon's rigid+isotropic-scale transform, so they are
        # computed on the RAW landmarks regardless of `canon` (canonicalizing here would
        # be inert yet could needlessly drop degenerate frames as NaN).
        parts.append(_angles3d(lm33))
    if feat in ("pos", "both"):
        if canon:  # viewpoint-invariant: pelvis frame + torso-normalized
            pos = canonicalize_pose(lm33, joints=BLAZEPOSE33)
        else:  # camera-frame positions, only pelvis-centered (viewpoint-variant)
            pos = lm33[:, :3] - (lm33[_L_HIP, :3] + lm33[_R_HIP, :3]) / 2
        parts.append(pos.reshape(-1))
    return np.concatenate(parts)


def _demo_feats(world, fi, fps, start_s, end_s, *, feat, canon):
    m = (fi >= int(start_s * fps)) & (fi <= int(end_s * fps))
    w = world[m]  # (f, P, 33, 4)
    dim = {"angles": 8, "pos": 99, "both": 107}[feat]
    if len(w) == 0:
        return np.empty((0, dim))
    present = ~np.all(np.isnan(w[..., 0]), axis=2)  # (f, P)
    p = int(np.argmax(present.mean(0)))
    feats = np.stack(
        [_frame_feature(w[k, p], feat=feat, canon=canon) for k in range(len(w))]
    )
    return feats[~np.any(np.isnan(feats), axis=1)]


LIFT_SUFFIX = {"bbox": "_3d.npz", "square": "_3d_sq.npz"}


def _matched_demo(kp2d, world, fi, fps, start_s, end_s, *, min_frames=4):
    """2D and 3D angle features for one demo on identical frames + person, or None."""
    m = (fi >= int(start_s * fps)) & (fi <= int(end_s * fps))
    k2, w3 = kp2d[m], world[m]
    if len(k2) == 0:
        return None
    present = ~np.all(np.isnan(k2[..., 0]), axis=2)  # (f, P) — the 2D primary_person rule
    p = int(np.argmax(present.mean(0)))
    f2 = np.stack([joint_angles(k2[k, p]) for k in range(len(k2))])
    f3 = np.stack([_angles3d(w3[k, p]) for k in range(len(w3))])
    ok = ~(np.any(np.isnan(f2), axis=1) | np.any(np.isnan(f3), axis=1))  # same frames
    if ok.sum() < min_frames:
        return None
    return f2[ok], f3[ok]


def matched_2d_vs_3d(lift: str = "bbox") -> dict:
    """Separability of 2D vs 3D joint angles on identical clips/frames/demos/person."""
    ss = store.segments_store()
    by2, by3, n_frames = [], [], 0
    for npz in sorted(BRIDGE.glob("*" + LIFT_SUFFIX[lift])):
        vid = npz.name[: -len(LIFT_SUFFIX[lift])]
        d3 = np.load(npz, allow_pickle=False)
        d2 = np.load(BRIDGE / f"{vid}.npz", allow_pickle=False)
        fi, fps = d3["frame_indices"], float(d3["fps"])
        pos = {int(f): i for i, f in enumerate(d2["frame_indices"])}
        kp2d = d2["keypoints"][[pos[int(f)] for f in fi]]  # stride-match 2D to the lift
        rec = ss[vid] if vid in ss else {}
        demos = [
            dm for dm in rec.get("demos", []) if dm.get("two_person_frac", 0) >= MIN_2P
        ]
        pairs = [
            _matched_demo(kp2d, d3["world"], fi, fps, dm["start_s"], dm["end_s"])
            for dm in demos
        ]
        pairs = [x for x in pairs if x is not None]
        if len(pairs) >= 3:
            by2.append([a for a, _ in pairs])
            by3.append([b for _, b in pairs])
            n_frames += sum(len(a) for a, _ in pairs)
    dist = lambda a, b: compare(a, b)["normalized"]  # noqa: E731
    r2 = loo_medoid_separability(by2, distance=dist, per_demo=True)
    r3 = loo_medoid_separability(by3, distance=dist, per_demo=True)
    boot = paired_bootstrap_separability(r2.pop("per_demo"), r3.pop("per_demo"))
    return {
        "mode": f"matched_angles_{lift}",
        "n_techniques": len(by2),
        "n_demos": sum(len(t) for t in by2),
        "n_frames": n_frames,
        "2d": r2,
        "3d": r3,
        # a = 2D, b = 3D; *_diff = 2D − 3D (CI straddling 0 = not distinguishable)
        "bootstrap_95ci": boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feat",
        default="angles",
        choices=["angles", "pos", "both"],
        help="angle (rotation-invariant), position (canon matters), or both",
    )
    ap.add_argument(
        "--no-canon",
        action="store_true",
        help="skip pelvis-frame canonicalization (old, invalid viewpoint test)",
    )
    ap.add_argument("--lift", default="bbox", choices=list(LIFT_SUFFIX),
                    help="raw bbox-crop lifts, or letterboxed square-crop lifts")
    ap.add_argument("--matched-2d", action="store_true",
                    help="fair 2D-vs-3D angle comparison on identical frames/demos/person")
    args = ap.parse_args()
    if args.matched_2d:
        print(json.dumps(matched_2d_vs_3d(args.lift)))
        return
    canon = not args.no_canon

    ss = store.segments_store()
    feats_by_tech = []
    for npz in sorted(BRIDGE.glob("*" + LIFT_SUFFIX[args.lift])):
        vid = npz.name[: -len(LIFT_SUFFIX[args.lift])]
        d = np.load(npz, allow_pickle=False)
        world, fi, fps = d["world"], d["frame_indices"], float(d["fps"])
        rec = ss[vid] if vid in ss else {}
        demos = [
            dm for dm in rec.get("demos", []) if dm.get("two_person_frac", 0) >= MIN_2P
        ]
        feats = [
            _demo_feats(
                world, fi, fps, dm["start_s"], dm["end_s"], feat=args.feat, canon=canon
            )
            for dm in demos
        ]
        feats = [f for f in feats if len(f) >= 4]
        if len(feats) >= 3:
            feats_by_tech.append(feats)

    rep = loo_medoid_separability(
        feats_by_tech, distance=lambda a, b: compare(a, b)["normalized"]
    )
    print(
        json.dumps(
            {
                "mode": f"3d_mediapipe_{args.feat}_{args.lift}" + ("_canon" if canon else "_raw"),
                **rep,
            }
        )
    )


if __name__ == "__main__":
    main()
