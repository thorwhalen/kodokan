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
- Canonicalization does **not** undo the *anisotropic* bbox-crop distortion the
  review named (a non-square crop stretches the world landmarks); that persists
  after canon. A complete viewpoint re-test still needs the two **deferred**
  pieces from ``misc/docs/adversarial-review.md``: square/letterboxed re-lifting,
  and a stride- & person-matched 2D-vs-3D comparison (here the 3D person is picked
  by presence-argmax while the 2D baseline uses ``primary_person``).

Usage::
    PYTHONPATH=<repo> python examples/eval_features_3d.py --feat pos [--no-canon]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kodokan import store
from kodokan.canon3d import BLAZEPOSE33, canonicalize_pose
from kodokan.compare import compare
from kodokan.config import EVAL_MIN_TWO_PERSON_FRAC as MIN_2P
from kodokan.recognize import loo_medoid_separability

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
    args = ap.parse_args()
    canon = not args.no_canon

    ss = store.segments_store()
    feats_by_tech = []
    for npz in sorted(BRIDGE.glob("*_3d.npz")):
        vid = npz.name[: -len("_3d.npz")]
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
                "mode": f"3d_mediapipe_{args.feat}" + ("_canon" if canon else "_raw"),
                **rep,
            }
        )
    )


if __name__ == "__main__":
    main()
