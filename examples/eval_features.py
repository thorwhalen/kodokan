"""Measure how separable techniques are under a given feature descriptor.

For each technique's demos (``--mode`` descriptors) it runs the leak-free
**leave-one-demo-out medoid** separability harness: a held-out demo's genuine
distance is to the medoid of its technique's *other* demos, impostor distances to
every other technique's medoid. Reports LOO nearest-medoid accuracy, genuine vs
impostor distance medians, and a standard ROC-AUC. Emits one JSON line.

Usage::
    PYTHONPATH=<repo> python examples/eval_features.py --mode angles_both
"""

import argparse
import json

from kodokan import store
from kodokan.compare import compare
from kodokan.config import EVAL_MIN_TWO_PERSON_FRAC as MIN_2P
from kodokan.descriptors import demo_descriptor
from kodokan.recognize import loo_medoid_separability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="angles")
    ap.add_argument(
        "--max-tech",
        type=int,
        default=0,
        help="cap #techniques (0 = all) for quick tests",
    )
    args = ap.parse_args()

    ps, ss = store.pose_store(), store.segments_store()
    feats_by_tech, feat_dim = [], 0
    for vid in sorted(ps):
        seq = ps[vid]
        rec = ss[vid] if vid in ss else {}
        demos = [
            d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P
        ]
        feats = [
            demo_descriptor(seq, d["start_s"], d["end_s"], mode=args.mode)
            for d in demos
        ]
        feats = [f for f in feats if len(f) >= 4]
        if len(feats) >= 3:
            feats_by_tech.append(feats)
            feat_dim = int(feats[0].shape[1])
        if args.max_tech and len(feats_by_tech) >= args.max_tech:
            break

    rep = loo_medoid_separability(
        feats_by_tech, distance=lambda a, b: compare(a, b)["normalized"]
    )
    print(json.dumps({"mode": args.mode, "feat_dim": feat_dim, **rep}))


if __name__ == "__main__":
    main()
