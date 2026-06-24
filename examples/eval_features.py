"""Measure how separable techniques are under a given feature descriptor.

Builds a medoid reference per technique using ``--mode`` descriptors, then reports
technique-ID accuracy (each technique's demos -> nearest reference), genuine vs
impostor distance medians, and a separation AUC = P(impostor distance > genuine
distance). Emits one JSON line (for the feature-comparison workflow).

Usage::
    PYTHONPATH=<repo> python examples/eval_features.py --mode angles_both
"""

import argparse
import json

import numpy as np

from kodokan import store
from kodokan.compare import compare
from kodokan.descriptors import demo_descriptor
from kodokan.score import build_reference

MIN_2P = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="angles")
    ap.add_argument("--max-tech", type=int, default=0, help="cap #techniques (0 = all) for quick tests")
    args = ap.parse_args()

    ps, ss = store.pose_store(), store.segments_store()
    techs = []
    for vid in sorted(ps):
        seq = ps[vid]
        rec = ss[vid] if vid in ss else {}
        demos = [d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P]
        feats = [demo_descriptor(seq, d["start_s"], d["end_s"], mode=args.mode) for d in demos]
        feats = [f for f in feats if len(f) >= 4]
        if len(feats) >= 3:
            techs.append({"vid": vid, "feats": feats, "ref": build_reference(feats)})
        if args.max_tech and len(techs) >= args.max_tech:
            break

    n = len(techs)
    M = np.zeros((n, n))
    gen, imp = [], []
    for i, ti in enumerate(techs):
        for j, tj in enumerate(techs):
            ds = [compare(f, tj["ref"]["reference"])["normalized"] for f in ti["feats"]]
            M[i, j] = float(np.mean(ds))
            (gen if i == j else imp).extend(ds)
    acc = float(np.mean([int(np.argmin(M[i]) == i) for i in range(n)])) if n else 0.0
    gen = np.array([g for g in gen if g > 1e-9])  # drop medoid-to-self zeros
    imp = np.array(imp)
    auc = float(np.mean(imp[:, None] > gen[None, :])) if len(gen) and len(imp) else float("nan")

    print(json.dumps({
        "mode": args.mode,
        "n_tech": n,
        "feat_dim": int(techs[0]["feats"][0].shape[1]) if techs else 0,
        "accuracy": round(acc, 3),
        "genuine_median": round(float(np.median(gen)), 4) if len(gen) else None,
        "impostor_median": round(float(np.median(imp)), 4) if len(imp) else None,
        "auc": round(auc, 3),
    }))


if __name__ == "__main__":
    main()
