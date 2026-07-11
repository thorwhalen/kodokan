"""Leave-one-out technique recognition: learned classifiers & role-consistent features.

Emits one JSON line with LOO accuracy vs chance for a (feature, method) pair, over
the dataset's two-person demos. Used by the recognition bake-off workflow.

Usage::
    PYTHONPATH=<repo> python examples/eval_learned.py --feature tori_angles --method pool_knn
"""

import argparse
import json

import numpy as np

from kodokan import store
from kodokan.config import EVAL_MIN_TWO_PERSON_FRAC as MIN_2P
from kodokan.recognize import (
    classification_metrics,
    confusion_pairs,
    demo_feature,
    dtw_1nn_predict,
    loo_pooled_predict,
    pooled_descriptor,
    tori_decision,
    tori_decision_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature",
        default="primary_angles",
        choices=["primary_angles", "tori_angles", "tori_angles_pos"],
    )
    ap.add_argument(
        "--method",
        default="dtw_1nn",
        choices=["dtw_1nn", "pool_centroid", "pool_knn", "pool_lda_knn"],
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="bootstrap resamples for accuracy CIs (0 to skip)",
    )
    args = ap.parse_args()

    ps, ss = store.pose_store(), store.segments_store()
    feats, labels = [], []
    tori_decisions = []
    n_classes = 0
    for vid in sorted(ps):
        try:  # robust to a clip being written by a concurrent batch run
            seq = ps[vid]
            rec = ss[vid] if vid in ss else {}
        except Exception:
            continue
        demos = [
            d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P
        ]
        fs = [
            demo_feature(seq, d["start_s"], d["end_s"], mode=args.feature)
            for d in demos
        ]
        # instrument the tori/uke heuristic the "role-consistent" features depend on
        if args.feature.startswith("tori"):
            tori_decisions.extend(
                tori_decision(seq, d["start_s"], d["end_s"]) for d in demos
            )
        fs = [f for f in fs if len(f) >= 4]
        if len(fs) >= 3:
            feats.extend(fs)
            labels.extend([n_classes] * len(fs))
            n_classes += 1

    y = np.array(labels)
    if args.method == "dtw_1nn":
        preds = dtw_1nn_predict(feats, labels)
    else:
        X = np.stack([pooled_descriptor(f) for f in feats])
        preds = loo_pooled_predict(X, y, method=args.method.replace("pool_", ""))

    m = classification_metrics(y, preds, n_boot=args.n_boot)
    out = {
        "feature": args.feature,
        "method": args.method,
        "n_samples": m["n"],
        "n_classes": m["n_classes"],
        "accuracy": m["top1"],  # leakage-free LOO top-1 (within-clip: upper bound)
        "accuracy_ci95": m.get("top1_ci"),
        "balanced_accuracy": m["balanced"],
        "balanced_ci95": m.get("balanced_ci"),
        "majority_baseline": m["majority_baseline"],
        "chance_uniform": round(1.0 / m["n_classes"], 3) if m["n_classes"] else None,
        "top_confusions": confusion_pairs(y, preds, top=5),
        "caveat": "within-clip LOO (class==1 video); upper bound, not clip-independent",
    }
    if tori_decisions:
        out["tori_heuristic"] = tori_decision_stats(tori_decisions)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
