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
from kodokan.recognize import (
    classification_metrics,
    demo_feature,
    dtw_1nn_predict,
    loo_pooled_predict,
    pooled_descriptor,
)

MIN_2P = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="primary_angles",
                    choices=["primary_angles", "tori_angles", "tori_angles_pos"])
    ap.add_argument("--method", default="dtw_1nn",
                    choices=["dtw_1nn", "pool_centroid", "pool_knn", "pool_lda_knn"])
    args = ap.parse_args()

    ps, ss = store.pose_store(), store.segments_store()
    feats, labels = [], []
    n_classes = 0
    for vid in sorted(ps):
        try:  # robust to a clip being written by a concurrent batch run
            seq = ps[vid]
            rec = ss[vid] if vid in ss else {}
        except Exception:
            continue
        demos = [d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P]
        fs = [demo_feature(seq, d["start_s"], d["end_s"], mode=args.feature) for d in demos]
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

    m = classification_metrics(y, preds)
    print(json.dumps({
        "feature": args.feature,
        "method": args.method,
        "n_samples": m["n"],
        "n_classes": m["n_classes"],
        "accuracy": m["top1"],          # leakage-free LOO top-1 (within-clip: upper bound)
        "balanced_accuracy": m["balanced"],
        "majority_baseline": m["majority_baseline"],
        "chance_uniform": round(1.0 / m["n_classes"], 3) if m["n_classes"] else None,
        "caveat": "within-clip LOO (class==1 video); upper bound, not clip-independent",
    }))


if __name__ == "__main__":
    main()
