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
    demo_feature,
    dtw_1nn_accuracy,
    pool_centroid_accuracy,
    pool_knn_accuracy,
    pool_lda_knn_accuracy,
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
        seq = ps[vid]
        rec = ss[vid] if vid in ss else {}
        demos = [d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P]
        fs = [demo_feature(seq, d["start_s"], d["end_s"], mode=args.feature) for d in demos]
        fs = [f for f in fs if len(f) >= 4]
        if len(fs) >= 3:
            feats.extend(fs)
            labels.extend([n_classes] * len(fs))
            n_classes += 1

    y = np.array(labels)
    n = len(feats)
    if args.method == "dtw_1nn":
        acc = dtw_1nn_accuracy(feats, labels)
    else:
        X = np.stack([pooled_descriptor(f) for f in feats])
        acc = {
            "pool_centroid": pool_centroid_accuracy,
            "pool_knn": pool_knn_accuracy,
            "pool_lda_knn": pool_lda_knn_accuracy,
        }[args.method](X, y)

    print(json.dumps({
        "feature": args.feature,
        "method": args.method,
        "n_samples": n,
        "n_classes": n_classes,
        "accuracy": round(acc, 3) if acc is not None else None,
        "chance": round(1.0 / n_classes, 3) if n_classes else None,
    }))


if __name__ == "__main__":
    main()
