"""HONEST recognition eval: leave-one-CLIP-out across independent sources.

Unlike eval_learned.py (within-clip leave-one-demo-out, an upper bound), this groups
demos by their **source video** and labels them by a cross-source ``technique_key``,
keeping only techniques present in **>=2 independent sources** (e.g. Kodokan-IJF +
Efficient Judo). Leave-one-clip-out then trains on *other clips* (largely other
sources) and tests on a held-out clip — so it cannot win by recognizing the video.

Usage::
    PYTHONPATH=<repo> python examples/eval_cross_source.py --feature tori_angles_pos --method lda_knn
"""

import argparse
import json

import numpy as np

from kodokan import store
from kodokan.acquire import SOURCES, canonical_technique_key, source_clips_dir
from kodokan.recognize import classification_metrics, demo_feature, loo_pooled_predict, pooled_descriptor

MIN_2P = 0.4


def _source_of(vid: str) -> str | None:
    for src in SOURCES:
        if any(source_clips_dir(src).glob(f"*({vid}).mp4")):
            return src
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="tori_angles_pos")
    ap.add_argument("--method", default="lda_knn", choices=["lda_knn", "knn", "centroid"])
    ap.add_argument("--min-sources", type=int, default=2)
    args = ap.parse_args()

    ps, ss = store.pose_store(), store.segments_store()
    # gather per-demo features with (technique_key, source, video_id)
    rows = []  # (key, source, vid, feat)
    key_sources: dict[str, set] = {}
    for vid in sorted(ps):
        try:
            seq, rec = ps[vid], (ss[vid] if vid in ss else {})
        except Exception:
            continue
        key = rec.get("technique_key") or canonical_technique_key(rec.get("technique", vid))
        src = rec.get("source") or _source_of(vid) or "unknown"
        demos = [d for d in rec.get("demos", []) if d.get("two_person_frac", 0) >= MIN_2P]
        for d in demos:
            f = demo_feature(seq, d["start_s"], d["end_s"], mode=args.feature)
            if len(f) >= 4:
                rows.append((key, src, vid, f))
        if any(len(demo_feature(seq, d["start_s"], d["end_s"], mode=args.feature)) >= 4 for d in demos):
            key_sources.setdefault(key, set()).add(src)

    keep = {k for k, s in key_sources.items() if len(s) >= args.min_sources}
    rows = [r for r in rows if r[0] in keep]
    if not rows:
        print(json.dumps({"error": "no technique has >=min_sources sources yet",
                          "keys_with_2plus_sources": sorted(keep)}))
        return

    keys = sorted({r[0] for r in rows})
    kidx = {k: i for i, k in enumerate(keys)}
    X = np.stack([pooled_descriptor(r[3]) for r in rows])
    y = np.array([kidx[r[0]] for r in rows])
    groups = np.array([r[2] for r in rows])  # video_id == CV group

    preds = loo_pooled_predict(X, y, method=args.method, groups=groups)
    m = classification_metrics(y, preds)
    print(json.dumps({
        "eval": "leave-one-CLIP-out (cross-source, honest)",
        "feature": args.feature,
        "method": args.method,
        "n_techniques": len(keys),
        "n_clips": int(len(set(groups))),
        "n_demos": m["n"],
        "sources": sorted({r[1] for r in rows}),
        "top1": m["top1"],
        "balanced_accuracy": m["balanced"],
        "majority_baseline": m["majority_baseline"],
        "chance_uniform": round(1.0 / len(keys), 3),
    }))


if __name__ == "__main__":
    main()
