"""Hand-labelling of the tori heuristic (#49): sample demos, then score agreement.

``recognize.tori_index`` decides which tracked judoka is tori (the thrower) from hip
heights alone. There is no ground truth to validate it against, so the owner labels a
stratified sample of demos by eye (in the web app's owner-only labelling page) and this
module closes the loop:

* :func:`build_label_manifest` -- pick ~60 demos (stratified over throw, source and the
  confident/abstain split) and emit, per demo, the two tracked people's boxes over time,
  normalised to the frame so the page can outline them on the clip in two colours. The
  heuristic's own answer is stored but the app never sends it to the labelling page.
* :func:`agreement_report` -- given the owner's labels, the agreement with the heuristic
  (Wilson CI), split by confident/abstained, plus the "can't tell" rate in each.

Command line::

    python -m kodokan.tori_labels build manifest.json --n 60
    python -m kodokan.tori_labels report manifest.json labels.jsonl

Needs the pose and segment stores (``~/kodokan_data``) for ``build`` only; ``report``
and the pure helpers need nothing but the two files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from kodokan.recognize import ToriDecision, tori_decision

#: The verdicts the labelling page records.
VERDICTS = ("A", "B", "cant_tell")

#: Minimum keypoint confidence for a joint to count towards a person's box.
MIN_KEYPOINT_CONF = 0.1

#: Padding added around the joints' extent, as a fraction of the box size.
BOX_PAD = 0.08


def person_boxes(
    pose_seq,
    start_s: float,
    end_s: float,
    *,
    slots: Sequence[int],
    min_conf: float = MIN_KEYPOINT_CONF,
    pad: float = BOX_PAD,
    max_frames: int = 120,
) -> dict:
    """Normalised boxes of the given person *slots* over a demo window.

    Returns ``{"t": [seconds from window start], "boxes": [[ [x0,y0,x1,y1] | None per slot ] per frame]}``
    with coordinates as fractions of the frame (so they scale to any rendition of the clip).
    Frames are thinned to at most *max_frames*; a slot with no usable joints in a frame is ``None``.
    """
    times = np.asarray(pose_seq.frame_indices, dtype=float) / float(pose_seq.fps)
    idx = np.flatnonzero((times >= start_s) & (times <= end_s))
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    w, h = float(pose_seq.width), float(pose_seq.height)
    frames, ts = [], []
    for f in idx:
        row = []
        for p in slots:
            kp = pose_seq.keypoints[f, p]
            ok = (kp[:, 2] >= min_conf) & np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
            if not ok.any():
                row.append(None)
                continue
            x0, x1 = kp[ok, 0].min(), kp[ok, 0].max()
            y0, y1 = kp[ok, 1].min(), kp[ok, 1].max()
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            row.append(
                [
                    round(float(max(0.0, (x0 - px) / w)), 4),
                    round(float(max(0.0, (y0 - py) / h)), 4),
                    round(float(min(1.0, (x1 + px) / w)), 4),
                    round(float(min(1.0, (y1 + py) / h)), 4),
                ]
            )
        frames.append(row)
        ts.append(round(float(times[f] - start_s), 3))
    return {"t": ts, "boxes": frames}


def labelled_slots(pose_seq, start_s: float, end_s: float) -> list[int]:
    """The two person slots shown to the labeller: the two most present in the window."""
    times = np.asarray(pose_seq.frame_indices, dtype=float) / float(pose_seq.fps)
    win = pose_seq.keypoints[(times >= start_s) & (times <= end_s)]
    if len(win) == 0:
        return []
    present = (~np.all(np.isnan(win[..., 0]), axis=2)).mean(0)  # (P,)
    top = np.argsort(-present)[:2]
    return sorted(int(p) for p in top)


def manifest_entry(
    *,
    throw: str,
    video_id: str,
    source: str,
    pose_seq,
    start_s: float,
    end_s: float,
    clip_file: str,
) -> dict | None:
    """One labelling item for a demo, or ``None`` if it does not have two trackable people."""
    slots = labelled_slots(pose_seq, start_s, end_s)
    if len(slots) < 2:
        return None
    dec: ToriDecision = tori_decision(pose_seq, start_s, end_s)
    return {
        "id": clip_file,
        "clip": clip_file,
        "throw": throw,
        "video_id": video_id,
        "source": source,
        "slots": slots,  # slots[0] is shown as "A", slots[1] as "B"
        "tori_slot": dec.index,  # the heuristic's answer: never sent to the labelling page
        "margin": None if not math.isfinite(dec.margin) else round(dec.margin, 4),
        "abstained": bool(dec.abstained),
        **person_boxes(pose_seq, start_s, end_s, slots=slots),
    }


def stratified_sample(
    entries: Sequence[dict], n: int = 60, *, abstain_share: float = 0.4, seed: int = 0
) -> list[dict]:
    """Pick *n* entries spread over throws and sources, with a guaranteed abstain share.

    The abstained demos are the ones whose role label is in doubt, so they get at least
    ``abstain_share`` of the sample (fewer only if there are not that many). Within each
    half, strata are ``(throw, source)`` cells drawn round-robin from a seeded shuffle, so
    no throw or source dominates.
    """
    rng = random.Random(seed)

    def draw(pool: list[dict], k: int) -> list[dict]:
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for e in pool:
            cells[(e["throw"], e.get("source"))].append(e)
        order = list(cells)
        rng.shuffle(order)
        for c in cells.values():
            rng.shuffle(c)
        out: list[dict] = []
        while len(out) < k and any(cells.values()):
            for key in order:
                if cells[key] and len(out) < k:
                    out.append(cells[key].pop())
        return out

    abst = [e for e in entries if e["abstained"]]
    conf = [e for e in entries if not e["abstained"]]
    k_abst = min(len(abst), round(n * abstain_share))
    picked = draw(abst, k_abst)
    picked += draw(conf, n - len(picked))
    picked += draw([e for e in abst if e not in picked], n - len(picked))  # top up
    rng.shuffle(picked)
    return picked


def build_label_manifest(
    *, n: int = 60, seed: int = 0, abstain_share: float = 0.4
) -> list[dict]:
    """Sample the demos of the local pose/segment stores into a labelling manifest."""
    from kodokan import store
    from kodokan.flashcards import build_catalog, clip_filename

    pose = store.pose_store()
    entries: list[dict] = []
    for key, tech in build_catalog().items():
        for clip in tech["clips"]:
            seq = pose.get(clip["video_id"])
            if seq is None:
                continue
            for d in clip["demos"]:
                e = manifest_entry(
                    throw=key,
                    video_id=clip["video_id"],
                    source=clip.get("source", "unknown"),
                    pose_seq=seq,
                    start_s=d["start_s"],
                    end_s=d["end_s"],
                    clip_file=clip_filename(
                        clip["video_id"], d["start_s"], d["end_s"] - d["start_s"]
                    ),
                )
                if e:
                    entries.append(e)
    return stratified_sample(entries, n, abstain_share=abstain_share, seed=seed)


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(centre - half, 3), round(centre + half, 3))


def latest_labels(rows: Iterable[dict]) -> dict[str, dict]:
    """The most recent label per demo id (the owner may re-label)."""
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r.get("_server_ts") or r.get("ts") or 0):
        if r.get("verdict") in VERDICTS:
            out[r["id"]] = r
    return out


def agreement_report(manifest: Sequence[dict], labels: Iterable[dict]) -> dict:
    """Agreement of ``tori_index`` with the owner's labels.

    A decided label ("A"/"B") agrees when the named slot equals the heuristic's. "Can't tell"
    is excluded from agreement and reported as a rate, overall and per confident/abstained
    group -- if abstentions are the genuinely ambiguous demos, their can't-tell rate is higher.
    """
    by_id = {e["id"]: e for e in manifest}
    lab = {i: r for i, r in latest_labels(labels).items() if i in by_id}
    groups: dict[str, dict] = {
        g: {"labelled": 0, "decided": 0, "agree": 0, "cant_tell": 0}
        for g in ("confident", "abstained")
    }
    for i, r in lab.items():
        e = by_id[i]
        g = groups["abstained" if e["abstained"] else "confident"]
        g["labelled"] += 1
        if r["verdict"] == "cant_tell":
            g["cant_tell"] += 1
            continue
        g["decided"] += 1
        g["agree"] += int(e["slots"][0 if r["verdict"] == "A" else 1] == e["tori_slot"])
    total = {
        k: sum(g[k] for g in groups.values())
        for k in ("labelled", "decided", "agree", "cant_tell")
    }
    for g in (*groups.values(), total):
        g["agreement"] = round(g["agree"] / g["decided"], 3) if g["decided"] else None
        g["ci95"] = wilson_interval(g["agree"], g["decided"])
        g["cant_tell_rate"] = (
            round(g["cant_tell"] / g["labelled"], 3) if g["labelled"] else None
        )
    return {
        "sampled": len(manifest),
        "unlabelled": len(manifest) - total["labelled"],
        "overall": total,
        **groups,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m kodokan.tori_labels", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build", help="sample demos into a labelling manifest (needs ~/kodokan_data)"
    )
    b.add_argument("out", type=Path)
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--abstain-share", type=float, default=0.4)
    r = sub.add_parser("report", help="agreement of tori_index with the owner's labels")
    r.add_argument("manifest", type=Path)
    r.add_argument("labels", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build_label_manifest(n=a.n, seed=a.seed, abstain_share=a.abstain_share)
        a.out.write_text(json.dumps(m))
        print(f"wrote {len(m)} items to {a.out}")
    else:
        print(
            json.dumps(
                agreement_report(
                    json.loads(a.manifest.read_text()), _read_jsonl(a.labels)
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
