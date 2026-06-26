"""Technique-recognition learning game (app #6) — logic layer.

Help students learn to recognize/name throws. No throw-detection is needed (ground
truth is known): we generate multiple-choice problems from the clip library —
**name -> pick the video** or **video -> pick the name** — with *confusable*
distractors, log every response (timestamped, with the exact problem) for adaptive
learning, and score with confusion-aware weighting.

"Confusability" between techniques is a **pose-shape similarity** (closeness of their
mean pooled descriptors) — an honest proxy for visual confusability that does NOT
require a working recognizer (the cross-source recognizer is currently at chance). It
can be swapped for the model's confusion matrix once recognition generalizes.

This module is the (UI-agnostic) logic + storage; a web UI plugs into it later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PathLike = str | Path
MODES = ("name_to_video", "video_to_name")

# Playlist section-header entries (e.g. "足技 / Ashi-waza") are category names, not
# throws; their canonical keys are excluded from the quiz catalog.
DIVIDER_KEYS = frozenset(
    {
        "tewaza",
        "ashiwaza",
        "koshiwaza",
        "sutemiwaza",
        "masutemiwaza",
        "yokosutemiwaza",
        "nagewaza",
        "katamewaza",
        "osaekomiwaza",
        "shimewaza",
        "kansetsuwaza",
    }
)


@dataclass
class Problem:
    """One multiple-choice flashcard problem."""

    problem_id: str
    mode: str
    target_key: str
    prompt: dict  # {"name": str} or {"clip": {...}}
    choices: list[dict]  # each: {"key", "name", optionally "clip"}
    correct_index: int

    def is_correct(self, chosen_index: int) -> bool:
        return chosen_index == self.correct_index


# --------------------------------------------------------------------------- #
# Distractor / target selection (pure, seedable)
# --------------------------------------------------------------------------- #


def _weighted_sample(
    items: list, weights: list[float], k: int, rng: random.Random
) -> list:
    """Sample k distinct items proportional to weights (no replacement)."""
    items, weights = list(items), [max(w, 1e-9) for w in weights]
    chosen = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                chosen.append(items.pop(i))
                weights.pop(i)
                break
    return chosen


def confusable_distractors(target_key, keys, *, n, similarity=None, rng=None) -> list:
    """Pick ``n`` distractor keys, preferring those most confusable with the target."""
    rng = rng or random.Random()
    pool = [k for k in keys if k != target_key]
    if similarity and target_key in similarity:
        weights = [similarity[target_key].get(k, 0.0) for k in pool]
        if not any(weights):
            weights = [1.0] * len(pool)
    else:
        weights = [1.0] * len(pool)
    return _weighted_sample(pool, weights, n, rng)


def make_problem(
    catalog: dict,
    target_key: str,
    *,
    mode: str = "video_to_name",
    n_choices: int = 4,
    similarity=None,
    rng: random.Random | None = None,
    clip_for=None,
) -> Problem:
    """Build a multiple-choice problem for ``target_key`` from a technique ``catalog``.

    ``catalog`` maps ``technique_key -> {"name": str, "clips": [clip dicts]}``.
    ``clip_for(key)`` returns a looping clip for a key (default: first clip); for
    ``name_to_video`` the choices are clips, for ``video_to_name`` they are names.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = rng or random.Random()
    clip_for = clip_for or (lambda k: (catalog[k].get("clips") or [None])[0])

    distractors = confusable_distractors(
        target_key, list(catalog), n=n_choices - 1, similarity=similarity, rng=rng
    )
    option_keys = distractors + [target_key]
    rng.shuffle(option_keys)
    correct_index = option_keys.index(target_key)

    if mode == "video_to_name":
        prompt = {"clip": clip_for(target_key)}
        choices = [{"key": k, "name": catalog[k]["name"]} for k in option_keys]
    else:  # name_to_video
        prompt = {"name": catalog[target_key]["name"]}
        choices = [
            {"key": k, "name": catalog[k]["name"], "clip": clip_for(k)}
            for k in option_keys
        ]

    return Problem(
        problem_id=f"{mode}-{target_key}-{rng.randrange(10**9)}",
        mode=mode,
        target_key=target_key,
        prompt=prompt,
        choices=choices,
        correct_index=correct_index,
    )


# --------------------------------------------------------------------------- #
# Confusion-aware scoring + adaptive selection
# --------------------------------------------------------------------------- #


def score_response(
    correct_key: str, chosen_key: str, *, similarity=None, max_partial: float = 0.5
) -> float:
    """Confusion-weighted credit in [0, 1].

    Correct = 1.0. A wrong answer earns *partial* credit proportional to how confusable
    the chosen technique is with the correct one (an honest mistake between look-alike
    throws is penalized less than confusing two obviously different throws).
    """
    if chosen_key == correct_key:
        return 1.0
    sim = (similarity or {}).get(correct_key, {}).get(chosen_key, 0.0)
    return round(min(max_partial, max(0.0, float(sim)) * max_partial), 3)


def next_target(keys, history, *, focus=None, rng: random.Random | None = None) -> str:
    """Pick the next technique to quiz: unseen and often-missed (within ``focus``) first.

    Spaced-repetition-ish weight = (wrongs + 1) / (seen + 1): unseen -> 1, frequently
    wrong -> high, mastered -> low. ``history`` is a list of response dicts.
    """
    rng = rng or random.Random()
    cand = [k for k in keys if (not focus or k in focus)]
    if not cand:
        raise ValueError("no candidate techniques (empty focus set?)")
    weights = []
    for k in cand:
        seen = [h for h in history if h.get("target_key") == k]
        wrong = sum(1 for h in seen if not h.get("correct"))
        weights.append((wrong + 1) / (len(seen) + 1))
    return _weighted_sample(cand, weights, 1, rng)[0]


# --------------------------------------------------------------------------- #
# Response logging (dol store) + store-backed catalog/confusability
# --------------------------------------------------------------------------- #


def responses_store(directory: PathLike | None = None):
    """A dict-like JSON store of logged responses (per-user, timestamped)."""
    from dol import JsonFiles

    from kodokan.config import data_dir

    directory = str(directory or (data_dir() / "responses"))
    Path(directory).mkdir(parents=True, exist_ok=True)
    return JsonFiles(directory)


def log_response(
    problem: Problem,
    chosen_index: int,
    *,
    user: str = "default",
    store=None,
    similarity=None,
    timestamp: str | None = None,
) -> dict:
    """Record a response (datetime + exact problem + choice + correctness + score)."""
    store = store if store is not None else responses_store()
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    chosen_key = problem.choices[chosen_index]["key"]
    rec = {
        "ts": ts,
        "user": user,
        "problem_id": problem.problem_id,
        "mode": problem.mode,
        "target_key": problem.target_key,
        "choice_keys": [c["key"] for c in problem.choices],
        "chosen_key": chosen_key,
        "correct": bool(problem.is_correct(chosen_index)),
        "score": score_response(problem.target_key, chosen_key, similarity=similarity),
    }
    store[f"{user}__{ts}.json".replace(":", "-")] = (
        rec  # filesystem-safe (no '/' subdir, no ':')
    )
    return rec


def build_catalog(*, min_two_person_frac: float = 0.4, min_demo_s: float = 1.0):
    """Build ``{technique_key: {"name", "clips":[...]}}`` from the pose/segments stores.

    Aggregates all sources: each clip contributes its demo intervals (for looping media),
    so a technique present in several sources offers varied multiple-choice media.
    """
    from kodokan import store
    from kodokan.acquire import canonical_technique_key

    ss = store.segments_store()
    catalog: dict[str, dict] = {}
    for vid in ss:
        rec = ss[vid]
        # recompute the key from the title so the canonical rules (e.g. "Escapes"
        # merging) apply even to segments tagged before those rules existed.
        key = canonical_technique_key(rec.get("technique") or vid)
        if key in DIVIDER_KEYS:  # skip playlist section-header pseudo-entries
            continue
        # clean display name: romaji part, drop "- Demo" / "Escapes" production suffixes
        name = rec.get("technique", vid).split("/")[-1].strip()
        name = re.sub(r"\s*-\s*demo\b", "", name, flags=re.I)
        name = re.sub(r"\s+escapes?\b", "", name, flags=re.I).strip()
        demos = [
            d
            for d in rec.get("demos", [])
            if d.get("two_person_frac", 1.0) >= min_two_person_frac
            and (d["end_s"] - d["start_s"]) >= min_demo_s
        ]
        if not demos:
            continue
        entry = catalog.setdefault(key, {"name": name, "clips": []})
        entry["clips"].append(
            {
                "video_id": vid,
                "source": rec.get("source", "unknown"),
                "source_url": rec.get("source_url"),
                "demos": [
                    {"start_s": d["start_s"], "end_s": d["end_s"]} for d in demos[:3]
                ],
            }
        )
    return catalog


def build_confusability(
    catalog: dict | None = None, *, feature: str = "tori_angles_pos"
) -> dict:
    """Technique×technique similarity (0..1) from mean pooled pose descriptors.

    ``similarity[a][b]`` high ⇒ techniques a, b look alike in pose-space (likely
    confusable for a learner). An honest, recognizer-free proxy; swap for the model's
    confusion matrix once cross-source recognition works.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    from kodokan import store
    from kodokan.recognize import demo_feature, pooled_descriptor

    catalog = catalog or build_catalog()
    ps = store.pose_store()
    vecs = {}
    for key, entry in catalog.items():
        descs = []
        for clip in entry["clips"]:
            if clip["video_id"] not in ps:
                continue
            seq = ps[clip["video_id"]]
            for d in clip["demos"]:
                f = demo_feature(seq, d["start_s"], d["end_s"], mode=feature)
                if len(f) >= 4:
                    descs.append(pooled_descriptor(f))
        if descs:
            vecs[key] = np.mean(np.stack(descs), axis=0)
    keys = list(vecs)
    if len(keys) < 2:
        return {}
    M = np.stack([vecs[k] for k in keys])
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    D = cdist(M, M)
    S = 1.0 - D / (D.max() + 1e-9)
    return {
        keys[i]: {keys[j]: float(round(S[i, j], 3)) for j in range(len(keys)) if j != i}
        for i in range(len(keys))
    }
