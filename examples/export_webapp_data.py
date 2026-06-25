"""Export the flashcard catalog to a static JSON for the kodokan web app.

Produces ``catalog.json`` consumed by the apps.thorwhalen.com/kodokan frontend:
per technique a clean name + the source clips (YouTube videoId + a short loop
segment start/end) + the most pose-confusable techniques (for distractors). Media
is played as **looping YouTube IFrame segments** (no redistribution).

Usage::
    PYTHONPATH=<repo> python examples/export_webapp_data.py [OUT.json]
"""

import json
import sys
from pathlib import Path

from kodokan import flashcards as fc

DEFAULT_OUT = Path(
    "/Users/thorwhalen/Dropbox/py/proj/tt/papp/migrated_apps/kodokan/data/catalog.json"
)
LOOP_MAX_S = 7.0   # cap a loop segment length
LOOP_MIN_S = 1.5
N_CONFUSABLE = 8


def _loop_segment(demos):
    """Pick a short, clean loop segment (first demo, capped to LOOP_MAX_S)."""
    for d in demos:
        dur = d["end_s"] - d["start_s"]
        if dur >= LOOP_MIN_S:
            start = float(d["start_s"])
            return {"start": round(start, 2), "end": round(start + min(dur, LOOP_MAX_S), 2)}
    if demos:
        return {"start": round(float(demos[0]["start_s"]), 2),
                "end": round(float(demos[0]["start_s"]) + LOOP_MAX_S, 2)}
    return None


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    catalog = fc.build_catalog()
    print(f"catalog: {len(catalog)} techniques; computing pose-shape confusability...", flush=True)
    sim = fc.build_confusability(catalog)

    techniques = {}
    for key, entry in catalog.items():
        clips = []
        for c in entry["clips"]:
            loop = _loop_segment(c.get("demos", []))
            if loop:
                clips.append({
                    "videoId": c["video_id"],
                    "source": c["source"],
                    "url": c.get("source_url"),
                    **loop,
                })
        if not clips:
            continue
        confusable = [k for k, _ in sorted(sim.get(key, {}).items(), key=lambda kv: -kv[1])[:N_CONFUSABLE]]
        techniques[key] = {
            "name": entry["name"],
            "clips": clips,
            "sources": sorted({c["source"] for c in clips}),
            "confusable": confusable,
        }

    payload = {
        "n_techniques": len(techniques),
        "techniques": techniques,
    }
    out.write_text(json.dumps(payload, indent=1))
    multi = sum(1 for t in techniques.values() if len(t["sources"]) >= 2)
    print(f"wrote {out} — {len(techniques)} techniques ({multi} with >=2 sources), "
          f"{sum(len(t['clips']) for t in techniques.values())} clips", flush=True)


if __name__ == "__main__":
    main()
