"""Flashcard learning game (app #6) — end-to-end logic demo (no UI).

Builds the technique catalog + pose-shape confusability from the clip library,
generates a few multiple-choice problems with confusable distractors, simulates
answers, logs them (timestamped) and shows confusion-weighted scoring + adaptive
next-technique selection.

Usage::
    PYTHONPATH=<repo> python examples/flashcard_demo.py
"""

import random

from kodokan import flashcards as fc


class _MemStore(dict):
    """In-memory stand-in for the response store (demo only; real one is dol-backed)."""


def main():
    cat = fc.build_catalog()
    multi = [k for k, v in cat.items() if len({c["source"] for c in v["clips"]}) >= 2]
    print(f"catalog: {len(cat)} techniques, {len(multi)} with >=2 sources (multiple-choice ready)")

    print("computing pose-shape confusability (recognizer-free proxy)...")
    sim = fc.build_confusability(cat)

    rng = random.Random(0)
    history = []
    focus = set(multi[:12])  # a focus subset the learner is studying
    for _ in range(5):
        target = fc.next_target(list(cat), history, focus=focus, rng=rng)
        mode = rng.choice(fc.MODES)
        p = fc.make_problem(cat, target, mode=mode, n_choices=4, similarity=sim, rng=rng)
        # simulate a learner who sometimes picks a confusable distractor
        chosen = p.correct_index if rng.random() < 0.6 else rng.randrange(len(p.choices))
        rec = fc.log_response(p, chosen, similarity=sim, store=_MemStore())
        history.append(rec)
        names = [c["name"] for c in p.choices]
        print(f"  [{mode}] target={cat[target]['name']!r} choices={names} "
              f"chosen={p.choices[chosen]['name']!r} correct={rec['correct']} score={rec['score']}")

    acc = sum(h["correct"] for h in history) / len(history)
    weighted = sum(h["score"] for h in history) / len(history)
    print(f"\nsession: raw acc={acc:.0%}  confusion-weighted score={weighted:.2f}")
    # which techniques is the learner weak on (drives future selection)?
    print("most confusable with the first focus technique:",
          sorted(sim.get(multi[0], {}).items(), key=lambda kv: -kv[1])[:3])


class _MemStore(dict):
    """In-memory stand-in for the response store (demo only)."""


if __name__ == "__main__":
    main()
