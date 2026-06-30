"""Build enriched throw + vocabulary metadata for the kodokan web app (epic #11, issue #14).

Parses the owner's classification reference (misc/docs/judo_throws_classification.md):
  - a Japanese word glossary (romaji/hiragana/kanji + EN/FR + throws each word appears in)
  - a 68-throw classification table (JP/EN/FR names + Te/Koshi/Ashi/Ma-/Yoko-sutemi category)

and merges it with the clip catalog (data/catalog.json) to emit two SSOT data files the
new app consumes:
  - vocab.json   : {wordKey: {romaji, hiragana, kanji, en, fr, throws:[throwKey]}}
  - throws.json  : {throwKey: {slug, romaji, jp:{kanji,hiragana}, en, fr, category,
                               words:[wordKey], clips:[...], confusable:[...], hasClips}}

Per-throw kanji/hiragana are composed from the throw's component words (the glossary's
reverse mapping); hiragana uses each word's primary reading (rendaku is approximate).

Usage::  PYTHONPATH=<repo> python examples/build_throw_metadata.py
"""

import json
import re
from pathlib import Path

from kodokan.acquire import canonical_technique_key

MD = Path("/Users/thorwhalen/Dropbox/py/proj/t/kodokan/misc/docs/judo_throws_classification.md")
CATALOG = Path("/Users/thorwhalen/Dropbox/py/proj/tt/papp/migrated_apps/kodokan/data/catalog.json")
# Curated 'commonly-confused throws' map (slug -> [slug]), from authoritative judo pedagogy
# (Kodokan-official series + corroborating sources) — replaces the weak pose-similarity proxy (#33).
CONFUSABLE = Path("/Users/thorwhalen/Dropbox/py/proj/t/kodokan/misc/docs/confusable_curated.json")
OUT_DIR = Path("/Users/thorwhalen/Dropbox/py/proj/tt/papp/migrated_apps/kodokan/data")


def _slug(romaji: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", romaji.lower()).strip("-")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_glossary(md: str) -> dict:
    """{wordKey: {romaji, alts, hiragana, kanji, en, fr, throws:[raw names]}}."""
    words = {}
    for line in md.splitlines():
        if not line.startswith("| **"):
            continue
        c = _cells(line)
        if len(c) != 5 or "(" not in c[0]:  # glossary rows have 5 cells + a "(kanji)"
            continue
        wordcell, en, fr, _count, throws = c
        m = re.search(r"\*\*(.+?)\*\*\s*(?:\(([^)]+)\))?", wordcell)
        if not m:
            continue
        romaji = m.group(1).strip()
        alts = [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
        jp = wordcell.split("<br>", 1)[1] if "<br>" in wordcell else ""
        hira = jp.split("(")[0].strip()
        kanji = (re.search(r"\(([^)]+)\)", jp) or [None, ""])[1]
        words[romaji.lower()] = {
            "romaji": romaji, "alts": alts,
            "hiragana": hira.split("/")[0].strip(),  # primary reading
            "kanji": kanji,
            "en": en, "fr": fr,
            "throws": [t.strip() for t in throws.split(",") if t.strip()],
        }
    return words


def parse_classification(md: str) -> dict:
    """{throwKey: {romaji, en, fr, category}} from the per-category tables."""
    throws, category = {}, None
    for line in md.splitlines():
        h = re.match(r"^####\s+([A-Za-z-]+?)-waza", line)
        if h:
            category = h.group(1) + "-waza"
            continue
        if not line.startswith("| **") or category is None:
            continue
        c = _cells(line)
        if len(c) != 4:
            continue
        romaji = re.search(r"\*\*(.+?)\*\*", c[0])
        if not romaji:
            continue
        romaji = romaji.group(1).strip()
        throws[canonical_technique_key(romaji)] = {
            "romaji": romaji, "en": c[1], "fr": c[2], "category": category,
        }
    return throws


def compose_jp(romaji: str, tok2word: dict):
    """Compose (kanji, hiragana) by concatenating the throw's component words in order."""
    kanji, hira, ok = [], [], True
    for tok in romaji.lower().split("-"):
        w = tok2word.get(tok)
        if not w:
            ok = False
            break
        kanji.append(w["kanji"])
        hira.append(w["hiragana"])
    return ("".join(kanji), "".join(hira)) if ok else (None, None)


def main():
    md = MD.read_text()
    glossary = parse_glossary(md)
    classification = parse_classification(md)
    catalog = json.loads(CATALOG.read_text())["techniques"]

    # token -> word lookup (primary + alt readings) for kanji/hiragana composition
    tok2word = {}
    for key, w in glossary.items():
        tok2word[key] = w
        for alt in w["alts"]:
            tok2word[alt.lower()] = w

    # vocab.json: word -> translations + throw keys it appears in
    vocab = {}
    for key, w in glossary.items():
        vocab[key] = {
            "romaji": w["romaji"], "hiragana": w["hiragana"], "kanji": w["kanji"],
            "en": w["en"], "fr": w["fr"],
            "throws": sorted({canonical_technique_key(t) for t in w["throws"]}),
        }

    # which words each throw uses (reverse of the glossary)
    throw_words: dict[str, list[str]] = {}
    for key, w in glossary.items():
        for t in w["throws"]:
            throw_words.setdefault(canonical_technique_key(t), []).append(key)

    # throws.json: union of the 68 official + our clip catalog
    throws = {}
    all_keys = set(classification) | set(catalog)
    for key in sorted(all_keys):
        cl = classification.get(key, {})
        cat = catalog.get(key, {})
        romaji = cl.get("romaji") or cat.get("name") or key
        kanji, hira = compose_jp(romaji, tok2word)
        throws[key] = {
            "slug": _slug(romaji),
            "romaji": romaji,
            "jp": {"kanji": kanji, "hiragana": hira},
            "en": cl.get("en"),
            "fr": cl.get("fr"),
            "category": cl.get("category"),  # None for katame-waza (not in nage-waza ref)
            "words": sorted(set(throw_words.get(key, []))),
            "clips": cat.get("clips", []),
            "confusable": [],  # filled below from the curated slug map
            "hasClips": bool(cat.get("clips")),
        }

    # Curated confusable map (slug -> [slug]); store as catalog KEYS to match the app's lookups.
    curated = json.loads(CONFUSABLE.read_text())["confusable"]
    slug2key = {t["slug"]: k for k, t in throws.items()}
    unresolved = set()
    for key, t in throws.items():
        out = []
        for s in curated.get(t["slug"], []):
            k = slug2key.get(s)
            if k:
                out.append(k)
            else:
                unresolved.add(s)
        t["confusable"] = out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vocab.json").write_text(json.dumps({"words": vocab}, ensure_ascii=False, indent=1))
    (OUT_DIR / "throws.json").write_text(json.dumps({"throws": throws}, ensure_ascii=False, indent=1))

    # report
    official = set(classification)
    have_clips = {k for k, t in throws.items() if t["hasClips"]}
    no_kanji = [k for k, t in throws.items() if t["jp"]["kanji"] is None]
    print(f"vocab words: {len(vocab)}")
    print(f"throws total: {len(throws)} (official nage-waza in file: {len(official)}, with clips: {len(have_clips)})")
    print(f"official without clips (text-only games/info ok): {sorted(official - have_clips)}")
    print(f"our catalog throws NOT in official file (katame etc.): {sorted(set(catalog) - official)}")
    print(f"throws with no composed kanji (unmatched tokens): {no_kanji}")
    n_conf = sum(1 for t in throws.values() if t["confusable"])
    print(f"throws with curated confusables: {n_conf}; unresolved curated slugs: {sorted(unresolved)}")
    print(f"sample: osotogari -> {json.dumps(throws.get('osotogari', {}).get('jp'), ensure_ascii=False)} {throws.get('osotogari',{}).get('en')} / {throws.get('osotogari',{}).get('fr')}")


if __name__ == "__main__":
    main()
