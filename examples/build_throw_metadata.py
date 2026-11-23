"""Build enriched throw + vocabulary metadata for the kodokan web app (epic #11, issue #14).

Parses the owner's classification reference (misc/docs/judo_throws_classification.md):
  - a Japanese word glossary (romaji/hiragana/kanji + EN/FR + throws each word appears in)
  - a 68-throw classification table (JP/EN/FR names + Te/Koshi/Ashi/Ma-/Yoko-sutemi category)

and merges it with the clip catalog (data/catalog.json) to emit two SSOT data files the
new app consumes:
  - vocab.json   : {wordKey: {romaji, alts, hiragana, kanji, en, fr, throws:[throwKey]}}
  - throws.json  : {throwKey: {slug, romaji, jp:{kanji,hiragana}, en, fr, category,
                               words:[wordKey], clips:[...], confusable:[...], hasClips}}

Which words a throw uses is derived by tokenizing its name against the glossary
(``kodokan.vocab.word_throws``: primary + declared alternate readings, so koshi/goshi are one
word), not taken from the glossary's hand-kept throw lists, which had missed some (#65).
Per-throw kanji/hiragana are composed from the throw's component words; hiragana uses each
word's primary reading (rendaku is approximate).

Usage::  PYTHONPATH=<repo> python examples/build_throw_metadata.py [OUT_DIR]
"""

import json
import os
import re
import sys
from pathlib import Path

from kodokan import vocab as kv
from kodokan.acquire import canonical_technique_key

_DOCS = Path(__file__).resolve().parents[1] / "misc" / "docs"
MD = _DOCS / "judo_throws_classification.md"
# Curated 'commonly-confused throws' map (slug -> [slug]), from authoritative judo pedagogy
# (Kodokan-official series + corroborating sources) — replaces the weak pose-similarity proxy (#33).
CONFUSABLE = _DOCS / "confusable_curated.json"
# The web app's data folder (catalog in, vocab.json and throws.json out): $KODOKAN_APP_DATA,
# else its place in the projects folder ($PP).
OUT_DIR = Path(
    os.environ.get("KODOKAN_APP_DATA")
    or Path(os.environ.get("PP", "~")).expanduser() / "tt" / "papp" / "migrated_apps" / "kodokan" / "data"
)
CATALOG = OUT_DIR / "catalog.json"


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


def parse_technique_kanji(md: str) -> dict:
    """{throwKey: kanji} from the two-cell ``| **Technique** | kanji |`` rows."""
    out = {}
    for line in md.splitlines():
        if not line.startswith("| **"):
            continue
        c = _cells(line)
        name = re.fullmatch(r"\*\*(.+?)\*\*", c[0]) if len(c) == 2 else None
        if name:
            out[canonical_technique_key(name.group(1).strip())] = c[1]
    return out


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


def main(out_dir: Path = OUT_DIR):
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

    # throws.json covers the union of the 68 official + our clip catalog
    all_keys = set(classification) | set(catalog)
    names = {
        k: classification.get(k, {}).get("romaji") or catalog.get(k, {}).get("name") or k
        for k in all_keys
    }

    # vocab.json: word -> translations + the throws whose NAME contains it (#65)
    # kanji of a technique: its words' kanji joined, unless the glossary file states it
    technique_kanji = {k: compose_jp(n, tok2word)[0] for k, n in names.items()}
    technique_kanji.update(parse_technique_kanji(md))
    homographs = kv.homographs_by_kanji(names, technique_kanji, glossary)
    word_throws = kv.word_throws(names, glossary, homographs=homographs)
    vocab = {}
    for key, w in glossary.items():
        vocab[key] = {
            "romaji": w["romaji"], "alts": w["alts"],
            "hiragana": w["hiragana"], "kanji": w["kanji"],
            "en": w["en"], "fr": w["fr"],
            "throws": word_throws[key],
        }

    # which words each throw uses (the reverse)
    throw_words: dict[str, list[str]] = {}
    for key, techs in word_throws.items():
        for t in techs:
            throw_words.setdefault(t, []).append(key)

    throws = {}
    for key in sorted(all_keys):
        cl = classification.get(key, {})
        cat = catalog.get(key, {})
        romaji = names[key]
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

    out_dir.mkdir(parents=True, exist_ok=True)
    # min_techniques: the app offers a word part only when this many techniques use it (#67);
    # written here so the threshold has one owner, kodokan.vocab.MIN_TECHNIQUES.
    vocab_doc = {"words": vocab, "min_techniques": kv.MIN_TECHNIQUES}
    (out_dir / "vocab.json").write_text(json.dumps(vocab_doc, ensure_ascii=False, indent=1))
    (out_dir / "throws.json").write_text(json.dumps({"throws": throws}, ensure_ascii=False, indent=1))

    # report
    official = set(classification)
    have_clips = {k for k, t in throws.items() if t["hasClips"]}
    no_kanji = [k for k, t in throws.items() if t["jp"]["kanji"] is None]
    print(f"vocab words: {len(vocab)}; name tokens no word claims: {kv.unmatched_tokens(names, glossary)}")
    print(f"words shared by >=k throws: {kv.threshold_table(word_throws)}")
    print(f"throws total: {len(throws)} (official nage-waza in file: {len(official)}, with clips: {len(have_clips)})")
    print(f"official without clips (text-only games/info ok): {sorted(official - have_clips)}")
    print(f"our catalog throws NOT in official file (katame etc.): {sorted(set(catalog) - official)}")
    print(f"throws with no composed kanji (unmatched tokens): {no_kanji}")
    n_conf = sum(1 for t in throws.values() if t["confusable"])
    print(f"throws with curated confusables: {n_conf}; unresolved curated slugs: {sorted(unresolved)}")
    print(f"sample: osotogari -> {json.dumps(throws.get('osotogari', {}).get('jp'), ensure_ascii=False)} {throws.get('osotogari',{}).get('en')} / {throws.get('osotogari',{}).get('fr')}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR)
