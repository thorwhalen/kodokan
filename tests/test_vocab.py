"""Tests for word parts of technique names (kodokan#65)."""

import importlib.util
from pathlib import Path

import pytest

from kodokan import vocab as kv

GLOSSARY = {
    "goshi": {"alts": ["koshi"]},
    "gari": {"alts": ["kari"]},
    "harai": {"alts": ["barai"]},
    "o": {},
    "soto": {},
    "ashi": {},
    "de": {},
    "guruma": {"alts": ["kuruma"]},
}
NAMES = {
    "ogoshi": "O-goshi",
    "koshiguruma": "Koshi-guruma",
    "haraigoshi": "Harai-goshi",
    "osotogari": "O-soto-gari",
    "deashiharai": "De-ashi-harai",
}


def test_rendaku_variants_are_one_part():
    wt = kv.word_throws(NAMES, GLOSSARY)
    assert wt["goshi"] == ["haraigoshi", "koshiguruma", "ogoshi"]
    assert wt["harai"] == ["deashiharai", "haraigoshi"]
    assert kv.parts_of_name("O-soto-gari", kv.variant_index(GLOSSARY)) == ["o", "soto", "gari"]


def test_variants_are_declared_not_guessed():
    # "te" devoices to itself and "de" is a different word: never merged silently
    assert kv.variant_index(GLOSSARY).get("te") is None
    assert kv.unmatched_tokens({"x": "Te-waza", "y": "Kaeshi-goshi"}, GLOSSARY) == {
        "kaeshi": None,
        "te": "de",  # suggested for a person to check, not merged
        "waza": None,
    }


def test_every_glossary_part_listed_even_unused():
    assert kv.word_throws({}, GLOSSARY)["ashi"] == []


def test_homographs_excluded():
    wt = kv.word_throws(
        {"a": "Kata-guruma", "b": "Kata-juji-jime"}, {"kata": {}}, homographs={"kata": ["b"]}
    )
    assert wt["kata"] == ["a"]


def test_part_stats_and_threshold():
    rows = kv.part_stats({"x": [1, 2, 3], "y": [1, 1, 2], "z": [4]}, min_techniques=3)
    assert [(r["part"], r["n_techniques"], r["offered"]) for r in rows] == [
        ("x", 3, True),
        ("y", 2, False),
        ("z", 1, False),
    ]
    assert kv.threshold_table({"x": [1, 2, 3], "y": [1, 2]}, upto=3) == {1: 2, 2: 2, 3: 1}


_BUILDER = Path(__file__).parents[1] / "examples" / "build_throw_metadata.py"


@pytest.mark.skipif(not _BUILDER.exists(), reason="builder script not present")
def test_real_glossary_threshold_keeps_the_owners_examples():
    """The threshold was chosen so the owner's named parts (ashi, goshi, harai, seoi) are offered."""
    spec = importlib.util.spec_from_file_location("builder", _BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    md = Path(__file__).parents[1] / "misc" / "docs" / "judo_throws_classification.md"
    if not md.exists():
        pytest.skip("classification reference not present")
    text = md.read_text()
    glossary = builder.parse_glossary(text)
    names = {k: t["romaji"] for k, t in builder.parse_classification(text).items()}
    wt = kv.word_throws(names, glossary)
    offered = {r["part"] for r in kv.part_stats(wt) if r["offered"]}
    assert {"ashi", "goshi", "harai", "seoi"} <= offered
    assert "koshiguruma" in wt["goshi"]  # the hand-kept list had missed it
    assert not kv.unmatched_tokens(names, glossary)  # every nage-waza token is a known part
