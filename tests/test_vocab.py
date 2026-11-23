"""Tests for word parts of technique names (kodokan#65)."""

import importlib.util
import json
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


def test_a_spelling_shared_by_two_parts_is_an_error():
    with pytest.raises(ValueError, match="koshi"):
        kv.variant_index({"goshi": {"alts": ["koshi"]}, "koshi": {}})


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
_MD = Path(__file__).parents[1] / "misc" / "docs" / "judo_throws_classification.md"


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


def test_homographs_found_from_kanji_not_a_hand_list():
    glossary = {"kata": {"kanji": "肩"}, "guruma": {"kanji": "車"}, "shiho": {"kanji": "四方", "alts": ["siho"]}}
    names = {"kataguruma": "Kata-guruma", "katahajime": "Kata-ha-jime", "kamishihogatame": "Kami-siho-gatame"}
    kanji = {"kataguruma": "肩車", "katahajime": "片羽絞"}  # no kanji given for the third: not judged
    found = kv.homographs_by_kanji(names, kanji, glossary)
    assert found == {"kata": frozenset({"katahajime"})}
    wt = kv.word_throws(names, glossary, homographs=found)
    assert wt["kata"] == ["kataguruma"]
    assert wt["shiho"] == ["kamishihogatame"]  # the spelling siho is the same part


@pytest.mark.skipif(not (_BUILDER.exists() and _MD.exists()), reason="builder or reference absent")
def test_katame_words_are_in_the_glossary_and_claimed():
    spec = importlib.util.spec_from_file_location("builder", _BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    text = _MD.read_text()
    glossary = builder.parse_glossary(text)
    assert {"gatame", "jime", "shiho", "kesa", "juji"} <= set(glossary)
    assert "siho" in glossary["shiho"]["alts"]
    assert builder.parse_technique_kanji(text)["katajujijime"] == "片十字絞"


@pytest.mark.skipif(not (_BUILDER.exists() and _MD.exists()), reason="builder or reference absent")
def test_the_builder_writes_the_threshold_into_vocab_json(tmp_path, monkeypatch):
    """The app reads its word-part threshold from vocab.json, so it has one owner (#67)."""
    spec = importlib.util.spec_from_file_location("builder", _BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if not builder.CATALOG.exists():
        pytest.skip("the web app's catalog is not on this machine")
    builder.main(tmp_path)
    doc = json.loads((tmp_path / "vocab.json").read_text())
    assert doc["min_techniques"] == kv.MIN_TECHNIQUES
    assert doc["words"]
