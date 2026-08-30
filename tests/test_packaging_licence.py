"""Guards on the licence perimeter this package declares in ``pyproject.toml``.

kodokan is MIT and its core dependency list is ``numpy`` alone. Exactly one
optional extra reaches copyleft: ``track`` brings ultralytics (AGPL-3.0-or-later).
That separation is a packaging fact with no runtime expression, so nothing in the
rest of the suite would notice it being undone -- these tests are what notice.
"""

from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    # These invariants are about packaging metadata, not runtime behaviour, so they
    # are platform- and interpreter-independent: the 3.12 and Windows jobs prove
    # them for every job. Skipping here beats making the package carry a parser.
    tomllib = pytest.importorskip(
        "tomli",
        reason="reading pyproject.toml needs tomllib (3.11+) or tomli",
    )

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Distributions that carry copyleft terms and must never reach an extra other than
#: ``track`` / ``all``. ultralytics pulls the other two itself.
COPYLEFT_DISTRIBUTIONS = frozenset(
    {"ultralytics", "ultralytics-thop", "ultralytics-platform"}
)

#: The only extras a user may opt into copyleft through.
COPYLEFT_EXTRAS = frozenset({"track", "all"})


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads(_PYPROJECT.read_text())


def _distribution_names(requirements):
    """Bare distribution names from PEP 508 requirement strings."""
    for requirement in requirements:
        head = requirement.split(";")[0].split("[")[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", " ", "@"):
            head = head.split(separator)[0]
        yield head.strip().lower().replace("_", "-")


def test_core_dependencies_stay_copyleft_free(pyproject):
    # `pip install kodokan` must pull nothing copyleft. This is the claim the
    # README's "Licensing of extras" section opens with.
    core = set(_distribution_names(pyproject["project"]["dependencies"]))
    assert core == {"numpy"}
    assert not core & COPYLEFT_DISTRIBUTIONS


def test_copyleft_only_reachable_through_the_named_extras(pyproject):
    # Putting ultralytics back into `pose` is the regression this exists to catch:
    # it is how a user who wanted RTMPose silently inherited the AGPL.
    for extra, requirements in pyproject["project"]["optional-dependencies"].items():
        found = set(_distribution_names(requirements)) & COPYLEFT_DISTRIBUTIONS
        if extra in COPYLEFT_EXTRAS:
            continue
        assert not found, f"copyleft {sorted(found)} leaked into the `{extra}` extra"


def test_track_is_the_extra_that_declares_ultralytics(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert "ultralytics" in set(_distribution_names(extras["track"]))
    assert "ultralytics" in set(_distribution_names(extras["all"]))


def test_every_copyleft_distribution_is_adjudicated(pyproject):
    # A licence check matches per distribution *name*, so excepting `ultralytics`
    # alone would leave its two AGPL companions flagged and un-adjudicated.
    exceptions = pyproject["tool"]["wads"]["licence"]["exceptions"]
    assert set(exceptions) == set(COPYLEFT_DISTRIBUTIONS)
    for name, reason in exceptions.items():
        assert "AGPL" in reason, f"{name}'s exception does not state the licence"


def test_the_licence_table_uses_the_schema_the_checker_reads(pyproject):
    # wads' LicencePolicy rejects unknown keys deliberately, so a typo here would be
    # a hard error rather than a silent fallback to defaults. Keep it parseable.
    known = {
        "enabled",
        "allowed",
        "forbidden",
        "exceptions",
        "include-extras",
        "unknown-is-failure",
        "unclassified-is-failure",
    }
    table = pyproject["tool"]["wads"]["licence"]
    assert set(table) <= known, f"unknown keys: {sorted(set(table) - known)}"
    assert isinstance(table["exceptions"], dict)  # a name -> reason map, not a list
    assert set(table["include-extras"]) == COPYLEFT_EXTRAS


def test_missing_ultralytics_names_the_extra_and_the_licence(monkeypatch):
    # The whole point of routing every ultralytics import through `_import_yolo`:
    # a user who reaches the tracking path without the extra should be told which
    # extra to install and what it costs, not get a bare ModuleNotFoundError.
    import builtins

    from kodokan.pose import _import_yolo

    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)
    with pytest.raises(ImportError) as excinfo:
        _import_yolo("the tracking path")
    message = str(excinfo.value)
    assert "kodokan[track]" in message
    assert "AGPL" in message
    assert "the tracking path" in message
