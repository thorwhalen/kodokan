"""The competition-calendar fetcher, on synthetic pages (no network)."""

import json

import pytest

from kodokan import competitions as c


def _agenda_page(events):
    """An FFJDA-style /evenement page: the agenda as JSON in a hidden input (with an
    apostrophe inside, the trap that breaks naive end-of-value searches)."""
    data = {"2026": {"11": events}}
    return f"<html><form><input type=\"hidden\" name=\"evenements\" value='{json.dumps(data, ensure_ascii=False)}' /></form></html>"


def _ev(i, title, postcode, owner="Comité BOUCHES DU RHONE JUD", kind="COMPETITION", day=1794096000):
    return {"id_evenement": str(i), "titre": title, "codePostal": postcode, "proprietaire": owner,
            "libelleType": kind, "date_debut": str(day), "date_fin": str(day), "slug": f"ev-{i}",
            "ville": "AIX EN PROVENCE", "niveau": "D", "dateCloture": str(day - 4 * 86400)}


PAGES = {
    "https://x.test/evenement": _agenda_page([
        _ev(1, "Tournoi d'Aix", "13100"),
        _ev(2, "Passage de grade", "13100", kind="PASSAGE DE GRADE"),
        _ev(3, "Coupe régionale", "06000", owner="Ligue PACA JUDO"),
        _ev(4, "Championnat ailleurs", "75000", owner="Ligue IDF"),
        _ev(5, "Déjà passé", "13001", day=1700000000),
    ]),
    "https://x.test/calendriers-8": '<a href="/uploads/fichier/CALENDRIER%2013%2026_27.pdf">cal</a>',
}


def _get(url):
    if url in PAGES:
        return PAGES[url].encode()
    if "/evenement/" in url:
        return b"<nav>menu</nav><h1>x</h1><p>Informations</p><p>BENJAMIN</p><p>2015</p><p>Retour \xc3\xa0 l'agenda</p><footer>f</footer>"
    raise AssertionError(url)


def test_snapshot_selects_the_area_and_writes_details(tmp_path):
    s = c.snapshot(tmp_path, site="https://x.test", get=_get, forecast=False,
                   since=c.date(2026, 11, 1))
    evs = json.loads((tmp_path / "agenda.json").read_text())
    assert [e["title"] for e in evs] == ["Coupe régionale", "Tournoi d'Aix"]  # sorted, area only
    assert s == {"agenda_total": 5, "in_area": 2, "details": 2, "forecast": None}
    assert evs[1]["date_start"] == "2026-11-08" and evs[1]["registration_closes"] == "2026-11-04"
    assert (tmp_path / "details" / "1.txt").read_text().splitlines()[1:] == ["Informations", "BENJAMIN", "2015"]


def test_forecast_link_is_read_from_the_page():
    assert c.forecast_pdf_url(PAGES["https://x.test/calendriers-8"], site="https://x.test") == (
        "https://x.test/uploads/fichier/CALENDRIER%2013%2026_27.pdf")
    assert c.forecast_pdf_url("<p>nothing</p>") is None


def test_a_page_without_the_agenda_says_so():
    with pytest.raises(ValueError, match="no embedded agenda"):
        c.parse_agenda("<html></html>")


def _event(**kw):
    ev = {"id": "e1", "date_start": "2026-11-29", "date_end": "2026-11-29", "title": "Shiren",
          "town": "Saint-Martin-de-Crau", "venue": None, "organiser": None, "level": "departemental",
          "categories": ["benjamins"], "birth_years": [2015, 2016], "format": None, "eligibility": "open",
          "eligible_clubs": [], "home_club_eligible": True, "eligibility_note": "Ouvert.",
          "registration": None, "deadline": None, "source_url": "https://x.test/e/1", "confidence": "confirmed"}
    return {**ev, **kw}


def _cal(*events):
    return {"updated": "2026-11-08", "home_club": "Club", "area": "13", "sources": [], "events": list(events)}


def test_validate_accepts_a_good_calendar_and_writes_it_sorted(tmp_path):
    later, sooner = _event(id="b", date_start="2027-01-10"), _event(id="a")
    p = c.write_calendar(_cal(later, sooner), tmp_path / "competitions.json")
    assert [e["id"] for e in json.loads(p.read_text())["events"]] == ["a", "b"]


@pytest.mark.parametrize("bad, problem", [
    (_event(eligibility="friendly"), "eligibility must be one of"),
    (_event(date_start="29/11/2026"), "is not YYYY-MM-DD"),
    (_event(eligibility="interclub", eligible_clubs=[]), "must name its clubs"),
    (_event(auc_eligible=True), "unknown keys"),
    ({k: v for k, v in _event().items() if k != "town"}, "missing 'town'"),
])
def test_validate_names_each_problem(bad, problem, tmp_path):
    assert any(problem in p for p in c.validate_calendar(_cal(bad)))
    with pytest.raises(ValueError, match="not publishable"):
        c.write_calendar(_cal(bad), tmp_path / "x.json")
