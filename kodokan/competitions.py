"""A calendar of upcoming judo competitions in one French département, and who may enter each.

The French judo federation (FFJDA) runs one agenda engine behind every regional and
departmental site. Each site's ``/evenement`` page embeds the *whole national agenda* as
JSON in a hidden form field, so one GET gives every published event with its town,
postcode, organiser, level and the extranet registration window; each event also has a
detail page with the birth years, minimum belt and the rules of entry. The department's
committee additionally publishes a season forecast as a PDF, which is the only source for
events more than four to six weeks ahead.

This module does the deterministic part of refreshing a calendar: fetch the agenda,
select the area, fetch the detail pages and the forecast, and write a *snapshot* folder.
Deciding who may enter each event (open, by qualification, a friendly interclub limited to
named clubs) is reading prose rules, and is left to the person or agent doing the refresh:
the ``kodokan-competitions`` skill is that procedure. The result is validated with
:func:`validate_calendar` before it is published.

>>> html = '<input type="hidden" name="evenements" value=\\'{"2026": {"11": [{"id_evenement": "1", "titre": "SHIREN", "codePostal": "13015", "date_debut": "1794096000", "date_fin": "1794096000", "libelleType": "COMPETITION", "proprietaire": "Comité X", "slug": "shiren"}]}}\\' />'
>>> [e["titre"] for e in parse_agenda(html)]
['SHIREN']
>>> normalise(parse_agenda(html)[0])["date_start"]
'2026-11-08'

Nothing here names a club: the home club, the area and the output path are arguments
(or a local config file the skill reads), so the module serves any club in any département.
"""

from __future__ import annotations

import html as _html
import json
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

#: The Bouches-du-Rhône committee's site; any FFJDA committee or league site has the same engine.
DFLT_SITE = "https://comite13judo.ffjudo.com"
DFLT_POSTCODE_PREFIXES = ("13",)
#: Organisers whose events count even when held outside the area (regional league, the committee).
DFLT_OWNERS = ("BOUCHES DU RHONE", "PACA")
#: Event types that are not competitions or events a young judoka enters.
DFLT_EXCLUDE_TYPES = ("PASSAGE DE GRADE",)
USER_AGENT = "kodokan-competitions/0.1 (+https://github.com/thorwhalen/kodokan)"
#: The agenda stores dates as epoch seconds at local midnight (Europe/Paris).
_PARIS = timezone(timedelta(hours=1))

ELIGIBILITY = ("open", "qualification", "interclub", "invitation", "unknown")
CONFIDENCE = ("confirmed", "announced", "inferred")
#: Keys of one published event, and the type each must have (``None`` allowed where listed).
EVENT_FIELDS: Mapping[str, tuple[type, ...]] = {
    "id": (str,),
    "date_start": (str,),
    "date_end": (str,),
    "title": (str,),
    "town": (str,),
    "venue": (str, type(None)),
    "organiser": (str, type(None)),
    "level": (str,),
    "categories": (list,),
    "birth_years": (list, type(None)),
    "format": (str, type(None)),
    "eligibility": (str,),
    "eligible_clubs": (list,),
    "home_club_eligible": (bool, type(None)),
    "eligibility_note": (str,),
    "registration": (str, type(None)),
    "deadline": (str, type(None)),
    "source_url": (str,),
    "confidence": (str,),
}
CALENDAR_FIELDS = ("updated", "home_club", "area", "sources", "events")


def http_get(url: str, *, timeout: float = 30.0) -> bytes:
    """GET ``url`` with an identifying User-Agent (the default fetcher)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_agenda(page_html: str) -> list[dict]:
    """Every event in an FFJDA site's ``/evenement`` page (the embedded national agenda).

    The JSON sits in ``<input name="evenements" value='...'>``; it is decoded with
    ``raw_decode`` from the opening brace, because apostrophes inside the JSON make
    searching for the closing quote unreliable.
    """
    m = re.search(r"""name=["']evenements["'][^>]*?value=(["'])""", page_html)
    if not m:
        raise ValueError("no embedded agenda (input name='evenements') in this page")
    data, _ = json.JSONDecoder().raw_decode(page_html[m.end() :])
    return [e for months in data.values() for evs in months.values() for e in evs]


def _iso(epoch) -> str | None:
    if epoch in (None, "", "None", "0"):
        return None
    return datetime.fromtimestamp(int(epoch), _PARIS).date().isoformat()


def detail_url(ev: Mapping, *, site: str = DFLT_SITE) -> str:
    return f"{site}/evenement/{ev['slug']}/{ev['id_evenement']}"


def normalise(ev: Mapping, *, site: str = DFLT_SITE) -> dict:
    """The fields of a raw agenda event that a calendar needs, with ISO dates."""
    return {
        "agenda_id": str(ev["id_evenement"]),
        "title": " ".join(str(ev.get("titre", "")).split()),
        "date_start": _iso(ev.get("date_debut")),
        "date_end": _iso(ev.get("date_fin")),
        "registration_opens": _iso(ev.get("dateOuverture")),
        "registration_closes": _iso(ev.get("dateCloture")),
        "venue": ev.get("lieu") or None,
        "address": " ".join(x for x in (ev.get("adresse1"), ev.get("adresse2")) if x)
        or None,
        "postcode": ev.get("codePostal") or None,
        "town": ev.get("ville") or None,
        "organiser": ev.get("proprietaire") or None,
        "level": ev.get("niveau") or None,  # D departmental, L league, CL, F federal
        "type": ev.get("libelleType") or None,
        "discipline": ev.get("discipline") or None,
        "url": detail_url(ev, site=site),
    }


def in_area(
    ev: Mapping,
    *,
    postcode_prefixes: Sequence[str] = DFLT_POSTCODE_PREFIXES,
    owners: Sequence[str] = DFLT_OWNERS,
    exclude_types: Sequence[str] = DFLT_EXCLUDE_TYPES,
) -> bool:
    """Whether an agenda event belongs in the area's calendar."""
    if (ev.get("libelleType") or "") in exclude_types:
        return False
    pc = str(ev.get("codePostal") or "")
    owner = _plain(ev.get("proprietaire") or "")
    return pc.startswith(tuple(postcode_prefixes)) or any(
        _plain(o) in owner for o in owners
    )


def _plain(s: str) -> str:
    """Upper case without accents, so 'Rhône' matches 'RHONE'."""
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).upper()


def detail_text(page_html: str) -> str:
    """The readable body of an event detail page: dates, venue, categories, birth years,
    belts and the free-text rules (weigh-in times, who may not enter, fees)."""
    body = re.sub(r"(?is)<(script|style|nav|header|footer)\b.*?</\1>", " ", page_html)
    body = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d)>", "\n", body)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", body))
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    start = text.find("Informations")
    end = text.find("Retour à l'agenda")
    return text[start if start >= 0 else 0 : end if end > start else None].strip()


def forecast_pdf_url(calendars_page_html: str, *, site: str = DFLT_SITE) -> str | None:
    """The link to the season's forecast calendar PDF on the committee's calendars page.

    Its file name changes with every revision, so it is always read from the page.
    """
    links = re.findall(r'href="([^"]+?\.pdf)"', calendars_page_html, flags=re.I)
    cal = [u for u in links if "calendrier" in u.lower()] or links
    if not cal:
        return None
    u = _html.unescape(cal[0])
    return u if u.startswith("http") else site + ("" if u.startswith("/") else "/") + u


def pdf_text(pdf: bytes) -> str:
    """The text of a PDF, laid out (``pdftotext -layout``, from poppler)."""
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError(
            "pdftotext is needed to read the forecast PDF: `brew install poppler` (macOS) "
            "or `apt install poppler-utils` (Debian/Ubuntu)"
        )
    r = subprocess.run(
        [exe, "-layout", "-", "-"], input=pdf, capture_output=True, check=True
    )
    return r.stdout.decode("utf-8", "replace")


def snapshot(
    out_dir: str | Path,
    *,
    site: str = DFLT_SITE,
    calendars_path: str = "/calendriers-8",
    since: date | None = None,
    get: Callable[[str], bytes] = http_get,
    details: bool = True,
    forecast: bool = True,
    **area,
) -> dict:
    """Fetch everything a calendar refresh reads, into ``out_dir``.

    Writes ``agenda.json`` (the area's events from ``since`` on, normalised),
    ``details/<agenda_id>.txt`` (each one's detail page as text) and ``forecast.txt``
    (the season forecast PDF as text, with ``forecast_url.txt``). Returns a summary.
    ``area`` takes :func:`in_area`'s keyword arguments.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    since = since or date.today()
    raw = parse_agenda(get(f"{site}/evenement").decode("utf-8", "replace"))
    evs = [normalise(e, site=site) for e in raw if in_area(e, **area)]
    evs = sorted(
        (
            e
            for e in evs
            if (e["date_end"] or e["date_start"] or "") >= since.isoformat()
        ),
        key=lambda e: (e["date_start"] or "", e["title"]),
    )
    (out / "agenda.json").write_text(json.dumps(evs, ensure_ascii=False, indent=1))
    summary = {
        "agenda_total": len(raw),
        "in_area": len(evs),
        "details": 0,
        "forecast": None,
    }
    if details:
        d = out / "details"
        d.mkdir(exist_ok=True)
        for e in evs:
            try:
                (d / f"{e['agenda_id']}.txt").write_text(
                    e["url"]
                    + "\n"
                    + detail_text(get(e["url"]).decode("utf-8", "replace"))
                )
                summary["details"] += 1
            except OSError as err:  # one unreachable page must not lose the rest
                (d / f"{e['agenda_id']}.error").write_text(f"{e['url']}\n{err}\n")
    if forecast:
        url = forecast_pdf_url(
            get(site + calendars_path).decode("utf-8", "replace"), site=site
        )
        if url:
            (out / "forecast_url.txt").write_text(url + "\n")
            (out / "forecast.txt").write_text(pdf_text(get(url)))
            summary["forecast"] = url
    return summary


def validate_calendar(cal: Mapping) -> list[str]:
    """Problems with a calendar about to be published (``[]`` means publishable).

    Checks the envelope and every event's keys, types, enumerations and dates, so a
    hand-edited refresh cannot ship a file the web page would mis-render.
    """
    problems = [f"missing top-level key {k!r}" for k in CALENDAR_FIELDS if k not in cal]
    for i, ev in enumerate(cal.get("events", [])):
        where = f"event {i} ({ev.get('id', '?')})"
        for k, types in EVENT_FIELDS.items():
            if k not in ev:
                problems.append(f"{where}: missing {k!r}")
            elif not isinstance(ev[k], types):
                problems.append(f"{where}: {k!r} is {type(ev[k]).__name__}")
        extra = set(ev) - set(EVENT_FIELDS)
        if extra:
            problems.append(f"{where}: unknown keys {sorted(extra)}")
        if ev.get("eligibility") not in ELIGIBILITY:
            problems.append(f"{where}: eligibility must be one of {ELIGIBILITY}")
        if ev.get("confidence") not in CONFIDENCE:
            problems.append(f"{where}: confidence must be one of {CONFIDENCE}")
        for k in ("date_start", "date_end", "deadline"):
            v = ev.get(k)
            if isinstance(v, str):
                try:
                    date.fromisoformat(v)
                except ValueError:
                    problems.append(f"{where}: {k} {v!r} is not YYYY-MM-DD")
        if ev.get("eligibility") == "interclub" and not ev.get("eligible_clubs"):
            problems.append(f"{where}: an interclub event must name its clubs")
    return problems


def write_calendar(cal: Mapping, path: str | Path) -> Path:
    """Validate ``cal`` and write it as JSON, events sorted by date. Raises on any problem."""
    problems = validate_calendar(cal)
    if problems:
        raise ValueError("calendar not publishable:\n  " + "\n  ".join(problems))
    cal = {
        **cal,
        "events": sorted(cal["events"], key=lambda e: (e["date_start"], e["title"])),
    }
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
    return p


def _main(argv: Iterable[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m kodokan.competitions", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "snapshot", help="fetch the agenda, detail pages and forecast into a folder"
    )
    s.add_argument("out_dir")
    s.add_argument("--site", default=DFLT_SITE)
    s.add_argument(
        "--postcode", action="append", help="postcode prefix (repeatable; default 13)"
    )
    s.add_argument("--no-details", action="store_true")
    s.add_argument("--no-forecast", action="store_true")
    v = sub.add_parser(
        "validate", help="check a calendar JSON file before publishing it"
    )
    v.add_argument("path")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "snapshot":
        area = {"postcode_prefixes": tuple(a.postcode)} if a.postcode else {}
        print(
            json.dumps(
                snapshot(
                    a.out_dir,
                    site=a.site,
                    details=not a.no_details,
                    forecast=not a.no_forecast,
                    **area,
                ),
                indent=1,
            )
        )
    else:
        problems = validate_calendar(json.loads(Path(a.path).read_text()))
        print("\n".join(problems) or "ok")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    _main()
