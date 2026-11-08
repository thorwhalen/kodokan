# kodokan.competitions

A calendar of upcoming judo competitions in one French département, and who may enter each.

The French judo federation (FFJDA) runs one agenda engine behind every regional and
departmental site. Each site’s `/evenement` page embeds the *whole national agenda* as
JSON in a hidden form field, so one GET gives every published event with its town,
postcode, organiser, level and the extranet registration window; each event also has a
detail page with the birth years, minimum belt and the rules of entry. The department’s
committee additionally publishes a season forecast as a PDF, which is the only source for
events more than four to six weeks ahead.

This module does the deterministic part of refreshing a calendar: fetch the agenda,
select the area, fetch the detail pages and the forecast, and write a *snapshot* folder.
Deciding who may enter each event (open, by qualification, a friendly interclub limited to
named clubs) is reading prose rules, and is left to the person or agent doing the refresh:
the `kodokan-competitions` skill is that procedure. The result is validated with
[`validate_calendar()`](#kodokan.competitions.validate_calendar) before it is published.

```pycon
>>> html = '<input type="hidden" name="evenements" value=\'{"2026": {"11": [{"id_evenement": "1", "titre": "SHIREN", "codePostal": "13015", "date_debut": "1794096000", "date_fin": "1794096000", "libelleType": "COMPETITION", "proprietaire": "Comité X", "slug": "shiren"}]}}\' />'
>>> [e["titre"] for e in parse_agenda(html)]
['SHIREN']
>>> normalise(parse_agenda(html)[0])["date_start"]
'2026-11-08'
```

Nothing here names a club: the home club, the area and the output path are arguments
(or a local config file the skill reads), so the module serves any club in any département.

### Module Attributes

| [`DFLT_SITE`](#kodokan.competitions.DFLT_SITE)          | The Bouches-du-Rhône committee's site; any FFJDA committee or league site has the same engine.   |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [`DFLT_OWNERS`](#kodokan.competitions.DFLT_OWNERS)        | Organisers whose events count even when held outside the area (regional league, the committee).  |
| [`DFLT_EXCLUDE_TYPES`](#kodokan.competitions.DFLT_EXCLUDE_TYPES) | Event types that are not competitions or events a young judoka enters.                           |
| [`EVENT_FIELDS`](#kodokan.competitions.EVENT_FIELDS)       | Keys of one published event, and the type each must have (`None` allowed where listed).          |

### Functions

| [`detail_text`](#kodokan.competitions.detail_text)(page_html)                             | The readable body of an event detail page: dates, venue, categories, birth years, belts and the free-text rules (weigh-in times, who may not enter, fees).   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detail_url`(ev, \*[, site])                                                                        |                                                                                                                                                              |
| [`forecast_pdf_url`](#kodokan.competitions.forecast_pdf_url)(calendars_page_html, \*[, site])  | The link to the season's forecast calendar PDF on the committee's calendars page.                                                                            |
| [`http_get`](#kodokan.competitions.http_get)(url, \*[, timeout])                       | GET `url` with an identifying User-Agent (the default fetcher).                                                                                              |
| [`in_area`](#kodokan.competitions.in_area)(ev, \*[, postcode_prefixes, owners, ...])  | Whether an agenda event belongs in the area's calendar.                                                                                                      |
| [`normalise`](#kodokan.competitions.normalise)(ev, \*[, site])                          | The fields of a raw agenda event that a calendar needs, with ISO dates.                                                                                      |
| [`parse_agenda`](#kodokan.competitions.parse_agenda)(page_html)                            | Every event in an FFJDA site's `/evenement` page (the embedded national agenda).                                                                             |
| [`pdf_text`](#kodokan.competitions.pdf_text)(pdf)                                      | The text of a PDF, laid out (`pdftotext -layout`, from poppler).                                                                                             |
| [`snapshot`](#kodokan.competitions.snapshot)(out_dir, \*[, site, calendars_path, ...]) | Fetch everything a calendar refresh reads, into `out_dir`.                                                                                                   |
| [`validate_calendar`](#kodokan.competitions.validate_calendar)(cal)                             | Problems with a calendar about to be published (`[]` means publishable).                                                                                     |
| [`write_calendar`](#kodokan.competitions.write_calendar)(cal, path)                          | Validate `cal` and write it as JSON, events sorted by date.                                                                                                  |

### kodokan.competitions.DFLT_EXCLUDE_TYPES *= ('PASSAGE DE GRADE',)*

Event types that are not competitions or events a young judoka enters.

### kodokan.competitions.DFLT_OWNERS *= ('BOUCHES DU RHONE', 'PACA')*

Organisers whose events count even when held outside the area (regional league, the committee).

### kodokan.competitions.DFLT_SITE *= 'https://comite13judo.ffjudo.com'*

The Bouches-du-Rhône committee’s site; any FFJDA committee or league site has the same engine.

### kodokan.competitions.EVENT_FIELDS *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[type](https://docs.python.org/3/builtins/functions.html#type), ...]]* *= {'birth_years': (<class 'list'>, <class 'NoneType'>), 'categories': (<class 'list'>,), 'confidence': (<class 'str'>,), 'date_end': (<class 'str'>,), 'date_start': (<class 'str'>,), 'deadline': (<class 'str'>, <class 'NoneType'>), 'eligibility': (<class 'str'>,), 'eligibility_note': (<class 'str'>,), 'eligible_clubs': (<class 'list'>,), 'format': (<class 'str'>, <class 'NoneType'>), 'home_club_eligible': (<class 'bool'>, <class 'NoneType'>), 'id': (<class 'str'>,), 'level': (<class 'str'>,), 'organiser': (<class 'str'>, <class 'NoneType'>), 'registration': (<class 'str'>, <class 'NoneType'>), 'source_url': (<class 'str'>,), 'title': (<class 'str'>,), 'town': (<class 'str'>,), 'venue': (<class 'str'>, <class 'NoneType'>)}*

Keys of one published event, and the type each must have (`None` allowed where listed).

### kodokan.competitions.detail_text(page_html)

The readable body of an event detail page: dates, venue, categories, birth years,
belts and the free-text rules (weigh-in times, who may not enter, fees).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.competitions.forecast_pdf_url(calendars_page_html, , site='https://comite13judo.ffjudo.com')

The link to the season’s forecast calendar PDF on the committee’s calendars page.

Its file name changes with every revision, so it is always read from the page.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### kodokan.competitions.http_get(url, , timeout=30.0)

GET `url` with an identifying User-Agent (the default fetcher).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### kodokan.competitions.in_area(ev, , postcode_prefixes=('13',), owners=('BOUCHES DU RHONE', 'PACA'), exclude_types=('PASSAGE DE GRADE',))

Whether an agenda event belongs in the area’s calendar.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### kodokan.competitions.normalise(ev, , site='https://comite13judo.ffjudo.com')

The fields of a raw agenda event that a calendar needs, with ISO dates.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.competitions.parse_agenda(page_html)

Every event in an FFJDA site’s `/evenement` page (the embedded national agenda).

The JSON sits in `<input name="evenements" value='...'>`; it is decoded with
`raw_decode` from the opening brace, because apostrophes inside the JSON make
searching for the closing quote unreliable.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.competitions.pdf_text(pdf)

The text of a PDF, laid out (`pdftotext -layout`, from poppler).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.competitions.snapshot(out_dir, \*, site='https://comite13judo.ffjudo.com', calendars_path='/calendriers-8', since=None, get=<function http_get>, details=True, forecast=True, \*\*area)

Fetch everything a calendar refresh reads, into `out_dir`.

Writes `agenda.json` (the area’s events from `since` on, normalised),
`details/<agenda_id>.txt` (each one’s detail page as text) and `forecast.txt`
(the season forecast PDF as text, with `forecast_url.txt`). Returns a summary.
`area` takes [`in_area()`](#kodokan.competitions.in_area)’s keyword arguments.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.competitions.validate_calendar(cal)

Problems with a calendar about to be published (`[]` means publishable).

Checks the envelope and every event’s keys, types, enumerations and dates, so a
hand-edited refresh cannot ship a file the web page would mis-render.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### kodokan.competitions.write_calendar(cal, path)

Validate `cal` and write it as JSON, events sorted by date. Raises on any problem.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
