---
name: kodokan-competitions
description: >-
  Refresh the calendar of upcoming youth judo competitions in a French département (by default the Bouches-du-Rhône, around Aix-en-Provence) and say, for each, who may enter: open to any licensed judoka, by qualification, a friendly (amical) interclub limited to named clubs, or by invitation, judged against the user's home club. Publishes the result to the Resources page of judo.thorwhalen.com. Use when asked to "refresh/update the competition calendar", "which competitions can we go to", "what judo competitions are coming up in Aix / the 13 / Bouches-du-Rhône", "is this tournament open to our club", "add this interclub to the calendar", or when the calendar on the site looks stale.
---

# kodokan-competitions: refresh the competition calendar

The deterministic part is code (`kodokan.competitions`); judging eligibility from prose rules is yours. A refresh takes about ten minutes.

## Before anything: the local config, never in git

This repository is **public**. The user's home club, district, children, and any message from their club stay out of it, out of commits, issues and PRs. They live in two local files:

- `~/.config/kodokan/competitions.toml`: `home_club` (the club's name as the site shows it), `home_club_district` (`"1"`, `"2"` or unset when unknown), `area` (e.g. `"Aix-en-Provence et Bouches-du-Rhône"`), `postcode_prefixes` (default `["13"]`), `site` (default the Comité 13 site).
- `~/.config/kodokan/competitions-manual.json`: events no federation source publishes (friendly interclubs announced by the club, by email or on Facebook), in the published event format below, each with `"confidence": "confirmed"` and a `source_url` or a note saying who announced it.

If the config is missing, ask the user once for the home club (and its district if they know it), write the file, and carry on.

## 1. Snapshot the sources

```bash
python -m kodokan.competitions snapshot $PP/_agent_work/kodokan/<YYYY-MM-DD>-calendar/snap
```

It writes `agenda.json` (the area's events from today on, from the FFJDA agenda embedded in the committee's `/evenement` page), `details/<agenda_id>.txt` (each event's detail page: birth years, belts, weigh-in, rules, registration window) and `forecast.txt` (the committee's season forecast PDF as text; the link is re-read from `/calendriers-8` every time because the file name changes). Needs `pdftotext` (poppler).

Why both: the agenda details only the next four to six weeks; the forecast PDF is the only source for the rest of the season, and it is a forecast (it has typos, such as a 2027 in an October 2026 line).

## 2. Build the events

Start from the previously published file (`~/.local/share/kodokan/resources/competitions.json`, or `$KODOKAN_APP_DATA_DIR/resources/...`) so the notes you or the user added are carried forward, then:

1. **Merge the agenda.** One published event per competition day and place: the agenda lists a Shiren as four events (benjamins, minimes, cadets, juniors); publish one with all four categories. Drop veterans-only events (Master Provence) and grading competitions unless the user asks. Keep jujitsu / JJB events only when youth categories enter, and say "jujitsu" or "JJB" in the title, since a judoka's parent will otherwise read it as judo.
2. **Read every detail file** and fill `birth_years`, `categories`, `deadline` (the extranet closing date), `registration` and the eligibility fields from the rules text.
3. **Add forecast events** that the agenda does not have yet, with `"confidence": "announced"`, and a note that the rules are not published yet. Never invent a date, a venue or a rule: unknown is `null` or `"unknown"`.
4. **Add the manual events** from `competitions-manual.json`.
5. **Drop past events** (`date_end` before today).

## 3. Judge eligibility (the part that needs reading)

| `eligibility` | When | `home_club_eligible` |
|---|---|---|
| `open` | any licensed FFJDA judoka of the right age and belt may be entered by their club | `true` |
| `qualification` | only those qualified by an earlier event (a championship, a circuit final, a criterium) | `null`, and the note names the qualifying event |
| `interclub` | a friendly (amical) event among named clubs, or limited to one district's clubs | `true` only if the home club is in `eligible_clubs` (or in that district); `false` if not; `null` if the district is unknown |
| `invitation` | selection or designation by the committee (stages, Groupe 13 tests) | `null` |
| `unknown` | rules not published | `null` |

`eligibility_note` is one French sentence a parent can act on: the belt minimum, the exclusions (e.g. "exclus : qualifiés national N-1"), and anything that changes the answer. Registration is almost always by the club through the FFJDA extranet: say so, and give the closing date. A friendly interclub that the user's club is not part of is still worth listing, marked `false`, so nobody asks twice.

## 4. Write, check, publish

Write the calendar with `kodokan.competitions.write_calendar(cal, path)`, which refuses anything `validate_calendar` flags:

```python
{
    "updated": "YYYY-MM-DD",
    "home_club": "<from the config>",
    "area": "<from the config>",
    "sources": [
        {"name": "Agenda du Comité 13 (FFJDA)", "url": "..."},
        {"name": "Calendrier prévisionnel 2026-2027 (PDF)", "url": "..."},
    ],
    "events": [...],
}
```

Each event has exactly the keys in `kodokan.competitions.EVENT_FIELDS`: `id` (a slug with the date), `date_start`, `date_end`, `title`, `town`, `venue`, `organiser`, `level` (`departemental`, `regional`, `national`), `categories`, `birth_years`, `format`, `eligibility`, `eligible_clubs`, `home_club_eligible`, `eligibility_note`, `registration`, `deadline`, `source_url`, `confidence`.

The path is the kodokan **app's** data root, `~/.local/share/kodokan/resources/competitions.json`. Then publish it from the platform repository (it ships the app's data dirs and never deletes):

```bash
cd $PP/tt/tw_platform && python deploy.py cmd-push-data --app kodokan
```

Verify https://judo.thorwhalen.com/api/kodokan/resources/competitions.json returns the new `updated` date and that https://judo.thorwhalen.com/fr/resources shows the events.

## 5. Report

Tell the user in a few lines: how many events, what changed since the last refresh (new events, deadlines in the next two weeks, events newly confirmed), and what needs them (an unknown district, an interclub only the club can confirm, a registration closing soon). Keep the working folder in `$PP/_agent_work/kodokan/`, never in this repository.

## Sources and what they cannot tell you

- The Comité 13 agenda (`/evenement`) and its detail pages: no login, reliable, four to six weeks ahead. The Ligue Sud site (`https://ligue-sud-judo.ffjudo.com/evenement`) runs the same engine and is a cross-check (`--site`).
- The forecast PDF: the whole season, provisional.
- **Not online:** friendly interclub tournaments (sent by email to clubs, or on Facebook), and which district a club belongs to. Ask the user; record the answer in the local config.
- **Needs a human:** the FFJDA extranet, where clubs enter judokas.
