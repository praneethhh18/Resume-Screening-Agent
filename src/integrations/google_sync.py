"""Google Sheets and Calendar integration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import Settings
from src.ranking_agent import RankedResume
from src.resume_processing import summarize_segments

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
SHEET_HEADERS = [
    "generated_at_utc",
    "job_name",
    "candidate",
    "email",
    "filename",
    "next_step",
    "fit_score",
    "heuristic_score",
    "similarity",
    "verdict",
    "strengths",
    "gaps",
    "matched_keywords",
    "missing_keywords",
    "experience_years",
    "summary",
    "links",
]


class GoogleIntegrationError(RuntimeError):
    """Raised when Google automation cannot proceed."""


@dataclass
class IntegrationResult:
    action: str
    count: int


def append_shortlist_to_sheet(
    settings: Settings, job_name: str, rankings: Iterable[RankedResume]
) -> IntegrationResult:
    if not settings.google_sheets_id:
        raise GoogleIntegrationError("GOOGLE_SHEETS_ID is not configured.")
    creds = _load_credentials(settings, [SHEETS_SCOPE])
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    _ensure_headers(service, settings)

    values = []
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    for entry in rankings:
        summary = entry.profile.summary or summarize_segments(entry.profile.raw_text)
        values.append(
            [
                timestamp,
                job_name,
                entry.profile.display_name,
                entry.profile.email or "",
                entry.profile.filename,
                entry.next_step,
                entry.fit_score,
                round(entry.heuristic_score, 2),
                round(entry.similarity, 3),
                entry.verdict,
                " | ".join(entry.strengths),
                " | ".join(entry.gaps),
                ", ".join(entry.matched_keywords),
                ", ".join(entry.missing_keywords),
                entry.experience_years or "",
                summary[:500],
                ", ".join(entry.profile.links),
            ]
        )

    if not values:
        return IntegrationResult(action="sheet", count=0)

    body = {"values": values}
    service.spreadsheets().values().append(
        spreadsheetId=settings.google_sheets_id,
        range=f"{settings.google_sheets_tab}!A:Q",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

    return IntegrationResult(action="sheet", count=len(values))


def create_calendar_holds(
    settings: Settings, job_name: str, rankings: Iterable[RankedResume]
) -> IntegrationResult:
    if not settings.google_calendar_id:
        raise GoogleIntegrationError("GOOGLE_CALENDAR_ID is not configured.")

    interviews = [entry for entry in rankings if entry.next_step == "Interview"]
    if not interviews:
        return IntegrationResult(action="calendar", count=0)

    creds = _load_credentials(settings, [CALENDAR_SCOPE])
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    tz = _safe_timezone(settings.interview_timezone)
    start_hour, start_minute = _parse_time(settings.interview_start_time)
    base_date = _next_business_date(settings.interview_days_offset)
    base_start = datetime.combine(base_date, time(start_hour, start_minute), tz)
    slot_delta = timedelta(minutes=max(15, settings.interview_duration_min))

    created = 0
    current_start = base_start
    for entry in interviews:
        description = _build_event_description(job_name, entry)
        event = {
            "summary": f"Interview – {entry.profile.display_name}",
            "description": description,
            "start": {"dateTime": current_start.isoformat(), "timeZone": tz.key},
            "end": {
                "dateTime": (current_start + slot_delta).isoformat(),
                "timeZone": tz.key,
            },
            "status": "tentative",
        }
        # Personal Gmail calendars + service accounts cannot invite attendees without
        # domain-wide delegation, so we create holds without adding guests.
        try:
            service.events().insert(
                calendarId=settings.google_calendar_id,
                body=event,
                sendUpdates="none",
            ).execute()
            created += 1
            current_start += slot_delta
        except HttpError as exc:  # pragma: no cover - runtime API error
            raise GoogleIntegrationError(f"Calendar event creation failed: {exc}") from exc

    return IntegrationResult(action="calendar", count=created)


def _build_event_description(job_name: str, entry: RankedResume) -> str:
    strengths = " | ".join(entry.strengths) or "n/a"
    gaps = " | ".join(entry.gaps) or "n/a"
    notes = " | ".join(entry.heuristic_notes) or "n/a"
    summary = entry.profile.summary or summarize_segments(entry.profile.raw_text)
    desc = (
        f"Job: {job_name}\nFit score: {entry.fit_score}\nHeuristic score: {entry.heuristic_score:.1f}\n"
        f"Strengths: {strengths}\nGaps: {gaps}\nNotes: {notes}\n\nSummary: {summary[:400]}"
    )
    if not entry.profile.email:
        desc += "\n\nCandidate email missing; add manually."
    return desc


def _load_credentials(settings: Settings, scopes: Sequence[str]):
    if not settings.google_service_account_file:
        raise GoogleIntegrationError("GOOGLE_SERVICE_ACCOUNT_FILE is not configured.")
    path = Path(settings.google_service_account_file)
    if not path.exists():
        raise GoogleIntegrationError(f"Service account file not found: {path}")
    return service_account.Credentials.from_service_account_file(str(path), scopes=list(scopes))


def _safe_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:  # pragma: no cover - fallback path
        return ZoneInfo("UTC")


def _parse_time(value: str) -> tuple[int, int]:
    try:
        hours, minutes = value.split(":", 1)
        return int(hours), int(minutes)
    except Exception:
        return 10, 0


def _next_business_date(offset_days: int) -> date:
    current = date.today() + timedelta(days=max(0, offset_days))
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _ensure_headers(service, settings: Settings) -> None:
    try:
        result = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=settings.google_sheets_id,
                range=f"{settings.google_sheets_tab}!1:1",
            )
            .execute()
        )
        values = result.get("values", [])
        if values and values[0][: len(SHEET_HEADERS)] == SHEET_HEADERS:
            return
    except HttpError:
        pass

    service.spreadsheets().values().update(
        spreadsheetId=settings.google_sheets_id,
        range=f"{settings.google_sheets_tab}!1:1",
        valueInputOption="RAW",
        body={"values": [SHEET_HEADERS]},
    ).execute()
