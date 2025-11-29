"""Offline screening runner for automated resume intake."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import get_settings
from src.integrations.google_sync import (
    GoogleIntegrationError,
    append_shortlist_to_sheet,
    create_calendar_holds,
)
from src.ranking_agent import RankedResume, ResumeRankingAgent
from src.resume_processing import SUPPORTED_SUFFIXES, derive_job_label, load_resumes_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score resumes from a folder and export the shortlist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job", required=True, help="Path to a job description text file")
    parser.add_argument(
        "--resume-dir",
        required=True,
        help="Directory containing PDF/DOCX/TXT resumes to evaluate",
    )
    parser.add_argument(
        "--output",
        default="output/shortlist.xlsx",
        help="Where to write the consolidated shortlist (csv or xlsx)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many candidates to surface")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable Gemini reasoning (requires GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--generate-invites",
        action="store_true",
        help="Create draft invite files for Interview-ready candidates",
    )
    parser.add_argument(
        "--invite-dir",
        default="output/invites",
        help="Folder to store draft invite text files",
    )
    parser.add_argument(
        "--job-name",
        help="Optional label stored in Google integrations (defaults to job file name)",
    )
    parser.add_argument(
        "--sync-sheet",
        action="store_true",
        help="Append shortlist rows to the configured Google Sheet",
    )
    parser.add_argument(
        "--create-events",
        action="store_true",
        help="Create tentative Google Calendar events for Interview-ready candidates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    job_path = Path(args.job)
    job_description = job_path.read_text(encoding="utf-8")
    resume_paths = _discover_resumes(Path(args.resume_dir))
    if not resume_paths:
        raise SystemExit("No resumes found in the provided directory.")
    if settings.max_resumes is not None and len(resume_paths) > settings.max_resumes:
        resume_paths = resume_paths[: settings.max_resumes]
        print(
            f"Processing first {settings.max_resumes} resumes (adjust MAX_RESUMES in .env to change)."
        )

    profiles = load_resumes_from_paths(resume_paths)
    agent = ResumeRankingAgent(require_google_key=args.use_llm)
    rankings = agent.rank(
        job_description,
        profiles,
        top_k=args.top_k,
        use_llm=args.use_llm,
    )

    df = _rankings_to_dataframe(rankings)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)

    job_name = args.job_name or derive_job_label(job_description, fallback=job_path.stem)

    if args.generate_invites:
        _generate_invites(rankings, Path(args.invite_dir))

    if args.sync_sheet:
        try:
            result = append_shortlist_to_sheet(settings, job_name, rankings)
            print(f"Synced {result.count} candidate(s) to Google Sheet {settings.google_sheets_id}.")
        except GoogleIntegrationError as exc:
            print(f"[warn] Google Sheets sync failed: {exc}")

    if args.create_events:
        try:
            result = create_calendar_holds(settings, job_name, rankings)
            if result.count:
                print(f"Created {result.count} calendar hold(s) on {settings.google_calendar_id}.")
            else:
                print("No Interview-ready candidates; no calendar events were created.")
        except GoogleIntegrationError as exc:
            print(f"[warn] Calendar automation failed: {exc}")

    print(f"Saved shortlist with {len(rankings)} candidates to {output_path}")


def _discover_resumes(folder: Path) -> list[Path]:
    if not folder.exists():
        raise SystemExit(f"Resume directory does not exist: {folder}")
    paths = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
    return paths


def _rankings_to_dataframe(rankings: Iterable[RankedResume]) -> pd.DataFrame:
    rows = []
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    for rank, entry in enumerate(rankings, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate": entry.profile.display_name,
                "email": entry.profile.email or "",
                "links": ", ".join(entry.profile.links),
                "fit_score": entry.fit_score,
                "heuristic_score": entry.heuristic_score,
                "next_step": entry.next_step,
                "verdict": entry.verdict,
                "strengths": " | ".join(entry.strengths),
                "gaps": " | ".join(entry.gaps),
                "matched_keywords": ", ".join(entry.matched_keywords),
                "missing_keywords": ", ".join(entry.missing_keywords),
                "generated_at_utc": timestamp,
            }
        )
    return pd.DataFrame(rows)


def _generate_invites(rankings: Iterable[RankedResume], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    template = (
        "Hi {name},\n\n"
        "Thanks for sharing your resume for {role}. We'd love to invite you to a first-round interview.\n"
        "Please reply to this email with your availability this week.\n\n"
        "Best,\nRecruiting Team"
    )
    created = 0
    for entry in rankings:
        if entry.next_step != "Interview" or not entry.profile.email:
            continue
        body = template.format(name=entry.profile.display_name, role="the open role")
        filename = entry.profile.display_name.replace(" ", "_") + ".txt"
        (target_dir / filename).write_text(
            f"To: {entry.profile.email}\nSubject: Interview Invitation\n\n{body}",
            encoding="utf-8",
        )
        created += 1
    if created:
        print(f"Created {created} invite draft(s) in {target_dir}")


if __name__ == "__main__":
    main()
