"""Watch a drop folder and auto-run the resume screener."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_settings
from src.integrations.google_sync import (
    GoogleIntegrationError,
    append_shortlist_to_sheet,
    create_calendar_holds,
)
from src.ranking_agent import ResumeRankingAgent
from src.resume_processing import SUPPORTED_SUFFIXES, derive_job_label, load_resumes_from_paths

from scripts.run_screening import (
    _discover_resumes,
    _generate_invites,
    _rankings_to_dataframe,
)


@dataclass
class WatchConfig:
    job_path: Path
    watch_dir: Path
    output_dir: Path
    top_k: int
    use_llm: bool
    generate_invites: bool
    invite_dir: Path
    debounce_seconds: float
    job_name: str
    sync_sheet: bool
    create_events: bool


class DropZoneHandler(FileSystemEventHandler):
    """Schedules screening runs whenever resumes arrive."""

    def __init__(self, config: WatchConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def on_created(self, event: FileSystemEvent) -> None:  # pragma: no cover - I/O glue
        self._maybe_schedule(event)

    def on_moved(self, event: FileSystemEvent) -> None:  # pragma: no cover - I/O glue
        self._maybe_schedule(event)

    def on_modified(self, event: FileSystemEvent) -> None:  # pragma: no cover - I/O glue
        self._maybe_schedule(event)

    def _maybe_schedule(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if event.is_directory or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return
        self._schedule_run()

    def _schedule_run(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.config.debounce_seconds, self._process_dropzone)
            self._timer.daemon = True
            self._timer.start()

    def _process_dropzone(self) -> None:
        try:
            perform_screening(self.config)
        finally:
            with self._lock:
                self._timer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a folder for new resumes and auto-generate shortlists",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job", required=True, help="Path to the job description text file")
    parser.add_argument(
        "--watch-dir",
        required=True,
        help="Directory where resumes will be dropped (PDF/DOCX/TXT)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/watch",
        help="Directory where timestamped shortlist CSVs are written",
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
        help="Emit interview invite drafts for Interview-ready candidates",
    )
    parser.add_argument(
        "--invite-dir",
        default="output/invites",
        help="Where to store invite drafts if enabled",
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=5.0,
        help="Wait time after the last file event before running a screening pass",
    )
    parser.add_argument(
        "--run-initial",
        action="store_true",
        help="Run one screening pass immediately on startup",
    )
    parser.add_argument(
        "--job-name",
        help="Optional label stored in Google automations (defaults to job file name)",
    )
    parser.add_argument(
        "--sync-sheet",
        action="store_true",
        help="Append each run to the configured Google Sheet",
    )
    parser.add_argument(
        "--create-events",
        action="store_true",
        help="Create calendar holds for Interview-ready candidates",
    )
    return parser.parse_args()


def perform_screening(config: WatchConfig) -> None:
    settings = get_settings()
    job_description = config.job_path.read_text(encoding="utf-8")
    resume_paths = _discover_resumes(config.watch_dir)
    if not resume_paths:
        print("[watcher] No resumes found yet; waiting for new files.")
        return

    if settings.max_resumes is not None and len(resume_paths) > settings.max_resumes:
        resume_paths = resume_paths[: settings.max_resumes]
        print(
            f"[watcher] Processing first {settings.max_resumes} resumes (set MAX_RESUMES for unlimited runs)."
        )

    profiles = load_resumes_from_paths(resume_paths)
    agent = ResumeRankingAgent(require_google_key=config.use_llm)
    rankings = agent.rank(
        job_description,
        profiles,
        top_k=config.top_k,
        use_llm=config.use_llm,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"shortlist_{timestamp}.csv"
    df = _rankings_to_dataframe(rankings)
    df.to_csv(output_path, index=False)

    if config.generate_invites:
        invite_root = config.invite_dir / timestamp
        _generate_invites(rankings, invite_root)

    if config.sync_sheet:
        try:
            result = append_shortlist_to_sheet(settings, config.job_name, rankings)
            if result.count:
                print(f"[watcher] Synced {result.count} candidate(s) to Google Sheets.")
        except GoogleIntegrationError as exc:
            print(f"[watcher] Google Sheets sync failed: {exc}")

    if config.create_events:
        try:
            result = create_calendar_holds(settings, config.job_name, rankings)
            if result.count:
                print(f"[watcher] Created {result.count} calendar hold(s).")
            else:
                print("[watcher] No Interview-ready candidates; calendar unchanged.")
        except GoogleIntegrationError as exc:
            print(f"[watcher] Calendar automation failed: {exc}")

    print(
        f"[watcher] {len(rankings)} candidates scored at {timestamp}. Shortlist saved to {output_path}."
    )


def main() -> None:
    args = parse_args()
    job_path = Path(args.job)
    watch_dir = Path(args.watch_dir)
    output_dir = Path(args.output_dir)
    invite_dir = Path(args.invite_dir)
    for path in (job_path, watch_dir):
        if path == watch_dir and not watch_dir.exists():
            watch_dir.mkdir(parents=True, exist_ok=True)
            continue
        if not path.exists():
            raise SystemExit(f"Required path does not exist: {path}")

    job_description = job_path.read_text(encoding="utf-8")

    config = WatchConfig(
        job_path=job_path,
        watch_dir=watch_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        use_llm=args.use_llm,
        generate_invites=args.generate_invites,
        invite_dir=invite_dir,
        debounce_seconds=args.debounce_seconds,
        job_name=args.job_name or derive_job_label(job_description, fallback=job_path.stem),
        sync_sheet=args.sync_sheet,
        create_events=args.create_events,
    )

    if args.run_initial:
        perform_screening(config)

    handler = DropZoneHandler(config)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    print(f"[watcher] Monitoring {watch_dir} for new resumes... Press Ctrl+C to stop.")

    try:
        observer.start()
        while True:  # pragma: no cover - long running loop
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - CLI exit
        print("\n[watcher] Stopping watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
