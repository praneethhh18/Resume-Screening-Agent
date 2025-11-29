"""Application configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

load_dotenv()

try:  # Only available when running inside Streamlit Cloud/local UI sessions
    import streamlit as st

    _STREAMLIT_SECRETS = st.secrets
except Exception:  # pragma: no cover - Streamlit not present outside UI runs
    _STREAMLIT_SECRETS = None


def _hydrate_from_streamlit_secrets() -> None:
    """Mirror Streamlit secrets into env vars + service account file."""

    if not _STREAMLIT_SECRETS:
        return

    for key, value in _STREAMLIT_SECRETS.items():
        if key == "GOOGLE_SERVICE_ACCOUNT_JSON":
            continue
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)

    if "GOOGLE_SERVICE_ACCOUNT_JSON" in _STREAMLIT_SECRETS:
        secrets_dir = Path("secrets")
        secrets_dir.mkdir(exist_ok=True)
        sa_path = secrets_dir / "service-account.json"
        sa_path.write_text(_STREAMLIT_SECRETS["GOOGLE_SERVICE_ACCOUNT_JSON"], encoding="utf-8")
        os.environ.setdefault("GOOGLE_SERVICE_ACCOUNT_FILE", str(sa_path))


_hydrate_from_streamlit_secrets()

DEFAULT_GEMINI_CANDIDATES: Sequence[str] = (
    "gemini-1.5-flash",
    "gemini-1.5-pro",
)

DISALLOWED_MODELS = {
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
}


@dataclass
class Settings:
    """Holds runtime configuration values."""

    google_api_key: Optional[str]
    gemini_model: str
    embeddings_model: str
    persist_directory: Path
    max_resumes: Optional[int] = None
    google_service_account_file: Optional[Path] = None
    google_sheets_id: Optional[str] = None
    google_sheets_tab: str = "Sheet1"
    google_calendar_id: Optional[str] = None
    interview_timezone: str = "UTC"
    interview_start_time: str = "10:00"
    interview_duration_min: int = 30
    interview_days_offset: int = 1

    def ensure_storage(self) -> None:
        """Make sure the local vector store directory exists."""

        self.persist_directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance with environment overrides."""

    persist_directory = Path(os.getenv("CHROMA_DB_DIR", "storage/chroma"))
    google_api_key = os.getenv("GOOGLE_API_KEY")
    max_resumes_env = os.getenv("MAX_RESUMES")
    max_resumes: Optional[int]
    if max_resumes_env is None or not max_resumes_env.strip():
        max_resumes = None
    else:
        try:
            parsed = int(max_resumes_env)
            max_resumes = parsed if parsed > 0 else None
        except ValueError:
            print("[config] MAX_RESUMES is not a valid integer; defaulting to unlimited uploads.")
            max_resumes = None

    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    sheets_id = os.getenv("GOOGLE_SHEETS_ID")
    sheets_tab = os.getenv("GOOGLE_SHEETS_TAB", "Sheet1")
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID")

    interview_timezone = os.getenv("INTERVIEW_TIMEZONE", "UTC")
    interview_start_time = os.getenv("INTERVIEW_START_TIME", "10:00")
    interview_duration_min = _safe_int(os.getenv("INTERVIEW_DURATION_MIN"), 30)
    interview_days_offset = _safe_int(os.getenv("INTERVIEW_DAYS_OFFSET"), 1)

    settings = Settings(
        google_api_key=google_api_key,
        gemini_model=_resolve_gemini_model(os.getenv("GEMINI_MODEL"), google_api_key),
        embeddings_model=os.getenv(
            "EMBEDDINGS_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        persist_directory=persist_directory,
        max_resumes=max_resumes,
        google_service_account_file=Path(service_account_path) if service_account_path else None,
        google_sheets_id=sheets_id,
        google_sheets_tab=sheets_tab,
        google_calendar_id=calendar_id,
        interview_timezone=interview_timezone,
        interview_start_time=interview_start_time,
        interview_duration_min=interview_duration_min,
        interview_days_offset=interview_days_offset,
    )
    settings.ensure_storage()
    return settings


def _resolve_gemini_model(explicit: Optional[str], api_key: Optional[str]) -> str:
    """Return the preferred Gemini model, falling back to auto-discovery."""

    if explicit:
        if explicit in DISALLOWED_MODELS:
            print(
                f"[config] GEMINI_MODEL '{explicit}' is blocked for this build; falling back to auto selection."
            )
        elif _model_accessible(explicit, api_key):
            return explicit
        else:
            print(
                f"[config] GEMINI_MODEL '{explicit}' is not available for the current API key; falling back to auto selection."
            )
    return _auto_select_gemini_model(api_key)


def _auto_select_gemini_model(api_key: Optional[str]) -> str:
    """Pick the first available Gemini model that supports content generation."""

    if not api_key:
        return DEFAULT_GEMINI_CANDIDATES[0]

    try:
        _configure_genai(api_key)
        models = list(genai.list_models())
    except Exception:
        return DEFAULT_GEMINI_CANDIDATES[0]

    usable_models = _filter_generate_capable(models)
    alias_map = _alias_map(usable_models)
    for candidate in DEFAULT_GEMINI_CANDIDATES:
        target = alias_map.get(candidate)
        if target and _model_accessible(target, api_key):
            return target

    for model in usable_models:
        short = _short_name(model.name)
        if _model_accessible(short, api_key):
            return short

    return DEFAULT_GEMINI_CANDIDATES[0]


def _filter_generate_capable(models: Iterable[object]) -> list[object]:
    """Return models that support generateContent."""

    capable: list[object] = []
    for model in models:
        name = getattr(model, "name", None)
        if not name:
            continue
        short = _short_name(name)
        if short in DISALLOWED_MODELS:
            continue
        if "generateContent" in getattr(model, "supported_generation_methods", []):
            capable.append(model)
    return capable


def _alias_map(models: Iterable[object]) -> dict[str, str]:
    """Map aliases to the short name for selection."""

    mapping: dict[str, str] = {}
    for model in models:
        name = getattr(model, "name", None)
        if not name:
            continue
        short = _short_name(name)
        for alias in _model_aliases(name):
            mapping[alias] = short
        mapping[short] = short
    return mapping


def _model_aliases(model_name: str) -> set[str]:
    """Return canonical variants for a Gemini model name."""

    short = _short_name(model_name)
    return {model_name, short}


def _short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _model_accessible(model_name: str, api_key: Optional[str]) -> bool:
    """Check if the caller can use the given Gemini model."""

    if not model_name or not api_key:
        return False

    short = _short_name(model_name)
    if short in DISALLOWED_MODELS:
        return False

    full_name = model_name if model_name.startswith("models/") else f"models/{model_name}"

    try:
        _configure_genai(api_key)
        genai.get_model(full_name)
        return True
    except GoogleAPIError:
        return False
    except Exception:
        return False


def _configure_genai(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


def _safe_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
