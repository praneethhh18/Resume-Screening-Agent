"""Utilities for reading, cleaning, and summarizing resumes."""
from __future__ import annotations

import io
import logging
import re
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import docx2txt
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "for",
    "from",
    "your",
    "their",
    "skills",
    "experience",
    "summary",
    "professional",
}


@dataclass
class ResumeProfile:
    """Represents a parsed resume that can be ranked later."""

    candidate_id: str
    filename: str
    display_name: str
    raw_text: str
    keywords: List[str]
    summary: str | None = None
    email: str | None = None
    links: List[str] = field(default_factory=list)

    def short_preview(self, max_chars: int = 360) -> str:
        """Return a truncated preview string for UI cards."""

        snippet = self.raw_text.strip().replace("\n", " ")
        return (snippet[: max_chars - 3] + "...") if len(snippet) > max_chars else snippet


def load_resumes(files: Iterable[object]) -> List[ResumeProfile]:
    """Read uploaded files into structured resume profiles."""

    profiles: List[ResumeProfile] = []
    for index, file_like in enumerate(files, start=1):
        if file_like is None:
            continue
        filename = getattr(file_like, "name", f"resume-{index}")
        suffix = Path(filename).suffix.lower()
        file_bytes = _read_bytes(file_like)
        profile = _create_profile(filename, suffix, file_bytes)
        profiles.append(profile)

    return profiles


def load_resumes_from_paths(paths: Sequence[Path]) -> List[ResumeProfile]:
    """Load resumes directly from filesystem paths for automation scripts."""

    profiles: List[ResumeProfile] = []
    for path in paths:
        suffix = path.suffix.lower()
        profile = _create_profile(path.name, suffix, path.read_bytes())
        profiles.append(profile)

    return profiles


def _read_bytes(file_like: object) -> bytes:
    """Return file content as bytes regardless of the input type."""

    if hasattr(file_like, "getvalue"):
        data = file_like.getvalue()
    elif hasattr(file_like, "read"):
        data = file_like.read()
    else:
        raise ValueError("Unsupported file object provided")

    if hasattr(file_like, "seek"):
        file_like.seek(0)
    return data


def _extract_text(data: bytes, suffix: str) -> str:
    """Extract plain text from PDF/DOCX/TXT sources."""

    if suffix == ".pdf":
        return _extract_pdf_text(data)

    if suffix == ".docx":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
            tmp.write(data)
            tmp.flush()
            return docx2txt.process(tmp.name)

    return data.decode("utf-8", errors="ignore")


def _extract_pdf_text(data: bytes) -> str:
    """Use PyPDF first, then fall back to pdfminer when needed."""

    buffer = io.BytesIO(data)
    reader = PdfReader(buffer)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if _has_meaningful_text(text):
        return text

    buffer.seek(0)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text

        text = pdfminer_extract_text(buffer)
        if _has_meaningful_text(text):
            return text
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("pdfminer fallback failed: %s", exc)

    buffer.seek(0)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while keeping paragraph breaks."""

    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _guess_display_name(filename: str, text: str) -> str:
    """Best-effort attempt to infer a candidate name."""

    first_non_empty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if 2 <= len(first_non_empty.split()) <= 6:
        return first_non_empty.title()

    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return stem.title()


def _extract_keywords(text: str, top_k: int = 8) -> List[str]:
    """Return a lightweight list of prominent keywords for UX flair."""

    tokens = re.findall(r"[A-Za-z][A-Za-z+.#-]{2,}", text)
    counter = Counter(token.lower() for token in tokens if token.lower() not in STOPWORDS)
    most_common = [word for word, _ in counter.most_common(top_k)]
    return [word.replace("#", "").replace("+", " ").title() for word in most_common]


EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
SPECIAL_DOMAINS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "linkedin.com",
    "behance.net",
    "dribbble.com",
    "notion.site",
    "notion.so",
    "kaggle.com",
    "devpost.com",
    "medium.com",
    "hashnode.dev",
    "angel.co",
    "portfolio.site",
)
SPECIAL_DOMAIN_REGEX = re.compile(
    r"(?:https?://)?(?:"
    + r"|".join(re.escape(domain) for domain in SPECIAL_DOMAINS)
    + r")[^\s<>]*",
    re.IGNORECASE,
)


def _extract_email(text: str) -> str | None:
    """Detect the first reasonable email address inside the resume."""

    matches = EMAIL_REGEX.findall(text)
    if not matches:
        return None
    return matches[0].lower()


def _extract_links(text: str, max_links: int = 5) -> List[str]:
    """Return distinct external links (portfolio, LinkedIn, GitHub, etc.)."""

    candidates = URL_REGEX.findall(text)
    candidates.extend(SPECIAL_DOMAIN_REGEX.findall(text))

    cleaned: List[str] = []
    seen = set()
    for link in candidates:
        normalized = _normalize_link(link)
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
        if len(cleaned) >= max_links:
            break
    return cleaned


def _normalize_link(raw: str) -> str:
    link = raw.strip().strip(",.;)>")
    if not link:
        return ""
    if not re.match(r"^[a-z]+://", link, re.IGNORECASE):
        link = "https://" + link.lstrip("/")
    return link


def _has_meaningful_text(text: str) -> bool:
    return bool(text and re.search(r"[A-Za-z0-9]", text))


def _create_profile(filename: str, suffix: str, file_bytes: bytes) -> ResumeProfile:
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    raw_text = _extract_text(file_bytes, suffix)
    cleaned_text = _normalize_whitespace(raw_text)
    display_name = _guess_display_name(filename, cleaned_text)
    keywords = _extract_keywords(cleaned_text)
    email = _extract_email(cleaned_text)
    links = _extract_links(cleaned_text)

    return ResumeProfile(
        candidate_id=str(uuid.uuid4()),
        filename=filename,
        display_name=display_name,
        raw_text=cleaned_text,
        keywords=keywords,
        email=email,
        links=links,
    )


def summarize_segments(text: str, max_chars: int = 1500) -> str:
    """Create a deterministic short summary used for quick previews."""

    parts = [segment.strip() for segment in text.splitlines() if segment.strip()]
    summary = " ".join(parts)
    return (summary[: max_chars - 3] + "...") if len(summary) > max_chars else summary


def derive_job_label(job_description: str, fallback: str = "Ad-hoc role") -> str:
    """Infer a human-friendly job label from the job description text."""

    for line in job_description.splitlines():
        line = line.strip()
        if line:
            words = line.split()
            short = " ".join(words[:8])
            return short[:60]
    return fallback
