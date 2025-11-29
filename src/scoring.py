"""Deterministic resume scoring utilities to complement LLM reasoning."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from .resume_processing import ResumeProfile, STOPWORDS


@dataclass
class JobProfile:
    """Structured representation of a job description for heuristics."""

    keywords: List[str]
    keyword_set: Set[str]
    required_years: int | None


@dataclass
class HeuristicResult:
    """Outcome of deterministic resume scoring."""

    total_score: float
    keyword_match_ratio: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    experience_years: int | None
    experience_score: float
    notes: List[str]


def build_job_profile(job_description: str, top_k: int = 24) -> JobProfile:
    """Extract high-signal keywords and experience hints from a job description."""

    tokens = _tokenize(job_description)
    counter = Counter(token for token in tokens if token not in STOPWORDS)
    keywords = [word for word, _ in counter.most_common(top_k)]
    required_years = _detect_years(job_description)
    return JobProfile(keywords=keywords, keyword_set=set(keywords), required_years=required_years)


def score_resume(job: JobProfile, resume: ResumeProfile) -> HeuristicResult:
    """Return a deterministic score for how well the resume matches the job."""

    resume_tokens = set(_tokenize(resume.raw_text))
    matched = sorted(job.keyword_set.intersection(resume_tokens))
    missing = sorted(word for word in job.keyword_set if word not in matched)

    coverage = len(matched) / len(job.keyword_set) if job.keyword_set else 0.0
    keyword_score = coverage * 80.0
    if coverage >= 0.85:
        keyword_score += 10.0
    elif coverage >= 0.7:
        keyword_score += 5.0

    resume_years = _detect_years(resume.raw_text)
    experience_score = _score_experience(job.required_years, resume_years)

    notes: List[str] = []
    if job.keyword_set:
        notes.append(f"Matched {len(matched)} of {len(job.keyword_set)} target keywords")
    if job.required_years:
        if resume_years is None:
            notes.append(f"Job requests {job.required_years}+ yrs; resume missing explicit years")
        else:
            notes.append(
                f"Job requires {job.required_years}+ yrs; resume references {resume_years} yrs"
            )
    else:
        if resume_years:
            notes.append(f"Resume references {resume_years} years of experience")

    total_score = min(100.0, keyword_score + experience_score)
    return HeuristicResult(
        total_score=total_score,
        keyword_match_ratio=coverage,
        matched_keywords=matched,
        missing_keywords=missing,
        experience_years=resume_years,
        experience_score=experience_score,
        notes=notes,
    )


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{2,}", text)]


def _detect_years(text: str) -> int | None:
    """Parse patterns like '7+ years' to extract a rough experience count."""

    matches = re.findall(r"(\d{1,2})\s*\+?\s*years", text, flags=re.IGNORECASE)
    if not matches:
        return None
    numeric = [int(value) for value in matches]
    return int(max(numeric))


def _score_experience(required: int | None, resume_years: int | None) -> float:
    if resume_years is None and required is None:
        return 10.0
    if resume_years is None:
        return 5.0 if required else 0.0
    if required is None:
        return min(30.0, resume_years * 2.0)

    if resume_years >= required:
        return 30.0
    if resume_years >= required * 0.75:
        return 20.0
    if resume_years >= required * 0.5:
        return 10.0
    return 0.0
