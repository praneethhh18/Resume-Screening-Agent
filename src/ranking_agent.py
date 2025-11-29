"""High level orchestration for ranking resumes."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
from uuid import uuid4

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from .config import get_settings
from .embeddings import get_embedding_model
from .resume_processing import ResumeProfile, summarize_segments
from .scoring import HeuristicResult, JobProfile, build_job_profile, score_resume


class CandidateEvaluation(BaseModel):
    """Structure enforced on the LLM output for transparency."""

    fit_score: int = Field(..., ge=0, le=100, description="0-100 fit score")
    strengths: List[str] = Field(default_factory=list, description="Top strengths")
    gaps: List[str] = Field(default_factory=list, description="Key gaps or concerns")
    verdict: str = Field(..., description="Short decision statement")


@dataclass
class RankedResume:
    """Structured result for UI consumption."""

    profile: ResumeProfile
    similarity: float
    fit_score: int
    strengths: List[str]
    gaps: List[str]
    verdict: str
    heuristic_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    next_step: str
    experience_years: int | None = None
    heuristic_notes: List[str] = field(default_factory=list)

    @property
    def blended_score(self) -> float:
        """Blend vector similarity with LLM reasoning for ordering."""

        llm_component = self.fit_score / 100
        heuristic_component = self.heuristic_score / 100
        return round(
            heuristic_component * 0.5 + llm_component * 0.3 + self.similarity * 0.2,
            4,
        )


logger = logging.getLogger(__name__)


class ResumeRankingAgent:
    """Coordinates embedding similarity with Gemini-led reasoning."""

    def __init__(self, temperature: float = 0.2, require_google_key: bool = True) -> None:
        self.settings = get_settings()
        self.embeddings = get_embedding_model()
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.settings.persist_directory)
        )
        self.max_structured_attempts = 2
        self._temperature = temperature

        self.scoring_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI HR partner. Blend the provided vector similarity "
                    "score with resume evidence to judge alignment with the job description. "
                    "Be fair, concise, and bias-aware. Respond using the CandidateEvaluation schema provided.",
                ),
                (
                    "human",
                    "Job Description:\n{job_description}\n\n"
                    "Candidate: {candidate_name}\n"
                    "Resume Summary:\n{resume_summary}\n\n"
                    "Detected Keywords: {resume_keywords}\n"
                    "Vector Similarity (0-1): {similarity_score}\n",
                ),
            ]
        )

        google_api_key = self.settings.google_api_key
        if require_google_key and not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing; provide one or run heuristic-only mode.")

        if google_api_key:
            self._base_llm = ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=google_api_key,
                temperature=temperature,
                max_output_tokens=1024,
                response_mime_type="application/json",
            )
            self.llm = self._base_llm.with_structured_output(CandidateEvaluation)
            self.scoring_chain = self.scoring_prompt | self.llm
        else:
            self._base_llm = None
            self.llm = None
            self.scoring_chain = None

    def rank(
        self,
        job_description: str,
        resumes: Sequence[ResumeProfile],
        top_k: int = 5,
        use_llm: bool = True,
    ) -> List[RankedResume]:
        """Rank resumes against the provided job description."""

        job_description = job_description.strip()
        if not job_description:
            raise ValueError("Job description cannot be empty.")
        if not resumes:
            raise ValueError("You must provide at least one resume to rank.")
        if use_llm and not self.llm:
            raise RuntimeError(
                "Gemini is disabled because GOOGLE_API_KEY is missing. Toggle heuristic-only mode."
            )

        job_profile = build_job_profile(job_description)
        vector_store, profile_lookup = self._build_index(resumes)
        k = min(top_k, len(resumes))
        raw_results = vector_store.similarity_search_with_score(job_description, k=k)

        ranked: List[RankedResume] = []
        seen_ids: set[str] = set()
        for document, distance in raw_results:
            candidate_id = document.metadata["candidate_id"]
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            profile = profile_lookup[candidate_id]
            similarity = self._distance_to_similarity(distance)
            heuristic = score_resume(job_profile, profile)

            if use_llm:
                evaluation = self._evaluate_candidate(
                    job_description, profile, similarity, heuristic
                )
            else:
                evaluation = self._heuristic_to_evaluation(heuristic)
            next_step = self._derive_next_step(evaluation.fit_score)
            ranked.append(
                RankedResume(
                    profile=profile,
                    similarity=round(similarity, 3),
                    fit_score=evaluation.fit_score,
                    strengths=evaluation.strengths,
                    gaps=evaluation.gaps,
                    verdict=evaluation.verdict,
                    heuristic_score=heuristic.total_score,
                    matched_keywords=heuristic.matched_keywords,
                    missing_keywords=heuristic.missing_keywords,
                    next_step=next_step,
                    experience_years=heuristic.experience_years,
                    heuristic_notes=heuristic.notes,
                )
            )

        ranked.sort(key=lambda result: result.blended_score, reverse=True)
        return ranked

    def _build_index(
        self, resumes: Sequence[ResumeProfile]
    ) -> tuple[Chroma, Dict[str, ResumeProfile]]:
        """Create a Chroma collection from the provided resumes."""

        documents: List[Document] = []
        profile_lookup: Dict[str, ResumeProfile] = {}
        for profile in resumes:
            summary = profile.summary or summarize_segments(profile.raw_text)
            profile.summary = summary
            documents.append(
                Document(
                    page_content=summary,
                    metadata={"candidate_id": profile.candidate_id},
                )
            )
            profile_lookup[profile.candidate_id] = profile

        collection_name = f"resume-agent-{uuid4().hex[:8]}"
        vector_store = Chroma(
            client=self._chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        vector_store.add_documents(documents)
        return vector_store, profile_lookup

    def _evaluate_candidate(
        self,
        job_description: str,
        profile: ResumeProfile,
        similarity: float,
        heuristic: HeuristicResult,
    ) -> CandidateEvaluation:
        """Use Gemini to explain why a candidate fits."""

        if not self.scoring_chain or not self._base_llm:
            raise RuntimeError("Gemini reasoning is disabled for this agent instance.")

        payload = {
            "job_description": job_description,
            "candidate_name": profile.display_name,
            "resume_summary": profile.summary or summarize_segments(profile.raw_text),
            "resume_keywords": ", ".join(profile.keywords[:8]) or "not provided",
            "similarity_score": similarity,
        }

        structured_error: Exception | None = None
        for attempt in range(1, self.max_structured_attempts + 1):
            try:
                evaluation = self.scoring_chain.invoke(payload)
                if evaluation is not None:
                    return evaluation
            except Exception as exc:  # pragma: no cover - runtime guardrail
                structured_error = exc
                logger.warning(
                    "Structured Gemini call failed (attempt %s/%s): %s",
                    attempt,
                    self.max_structured_attempts,
                    exc,
                )

        logger.warning(
            "Falling back to manual parsing for %s after error: %s",
            payload.get("candidate_name"),
            structured_error,
        )
        return self._fallback_evaluation(payload, heuristic)

    def _fallback_evaluation(
        self, payload: Dict[str, object], heuristic: HeuristicResult | None
    ) -> CandidateEvaluation:
        """Retry without structured output and parse JSON manually."""

        if not self._base_llm:
            return self._default_evaluation(heuristic, reason="Gemini disabled")

        messages = self.scoring_prompt.format_messages(**payload)
        raw_response = self._base_llm.invoke(messages)
        text = self._ensure_text(raw_response)
        if not text:
            logger.error(
                "Gemini returned empty response for candidate %s",
                payload.get("candidate_name"),
            )
            return self._default_evaluation(
                heuristic,
                reason="Gemini returned empty response",
            )

        try:
            return CandidateEvaluation.model_validate_json(text)
        except Exception as err:
            logger.warning("Primary JSON parse failed: %s", err)
            try:
                cleaned = self._extract_json_block(text)
                return CandidateEvaluation.model_validate_json(cleaned)
            except Exception as final_err:
                logger.error("Unable to parse Gemini response: %s", final_err)
                return self._default_evaluation(
                    heuristic,
                    reason="Gemini JSON parsing failed",
                )

    @staticmethod
    def _derive_next_step(score: int) -> str:
        if score >= 70:
            return "Interview"
        if score >= 50:
            return "Maybe"
        return "Skip"

    def _extract_json_block(content: Any) -> str:
        """Best effort to extract a JSON object from an LLM response."""

        if callable(content):
            content = content()
        if not isinstance(content, str):
            content = str(content or "")
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return content[start : end + 1]
        raise ValueError("Unable to extract JSON from model response.")

    @staticmethod
    def _ensure_text(response: Any) -> str:
        """Normalize LangChain/Vertex responses into a plain string."""

        if response is None:
            return ""

        if isinstance(response, str):
            return response

        content = getattr(response, "content", None)
        text_segments: List[str] = []
        if isinstance(content, str):
            text_segments.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_segments.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if callable(text):
                        text = text()
                    if text:
                        text_segments.append(str(text))
                elif hasattr(part, "text"):
                    text = part.text
                    if callable(text):
                        text = text()
                    if text:
                        text_segments.append(str(text))

        if not text_segments:
            text_attr = getattr(response, "text", None)
            if callable(text_attr):
                text_attr = text_attr()
            if text_attr:
                text_segments.append(str(text_attr))

        return "\n".join(segment for segment in text_segments if segment).strip()

    @staticmethod
    def _default_evaluation(
        heuristic: HeuristicResult | None,
        reason: str,
    ) -> CandidateEvaluation:
        """Return a safe fallback evaluation when Gemini fails."""

        if heuristic:
            return ResumeRankingAgent._heuristic_to_evaluation(heuristic, reason)

        return CandidateEvaluation(
            fit_score=0,
            strengths=[],
            gaps=[reason],
            verdict="Unable to evaluate",
        )

    @staticmethod
    def _heuristic_to_evaluation(
        heuristic: HeuristicResult, reason: str | None = None
    ) -> CandidateEvaluation:
        """Convert deterministic scoring into a CandidateEvaluation shell."""

        strengths = [
            f"Matched {len(heuristic.matched_keywords)} key skills",
        ]
        strengths.extend(heuristic.notes)
        gaps: List[str] = []
        if heuristic.missing_keywords:
            gaps.append("Missing: " + ", ".join(heuristic.missing_keywords[:5]))
        if not gaps:
            gaps.append("No critical gaps detected")
        if reason:
            gaps.insert(0, reason)

        verdict = (
            "Strong match" if heuristic.total_score >= 70 else "Moderate alignment"
            if heuristic.total_score >= 45
            else "Limited fit"
        )

        return CandidateEvaluation(
            fit_score=int(round(heuristic.total_score)),
            strengths=strengths,
            gaps=gaps,
            verdict=verdict,
        )

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert Chroma distance (lower is better) to a 0-1 similarity."""

        distance = max(distance, 1e-9)
        similarity = 1.0 / (1.0 + distance)
        return max(0.0, min(1.0, similarity))

