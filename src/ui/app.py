"""Streamlit entry point for the resume screening agent."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import Settings, get_settings
from src.integrations.google_sync import (
    GoogleIntegrationError,
    append_shortlist_to_sheet,
    create_calendar_holds,
)
from src.ranking_agent import RankedResume, ResumeRankingAgent
from src.resume_processing import derive_job_label, load_resumes


@st.cache_resource(show_spinner=False)
def _get_agent(require_llm: bool) -> ResumeRankingAgent:
    """Cache the resume agent to avoid re-loading Gemini + embeddings."""

    return ResumeRankingAgent(require_google_key=require_llm)


def run() -> None:
    """Render the interactive resume screening experience."""

    st.set_page_config(page_title="Resume Screening Agent", page_icon="🧠", layout="wide")
    _apply_theme()
    settings = get_settings()
    sheet_ready = bool(settings.google_service_account_file and settings.google_sheets_id)
    calendar_ready = bool(settings.google_service_account_file and settings.google_calendar_id)

    st.title("🧠 Resume Intelligence Hub")
    st.caption("Rank resumes with transparent heuristics and optional reasoning context")

    with st.sidebar:
        st.header("Screening Controls")
        top_k = st.slider("Top candidates to highlight", min_value=3, max_value=15, value=5)
        show_similarity = st.toggle("Show similarity score", value=True)
        use_llm = st.toggle(
            "Use resume reasoning",
            value=False,
            help="Adds detailed resume insights for each candidate",
        )

        st.subheader("Automation")
        sync_to_sheet = st.toggle(
            "Sync to Google Sheet",
            value=False,
            disabled=not sheet_ready,
            help="Append shortlist rows to the configured sheet.",
        )
        create_events = st.toggle(
            "Create Calendar holds",
            value=False,
            disabled=not calendar_ready,
            help="Drop tentative events for Interview-ready candidates.",
        )
        st.caption("all processingg stays local, you resume Intelligence hub is active with automations")

    st.markdown("### Job Inputs")
    col_desc, col_upload = st.columns((1.6, 1))
    with col_desc:
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste or type the responsibilities, required skills, and nice-to-haves...",
            height=260,
        )
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Files stay on your machine; only distilled text is sent to Gemini for reasoning.",
        )

    if not settings.google_api_key and use_llm:
        st.warning(
            "Add your GOOGLE_API_KEY to the .env file or disable Gemini reasoning in the sidebar."
        )
        return

    if st.button("Run Screening", type="primary", width="stretch"):
        if not job_description.strip():
            st.error("Please provide a job description first.")
            return
        if not uploaded_files:
            st.error("Upload at least one resume to begin.")
            return

        limited_files = uploaded_files
        if settings.max_resumes is not None:
            limited_files = uploaded_files[: settings.max_resumes]
            if len(uploaded_files) > settings.max_resumes:
                st.info(
                    f"Only the first {settings.max_resumes} resumes are processed in this run. Increase MAX_RESUMES to change."
                )

        with st.spinner("Crunching embeddings, scoring candidates, and writing feedback..."):
            try:
                profiles = load_resumes(limited_files)
                agent = _get_agent(use_llm)
                rankings = agent.rank(
                    job_description,
                    profiles,
                    top_k=top_k,
                    use_llm=use_llm,
                )
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                st.error(f"Screening failed: {exc}")
                st.stop()

        if not rankings:
            st.warning("No candidates could be evaluated.")
            return

        job_name = derive_job_label(job_description)
        _run_integrations(
            rankings,
            job_name,
            settings,
            sync_sheet=sync_to_sheet,
            create_events=create_events,
        )
        _render_results(rankings, show_similarity, show_heuristics=True, job_name=job_name)


def _run_integrations(
    rankings: List[RankedResume],
    job_name: str,
    settings: Settings,
    *,
    sync_sheet: bool,
    create_events: bool,
) -> None:
    if not rankings:
        return

    if sync_sheet:
        try:
            result = append_shortlist_to_sheet(settings, job_name, rankings)
            st.success(f"Synced {result.count} candidate(s) to Google Sheets.")
        except GoogleIntegrationError as exc:
            st.warning(f"Google Sheets sync failed: {exc}")

    if create_events:
        try:
            result = create_calendar_holds(settings, job_name, rankings)
            if result.count:
                st.success(f"Created {result.count} calendar hold(s) for Interview-ready candidates.")
            else:
                st.info("No Interview-ready candidates, so no calendar events were scheduled.")
        except GoogleIntegrationError as exc:
            st.warning(f"Calendar automation failed: {exc}")


def _render_results(
    results: List[RankedResume],
    show_similarity: bool,
    show_heuristics: bool,
    *,
    job_name: str,
) -> None:
    """Visualize agent decisions with Streamlit widgets."""

    leaderboard_rows = []
    st.subheader("Shortlist")
    run_timestamp = datetime.utcnow().isoformat(timespec="seconds")
    st.caption(f"{job_name} · generated {run_timestamp} UTC")
    st.divider()
    for idx, result in enumerate(results, start=1):
        with st.container():
            st.markdown("<div class='shortlist-card'>", unsafe_allow_html=True)
            header = f"{idx}. {result.profile.display_name}"
            if show_similarity:
                header += f" · sim {result.similarity:.2f}"
            badge = _next_step_badge(result.next_step)
            st.markdown(f"<div class='card-header'><h3>{header}</h3>{badge}</div>", unsafe_allow_html=True)
            contact_bits = []
            if result.profile.email:
                contact_bits.append(f"📧 {result.profile.email}")
            if result.profile.links:
                link_labels = [
                    f"[{_link_label(link)}]({link})" for link in result.profile.links[:3]
                ]
                contact_bits.append("🔗 " + ", ".join(link_labels))
            if contact_bits:
                st.caption(" | ".join(contact_bits))
            st.progress(result.fit_score / 100, text=f"Fit score {result.fit_score}/100")

            col_left, col_right = st.columns([3, 1])
            col_left.markdown(f"**Verdict:** {result.verdict}")
            col_right.metric("Blended score", f"{result.blended_score:.2f}")
            if show_heuristics:
                col_right.caption(
                    f"Heuristic score {result.heuristic_score:.1f}\n"
                    f"Matched: {len(result.matched_keywords)}"
                )

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Top strengths**")
                for point in result.strengths or ["Not specified"]:
                    st.write(f"• {point}")
            with cols[1]:
                st.markdown("**Risks / gaps**")
                for point in result.gaps or ["None detected"]:
                    st.write(f"• {point}")

            keyword_line = ", ".join(result.matched_keywords[:8]) or "n/a"
            missing_line = ", ".join(result.missing_keywords[:5]) or "none"
            st.caption(
                f"Heuristic keywords ➜ matched: {keyword_line} | missing: {missing_line}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        leaderboard_rows.append(
            {
                "rank": idx,
                "candidate": result.profile.display_name,
                "next_step": result.next_step,
                "fit_score": result.fit_score,
                "similarity": result.similarity,
                "blended_score": result.blended_score,
                "verdict": result.verdict,
                "generated_at_utc": run_timestamp,
            }
        )

    df = pd.DataFrame(leaderboard_rows)
    st.subheader("Reports")
    tabs = st.tabs(["Leaderboard", "Raw data"])
    with tabs[0]:
        st.dataframe(df, width="stretch", hide_index=True)
    with tabs[1]:
        st.dataframe(df, width="stretch")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv_bytes,
            file_name="resume_ranking.csv",
            mime="text/csv",
            type="primary",
        )


def _link_label(link: str) -> str:
    parsed = urlparse(link)
    host = parsed.netloc or link
    host = host.replace("www.", "")
    return host or link


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .stApp {background-color: #050714; color: #f5f7ff; font-family: 'Inter', sans-serif;}
        section[data-testid="stSidebar"] {background: #0d142a; border-right: 1px solid rgba(255,255,255,0.05);}
        .stButton>button {border-radius: 999px; font-weight: 600; padding: 0.65rem 1.8rem;}
        .stTabs [data-baseweb="tab"] {font-weight: 600;}
        .stProgress > div > div {background: linear-gradient(90deg,#7c4dff,#5ad1ff);}
        .shortlist-card {background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 1.2rem; border-radius: 18px; margin-bottom: 1rem;}
        .card-header {display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;}
        .next-step {padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600;}
        .next-step-Interview {background: rgba(52,199,89,0.2); color: #8ef5b9;}
        .next-step-Maybe {background: rgba(255,193,7,0.2); color: #ffe08a;}
        .next-step-Skip {background: rgba(255,99,132,0.2); color: #ffb0c1;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _next_step_badge(label: str) -> str:
    safe = label.replace(" ", "-")
    return f"<span class='next-step next-step-{safe}'>{label}</span>"


if __name__ == "__main__":
    run()
