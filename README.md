# Resume Screening Agent

Boutique recruiting teams need a transparent way to triage resumes fast. This project ingests PDF/DOCX/TXT resumes, indexes them locally with Chroma, and blends deterministic scoring with Gemini reasoning to produce a ranked shortlist plus automation hooks (Sheets, Calendar, CSV exports, drop-zone watcher).

## Overview of the Agent

- **Workflow**: Recruiters paste a job description, upload resumes, and get a ranked shortlist with strengths, gaps, verdict, and next steps (Interview / Maybe / Skip).
- **Hybrid intelligence**: Keyword heuristics + similarity narrow the candidates, and optional “resume reasoning” (Gemini) adds richer analysis.
- **Local-first design**: Raw files stay on the machine; only distilled text reaches Gemini and only when reasoning is enabled.

### Architecture at a Glance

```text
Users ──▶ Streamlit UI (`src/ui/app.py`)
          │ gathers JD + resumes, toggles automations
          ▼
   ResumeRankingAgent (`src/ranking_agent.py`)
      ├─▶ Deterministic heuristics (`src/scoring.py`)
      ├─▶ Embedding search (Chroma, `storage/chroma`)
      └─▶ Optional Gemini reasoning via LangChain
          │
          ▼
   Outputs
      ├─▶ Streamlit cards + CSV download
      ├─▶ Google Sheets rows (`append_shortlist_to_sheet`)
      └─▶ Calendar holds (`create_calendar_holds`)
```

*Future n8n orchestration*: expose an API endpoint or CLI wrapper, let n8n trigger the run, and route ranked candidates to Slack/ATS. Details in **Potential Improvements**.

## Features & Limitations

| ✅ Feature | Notes |
| --- | --- |
| Multi-format ingestion | PDF, DOCX, TXT handled via `load_resumes` and normalized locally. |
| Transparent scoring | Blended score combines heuristics, embedding similarity, and optional LLM reasoning with clear thresholds. |
| Automations | One-click Google Sheets sync, Calendar holds, CSV download, CLI batch runner, drop-zone watcher. |
| Explainability | Cards list strengths, risks, matched/missing keywords, verdict rationale. |

| ⚠️ Limitation | Mitigation |
| --- | --- |
| Single-job UI | Currently handles one JD at a time; use CLI/watcher for batches. |
| Gemini dependency for deep reasoning | Heuristic-only mode exists, but LLM insights require a valid Google API key. |
| Minimal automated tests | Placeholder pytest exists; add deterministic tests around parsing/scoring before production deployments. |

## Tech Stack & APIs

- **Language/runtime**: Python 3.13
- **Frameworks**: Streamlit, LangChain, Pydantic
- **LLM**: Google Gemini via `langchain-google-genai`
- **Vector DB**: Chroma (DuckDB/Parquet on disk)
- **Automations**: Google Sheets + Calendar APIs (`google-api-python-client`)
- **Other tooling**: Watchdog (drop-zone), Pandas (exports), uvicorn/FastAPI ready if you expose API endpoints later

## Setup & Run Instructions

1. **Environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configuration**
    - Copy `.env.example` → `.env` and set:

       ```env
       GOOGLE_API_KEY=your-google-key
       CHROMA_DB_DIR=storage/chroma
       # MAX_RESUMES=100            # optional cap per run
       # GEMINI_MODEL=gemini-1.5-flash  # optional override
       ```

    - Optional automations:

       ```env
       GOOGLE_SERVICE_ACCOUNT_FILE=secrets/service-account.json
       GOOGLE_SHEETS_ID=xxx
       GOOGLE_SHEETS_TAB=Sheet1
       GOOGLE_CALENDAR_ID=xxx@group.calendar.google.com
       INTERVIEW_TIMEZONE=UTC
       INTERVIEW_START_TIME=10:00
       INTERVIEW_DURATION_MIN=30
       INTERVIEW_DAYS_OFFSET=1
       ```

3. **Streamlit UI**

   ```bash
   streamlit run src/ui/app.py
   ```

4. **Batch CLI** (`scripts/run_screening.py`)

   ```bash
   python scripts/run_screening.py --job data/job.txt --resume-dir data/resumes --top-k 5 --output output/shortlist.xlsx --use-llm --sync-sheet --create-events
   ```

5. **Drop-zone watcher** (`scripts/watch_resumes.py`)

   ```bash
   python scripts/watch_resumes.py --job data/job.txt --watch-dir dropzone --run-initial --sync-sheet
   ```

   Any new resume dropped into `dropzone/` triggers a fresh shortlist saved to `output/watch/shortlist_<timestamp>.csv` plus optional invites/syncs.

6. **Testing**

   ```bash
   pytest
   ```

   (Add task-specific tests as you extend heuristics or agents.)

## Potential Improvements

1. **Multi-job library**: Persist job descriptions + metadata, let users select a job in the UI, and keep results segregated for better analytics.
2. **n8n automation agent**:
   - Add a small FastAPI wrapper exposing `/rank` so n8n can trigger runs via webhook.
   - n8n workflow steps: receive job payload → upload files to shared storage → call `/rank` → branch into Sheets/Slack/ATS nodes → log results back to the agent.
   - Use n8n for retries, conditional routing (e.g., auto-email Interview-ready candidates, escalate risks to HRBP).
3. **Richer analytics**: Add a dashboard summarizing average fit per job, keyword coverage trends, and export history.
4. **LLM fallback**: Support additional models (OpenAI, Claude) or a local instruct model for air-gapped environments.
5. **Unit test coverage**: Formalize tests for resume parsing, heuristic scoring, and integration boundaries.

