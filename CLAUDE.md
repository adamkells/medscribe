# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedScribe Agent is an agentic clinical documentation and ICD-10 coding system built on HealthChain. It combines biomedical NLP (scispaCy), LLM reasoning (MedGemma 4B), and structured tool dispatch to automate medical coding with physician-reviewable CDS Hooks output.

## Common Commands

```bash
# Install dependencies
uv sync

# Run Streamlit demo (no GPU/keys needed)
MEDGEMMA_MODE=demo uv run streamlit run medscribe/demo/app.py

# Run integration tests
MEDGEMMA_MODE=demo uv run python medscribe/sandbox_tests/test_integration.py
MEDGEMMA_MODE=demo uv run python medscribe/sandbox_tests/test_integration.py --server

# Run benchmark evaluation (F1, precision, recall, latency)
MEDGEMMA_MODE=demo uv run python -m medscribe.eval.benchmark

# Run gateway server (SOAP/CDA + CDS Hooks endpoints)
MEDGEMMA_MODE=demo uv run uvicorn medscribe.src.gateway.app:app --port 8000

# Lint
uv run ruff check src/
```

## Architecture

### 5-Step Agentic Pipeline (`src/agent/orchestrator.py` → `MedScribeAgent.run()`)

1. **NLP Extraction** — scispaCy `en_core_sci_sm` extracts medical entities via `src/pipeline/coding_pipeline.py`
2. **LLM Reasoning** — MedGemma 4B returns structured JSON (diagnoses with ICD-10, procedures with CPT, medications) via `src/models/medgemma.py`
3. **Dual-Pathway Validation** — `src/agent/validator.py` compares NLP entities vs LLM diagnoses, detecting `llm_only` and `nlp_only` discrepancies
4. **Tool-Based Gap Resolution** — `src/agent/tools.py` dispatches FHIR search tools (conditions, medications, allergies, observations) based on discrepancy type, then re-prompts LLM with evidence
5. **Output Generation** — Constructs FHIR Bundle and CDS Hooks cards (info/warning/critical)

### LLM Client Factory

`create_client(mode)` in `src/models/medgemma.py` supports multiple backends:
- `demo` — Pure Python keyword matching (no GPU/keys needed)
- `endpoint` — HuggingFace Inference Endpoints (requires `HF_ENDPOINT_URL`, `HF_TOKEN`)
- `vertex` — Google Vertex AI (requires `VERTEX_ENDPOINT_ID`, `VERTEX_PROJECT_ID`)
- `real` — Local GPU with 4-bit quantization
- `auto` — Tries endpoint → vertex → real → demo

### Gateway Integration

`src/gateway/app.py` creates a HealthChainAPI with SOAP/CDA (Epic NoteReader) and CDS Hooks REST endpoints.

### Prompt Templates

System prompts live in `src/models/prompts/reasoning.yaml` (reasoning, resolution, triage prompts).

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MEDGEMMA_MODE` | LLM client selection (`demo`, `endpoint`, `vertex`, `real`, `auto`) |
| `HF_ENDPOINT_URL` | HuggingFace Inference Endpoint URL |
| `HF_TOKEN` | HuggingFace API token |
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint ID |
| `VERTEX_PROJECT_ID` | GCP project ID |
| `VERTEX_REGION` | GCP region (default: `us-central1`) |

## Key Design Decisions

- **Dual-pathway validation** (NLP + LLM) catches errors that either alone would miss
- **Structured tool dispatch** uses Python logic (not LLM-chosen) to select tools based on discrepancy type
- Tool registry falls back to mock data pools when no FHIR gateway is provided
- Project uses `uv` as the package manager with `pyproject.toml` (setuptools-based, Python 3.10-3.11)
