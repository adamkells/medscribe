# MedScribe Agent - Project Status

**Last updated**: February 14, 2026

**Branch**: `medscribe_project` (inside HealthChain repo)

---

## Executive Summary

MedScribe Agent is an agentic clinical documentation and coding system for the MedGemma Impact Challenge. It combines MedGemma 4B (served via HuggingFace Inference Endpoints), HealthChain (EHR middleware), and an autonomous multi-step orchestrator to process clinical notes, extract and validate medical codes, and surface alerts to physicians via CDS Hooks.

As of Feb 14, the core agent pipeline, HealthChain gateway, HF Inference Endpoint client, Streamlit demo, and benchmark framework are all complete and working. Real MedGemma inference has been validated end-to-end. The next priority is implementing structured tool dispatch (giving the agent HealthChain tools for agentic FHIR queries), then video/writeup for submission.

---

## What Works Today

### Agent Pipeline (Phase 1 - Complete)
- **NLP extraction**: scispaCy `en_core_sci_sm` extracts medical entities from clinical notes via HealthChain's `MedicalCodingPipeline`
- **LLM reasoning**: `HFEndpointMedGemmaClient` calls MedGemma 4B via HF Inference Endpoints (TGI, Gemma chat template). `DemoMedGemmaClient` for offline dev.
- **Dual-pathway validation**: `DualPathwayValidator` compares NLP vs LLM findings and flags discrepancies using fuzzy text matching
- **Resolution**: Agent autonomously re-prompts MedGemma with discrepancies (and optionally patient FHIR history) to resolve conflicts
- **CDS card output**: Proper `Card` Pydantic models with `Source`, `IndicatorEnum`, and detail text per CDS Hooks spec

### Gateway Integration (Phase 2 - Complete)
- **`create_app()` factory**: Builds the full HealthChainAPI with pipeline, MedGemma client, and agent wired together
- **NoteReader (SOAP/CDA)**: `ProcessDocument` handler parses CDA XML, runs agent on extracted text, returns CDA response for Epic writeback
- **CDS Hooks (REST)**: `encounter-discharge` hook extracts note from prefetch `DocumentReference`, runs agent, returns `CDSResponse` with validated cards
- **Registered routes**: `/notereader/`, `/cds/cds-services/medscribe-coding-review`, `/health`, `/docs`

### HF Inference Endpoint (Phase 2.5 - Complete)
- **Endpoint deployed**: MedGemma 4B-it on HF Inference Endpoints (T4 GPU, TGI engine, us-east-1)
- **Client implemented**: `HFEndpointMedGemmaClient` with Gemma chat template formatting, retry logic, 300s timeout
- **Factory updated**: `create_client()` supports modes: `demo`, `endpoint`, `vertex`, `real`, `auto` (auto-detects `HF_ENDPOINT_URL`)
- **Validated end-to-end**: `reason_over_note()` returns correct ICD-10 codes (J18.9 pneumonia, E11 diabetes). `reason_with_resolution()` resolves discrepancies with clinical reasoning.
- **Latency on T4**: ~35s for reasoning, ~12s for resolution. Will improve with L4/A10G upgrade.

### Streamlit Demo (Phase 3 - Complete)
- Interactive demo with 4 sample clinical notes
- Progressive pipeline visualization (5 steps)
- 6 tabs: CDS Cards, NLP Entities, LLM Reasoning, Discrepancies, FHIR Bundle, Metrics

### Evaluation Framework (Phase 3 - Complete)
- 10 test cases across 7 clinical specialties
- Benchmark runner with F1, precision, recall, latency metrics
- Demo baseline: F1=0.727, Precision=0.622, Recall=0.900
- Real model benchmark pending GPU upgrade (T4 too slow for full 10-case run)

### Testing
- Integration tests pass in both demo and endpoint modes
- Gateway creation test passes: all routes registered correctly

### Run It

```bash
# Demo mode (no GPU or endpoint needed):
MEDGEMMA_MODE=demo uv run python medscribe/sandbox_tests/test_integration.py

# With real HF Inference Endpoint:
export $(cat .env | xargs) && MEDGEMMA_MODE=endpoint uv run python medscribe/sandbox_tests/test_integration.py

# Launch Streamlit demo:
MEDGEMMA_MODE=demo uv run streamlit run medscribe/demo/app.py

# Run benchmark (demo baseline):
MEDGEMMA_MODE=demo uv run python medscribe/eval/benchmark.py
```

---

## What's Next (Priority Order)

### 1. Structured Tool Dispatch (Phase 3a — ~8-12 hrs)
Give the agent a registry of HealthChain tools it dispatches based on discrepancy types. See spec doc section 3.5 and TASKS.md section 2.5 for full design.

| Task | Effort | Status |
|------|--------|--------|
| Create `ToolRegistry` + 7 tool handlers (`src/agent/tools.py`) | 3 hrs | TODO |
| Integrate tool dispatch into orchestrator | 3 hrs | TODO |
| Enhance FHIR bundle construction with `create_condition()` / `create_medication_statement()` | 2 hrs | TODO |
| Add "Agent Trace" tab to Streamlit demo | 1 hr | TODO |
| Update architecture diagram | 0.5 hr | TODO |

### 2. Upgrade Endpoint & Run Full Benchmark (~1 hr)
| Task | Effort | Status |
|------|--------|--------|
| Upgrade HF endpoint to L4 or A10G GPU | 15 min | TODO |
| Run full 10-case benchmark with real MedGemma | 30 min | TODO |
| Record real model F1/precision/recall numbers | 15 min | TODO |

### 3. Video & Submission (Phase 4 — Feb 17-24)
| Task | Effort | Status |
|------|--------|--------|
| Record 3-minute video demo | 1 day | TODO |
| Write 3-page Kaggle writeup | 1 day | TODO |
| Reproducible Kaggle notebook | 0.5 day | TODO |
| Final submission on Kaggle by Feb 24 | 0.5 day | TODO |

### Stretch Goals
- Fine-tune MedGemma 4B on ICD-10 coding (LoRA on MIMIC-III) for Novel Task Prize
- UMLS entity linker KB for improved NER precision
- Live hosted demo on Streamlit Cloud

---

## Architecture at a Glance

```
Clinical Note (CDA/FHIR)
        |
        v
  [NoteReader SOAP]  or  [CDS Hooks REST]
        |                       |
        v                       v
  CdaAdapter.parse()     prefetch DocumentReference
        |                       |
        +--------> note text <--+
                      |
                      v
            MedScribeAgent.run()
            ├── Step 1: NLP (scispaCy + MedicalCodingPipeline)
            ├── Step 2: LLM (MedGemma 4B via HF Inference Endpoint)
            ├── Step 3: Validate (dual-pathway comparison)
            ├── Step 4: Tool Dispatch (planned)
            │     ├── search_patient_conditions
            │     ├── search_patient_medications
            │     ├── search_patient_allergies
            │     ├── search_patient_observations
            │     └── re-prompt MedGemma with evidence
            └── Step 5: Output
                  ├── FHIR Bundle (Conditions, Procedures, MedicationStatements)
                  ├── CDS Cards (info/warning/critical)
                  └── Tool Calls trace (for explainability)
                          |
                          v
                  CDSResponse / CdaResponse
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/agent/orchestrator.py` | Core agentic loop (5-step pipeline) |
| `src/agent/validator.py` | NLP vs LLM dual-pathway comparison |
| `src/agent/gap_filler.py` | FHIR history search + relevance scoring |
| `src/agent/tools.py` | HealthChain tool registry (PLANNED) |
| `src/gateway/app.py` | HealthChainAPI with NoteReader + CDS Hooks |
| `src/pipeline/coding_pipeline.py` | scispaCy NER + UMLS entity linking |
| `src/models/medgemma.py` | MedGemma clients (HF Endpoint, local, Vertex, demo) + factory |
| `src/models/prompts/reasoning.yaml` | LLM prompt templates |
| `demo/app.py` | Streamlit interactive demo |
| `sandbox_tests/test_integration.py` | End-to-end integration tests |
| `eval/benchmark.py` | F1, latency metrics runner |
| `eval/test_cases.json` | 10 benchmark test cases (7 specialties) |

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `MEDGEMMA_MODE` | Client selection: `demo`, `endpoint`, `real`, `vertex`, `auto` | No (default: `auto`) |
| `HF_ENDPOINT_URL` | HuggingFace Inference Endpoint URL | For endpoint mode |
| `HF_TOKEN` | HuggingFace API token | For endpoint mode |
| `MEDGEMMA_MODEL` | HuggingFace model ID (default: `google/medgemma-4b-it`) | For real mode |
