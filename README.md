# MedScribe Agent

Agentic clinical documentation and ICD-10 coding system built on [HealthChain](https://github.com/dotimplement/HealthChain). Combines biomedical NLP (scispaCy), LLM reasoning (MedGemma 4B), and structured tool dispatch to automate medical coding with physician-reviewable CDS Hooks output.

Submitted to the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

## How It Works

MedScribe runs a 5-step agentic pipeline on clinical notes:

1. **NLP Extraction** — scispaCy extracts medical entities from the note
2. **LLM Reasoning** — MedGemma 4B identifies diagnoses with ICD-10 codes, procedures (CPT), and medications
3. **Dual-Pathway Validation** — compares NLP vs LLM findings and flags discrepancies
4. **Tool-Based Gap Resolution** — dispatches FHIR search tools based on discrepancy type, then re-prompts MedGemma with evidence
5. **Output Generation** — produces a FHIR Bundle and CDS Hooks cards for physician review

The dual-pathway design catches errors that either NLP or LLM alone would miss.

```
Clinical Note
     │
     ▼
HealthChain Gateway (NoteReader SOAP/CDA or CDS Hooks REST)
     │
     ├─► Step 1: NLP (scispaCy) ──────────────────────┐
     │                                                  ▼
     └─► Step 2: MedGemma LLM ──────────► Step 3: Dual-Pathway Validation
                                                        │
                                          Discrepancies?├─ No ──► Step 5: Output
                                                        │
                                                       Yes
                                                        │
                                          ┌─────────────▼─────────────┐
                                          │ Step 4: Tool Dispatch      │
                                          │  search_patient_conditions │
                                          │  search_patient_medications│
                                          │  search_patient_allergies  │
                                          │  → re-prompt MedGemma      │
                                          └─────────────┬─────────────┘
                                                        │
                                                        ▼
                                          Step 5: FHIR Bundle + CDS Cards
```

## Benchmark Results

Evaluated on 6 clinical scenarios (inpatient, cardiology, critical care, behavioral health, single-condition, edge case):

| Metric | Demo baseline | MedGemma 4B (HF Endpoint, T4) |
|--------|--------------|-------------------------------|
| F1 | 0.727 | **0.694** |
| Precision | 0.622 | **0.694** |
| Recall | 0.900 | **0.694** |
| Latency | ~24ms/note | ~39s/note |

## Quickstart

```bash
# Install dependencies
uv sync
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Run Streamlit demo (no GPU or API keys needed)
MEDGEMMA_MODE=demo uv run streamlit run demo/app.py

# Run with real MedGemma via HuggingFace Inference Endpoint
export HF_ENDPOINT_URL="https://your-endpoint.aws.endpoints.huggingface.cloud"
export HF_TOKEN="hf_..."
MEDGEMMA_MODE=endpoint uv run streamlit run demo/app.py

# Run integration tests
MEDGEMMA_MODE=demo uv run python sandbox_tests/test_integration.py

# Run benchmark
MEDGEMMA_MODE=demo uv run python eval/benchmark.py
```

## MedGemma Backends

Set `MEDGEMMA_MODE` to select the inference backend:

| Mode | Description | Requirements |
|------|-------------|--------------|
| `demo` | Keyword-based mock client | None |
| `endpoint` | HuggingFace Inference Endpoints | `HF_ENDPOINT_URL`, `HF_TOKEN` |
| `vertex` | Google Vertex AI | `VERTEX_ENDPOINT_ID`, `VERTEX_PROJECT_ID` |
| `real` | Local GPU (4-bit quantized) | CUDA GPU, HF model access |
| `auto` | Auto-detects available backend | — |

## Project Structure

```
medscribe/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py      # MedScribeAgent — 5-step agentic loop
│   │   ├── validator.py         # DualPathwayValidator — NLP vs LLM comparison
│   │   ├── tools.py             # ToolRegistry — FHIR search tools
│   │   └── gap_filler.py        # FHIR history search + relevance scoring
│   ├── models/
│   │   ├── medgemma.py          # MedGemma clients + create_client() factory
│   │   └── prompts/
│   │       └── reasoning.yaml   # System prompt templates
│   ├── pipeline/
│   │   └── coding_pipeline.py   # scispaCy NER pipeline
│   └── gateway/
│       └── app.py               # HealthChain gateway (NoteReader + CDS Hooks)
├── demo/
│   └── app.py                   # Streamlit interactive demo
├── eval/
│   ├── benchmark.py             # F1, precision, recall, latency runner
│   └── test_cases.json          # 6 benchmark test cases
├── notebooks/
│   └── 02_kaggle_medgemma.ipynb # Kaggle reproducibility notebook
└── sandbox_tests/
    └── test_integration.py      # End-to-end integration tests
```

## Requirements

- Python 3.10–3.11
- [uv](https://docs.astral.sh/uv/) package manager
- For local GPU inference: CUDA-compatible GPU with 8GB+ VRAM
- For endpoint inference: HuggingFace account with MedGemma license accepted

## Built With

- [HealthChain](https://github.com/dotimplement/HealthChain) — EHR middleware (FHIR, CDS Hooks, NoteReader)
- [MedGemma 4B](https://huggingface.co/google/medgemma-4b-it) — Google's medical-domain LLM
- [scispaCy](https://allenai.github.io/scispacy/) — Biomedical NLP
- [Streamlit](https://streamlit.io/) — Interactive demo
