# Getting Started

## Prerequisites

- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional) CUDA-capable GPU for local MedGemma inference
- (Optional) HuggingFace account with MedGemma access for endpoint mode

## Installation

From the HealthChain project root:

```bash
# Install HealthChain + MedScribe dependencies
cd medscribe
uv sync
```

The scispaCy biomedical model is installed automatically as a dependency:

```bash
# Verify scispaCy model is available
uv run python -c "import spacy; spacy.load('en_core_sci_sm')"
```

## Quick Start: Run the Demo

The fastest way to see MedScribe in action uses demo mode (no GPU or API keys required):

```bash
MEDGEMMA_MODE=demo uv run streamlit run medscribe/demo/app.py
```

This launches a Streamlit app at `http://localhost:8501` with:

- 4 pre-loaded clinical notes covering different specialties
- 5-step pipeline visualization showing each processing stage
- 7 result tabs: CDS Cards, NLP Entities, LLM Reasoning, Discrepancies, Agent Trace, FHIR Bundle, Metrics

Select a sample note from the sidebar dropdown and click **Run Agent**.

## Quick Start: Python API

```python
from medscribe.src.agent.orchestrator import MedScribeAgent
from medscribe.src.models.medgemma import create_client
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

# Build components
pipeline = build_coding_pipeline_simple()
client = create_client()  # uses MEDGEMMA_MODE env var

# Create agent
agent = MedScribeAgent(
    coding_pipeline=pipeline,
    medgemma_client=client,
)

# Run on a clinical note
result = agent.run(
    "Patient presents with hypertension and type 2 diabetes on metformin.",
    patient_id="Patient/123",
)

# Access results
print(f"Diagnoses: {len(result['llm_reasoning']['diagnoses'])}")
print(f"CDS Cards: {len(result['cds_cards'])}")
print(f"Tool calls: {len(result['tool_calls'])}")
print(f"FHIR Bundle entries: {len(result['fhir_bundle'].entry)}")
```

## Quick Start: Gateway Server

Run MedScribe as a production-ready API server with SOAP/CDA and CDS Hooks endpoints:

```bash
MEDGEMMA_MODE=demo uv run uvicorn medscribe.src.gateway.app:app --port 8000
```

This exposes:

- `POST /notereader/` — SOAP/CDA endpoint (Epic NoteReader compatible)
- `POST /cds/cds-services/medscribe-coding-review` — CDS Hooks endpoint
- `GET /cds/cds-discovery` — CDS Hooks service discovery
- `GET /docs` — OpenAPI interactive documentation

## Running Integration Tests

```bash
# Basic tests (no server required)
MEDGEMMA_MODE=demo uv run python medscribe/sandbox_tests/test_integration.py

# With server-based SOAP and CDS Hooks tests
MEDGEMMA_MODE=demo uv run python medscribe/sandbox_tests/test_integration.py --server
```

## Running the Benchmark

```bash
MEDGEMMA_MODE=demo uv run python -m medscribe.eval.benchmark
```

See [Evaluation](evaluation.md) for details on test cases and metrics.

## Choosing a MedGemma Mode

MedScribe supports multiple LLM backends. Set the `MEDGEMMA_MODE` environment variable:

| Mode | Requirements | Use Case |
|------|-------------|----------|
| `demo` | None | Local development, demos, CI |
| `endpoint` | `HF_ENDPOINT_URL`, `HF_TOKEN` | Production with HuggingFace Inference Endpoints |
| `vertex` | GCP credentials, `VERTEX_ENDPOINT_ID` | Production with Vertex AI |
| `real` | CUDA GPU, ~8GB VRAM | Local GPU inference with 4-bit quantization |
| `auto` | Varies | Tries endpoint, then GPU, then demo |

See [Configuration](configuration.md) for all environment variables and [Models](models.md) for client details.

## Next Steps

- [Architecture](architecture.md) — understand the 5-step pipeline design
- [Agent](agent.md) — learn about the orchestrator, validator, and tool dispatch
- [Demo](demo.md) — explore the Streamlit UI features
- [Gateway](gateway.md) — integrate with EHR systems
