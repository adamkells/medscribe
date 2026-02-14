# Configuration

## Environment Variables

### LLM Client Selection

| Variable | Purpose | Values | Default |
|----------|---------|--------|---------|
| `MEDGEMMA_MODE` | Selects the MedGemma client implementation | `demo`, `endpoint`, `vertex`, `real`, `auto` | `auto` |

**Mode behavior:**

- `demo` — Pure Python keyword matching. No GPU, no API keys, no network. Best for development and CI.
- `endpoint` — HuggingFace Inference Endpoints. Requires `HF_ENDPOINT_URL` and `HF_TOKEN`.
- `vertex` — Google Cloud Vertex AI. Requires GCP credentials and `VERTEX_ENDPOINT_ID`.
- `real` — Local GPU inference with 4-bit quantization. Requires CUDA.
- `auto` — Tries endpoint (if `HF_ENDPOINT_URL` set), then GPU (if CUDA available), then demo.

### HuggingFace Inference Endpoints

| Variable | Purpose | Required for |
|----------|---------|-------------|
| `HF_ENDPOINT_URL` | Full URL of the HuggingFace Inference Endpoint | `endpoint` mode |
| `HF_TOKEN` | HuggingFace API token with endpoint access | `endpoint` mode |

### Vertex AI

| Variable | Purpose | Default |
|----------|---------|---------|
| `VERTEX_ENDPOINT_ID` | Vertex AI endpoint identifier | (required) |
| `VERTEX_PROJECT_ID` | GCP project ID | (required) |
| `VERTEX_REGION` | GCP region | `us-central1` |

Vertex AI authentication uses `google.auth.default()` — configure via `GOOGLE_APPLICATION_CREDENTIALS` or default GCP credentials.

### Local GPU Inference

| Variable | Purpose | Default |
|----------|---------|---------|
| `MEDGEMMA_MODEL` | HuggingFace model ID | `google/medgemma-4b-it` |

Requires CUDA-capable GPU with ~8GB VRAM. The model is loaded with 4-bit NF4 quantization.

## Prompt Templates

**File:** `medscribe/src/models/prompts/reasoning.yaml`

System prompts are loaded from YAML at import time. The file contains three templates:

### `reasoning_system`

Used by `reason_over_note()` (Step 2). Instructs the LLM to:
- Extract all diagnoses, procedures, and medications
- Assign ICD-10 or CPT codes with confidence levels
- Flag documentation gaps
- Cross-reference with NLP-extracted entities
- Return structured JSON

### `resolution_system`

Used by `reason_with_resolution()` (Step 4). Instructs the LLM to:
- Review discrepancies between NLP and prior LLM analysis
- Consider patient history evidence (from tool dispatch)
- Determine correct coding with reasoning
- Return resolved items as structured JSON

### `triage_system`

Defined but not currently active. Intended for severity classification of CDS alerts.

## Pipeline Configuration

### scispaCy Model

The default model is `en_core_sci_sm` (small biomedical NER). To use a different model:

```python
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

pipeline = build_coding_pipeline_simple(model_name="en_core_sci_lg")
```

Available scispaCy models: `en_core_sci_sm`, `en_core_sci_md`, `en_core_sci_lg`, `en_core_sci_scibert`.

### UMLS Entity Linking

For the full pipeline with CUI codes and automatic FHIR problem list extraction:

```python
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline

pipeline = build_coding_pipeline(
    linker_name="umls",              # or "mesh", "rxnorm", "go", "hpo"
    resolve_abbreviations=True,
    min_linking_score=0.7,
)
```

The UMLS knowledge base (~3 GB) downloads on first use.

## Agent Configuration

### MedScribeAgent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coding_pipeline` | `MedicalCodingPipeline` | (required) | scispaCy NER pipeline |
| `medgemma_client` | `MedGemmaClientProtocol` | (required) | LLM inference client |
| `fhir_gateway` | `FHIRGateway` or `None` | `None` | Live FHIR server connection |
| `fhir_source` | `str` | `"epic"` | Named source for FHIR gateway searches |
| `confidence_threshold` | `float` | `0.7` | Low-confidence threshold |

### ToolRegistry Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fhir_gateway` | `FHIRGateway` or `None` | `None` | Enables real FHIR searches; mock data when `None` |
| `fhir_source` | `str` | `"epic"` | Named source for gateway searches |

### DualPathwayValidator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | `float` | `0.8` | Reserved for future use (current matching uses substring containment) |

## Dependencies

Defined in `medscribe/pyproject.toml`:

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | >=4.50.0 | HuggingFace model loading |
| `bitsandbytes` | >=0.45.0 | 4-bit quantization (local GPU) |
| `accelerate` | >=0.34.0 | Device mapping for quantized models |
| `scispacy` | >=0.5.4 | Biomedical NER |
| `pyyaml` | >=6.0 | Prompt template loading |
| `streamlit` | >=1.38.0 | Demo UI |
| `en_core_sci_sm` | 0.5.4 | scispaCy biomedical model (from S3) |

### Development

| Package | Purpose |
|---------|---------|
| `jupyter` | Notebook development |
| `ipykernel` | Jupyter kernel |
| `ruff` | Linting and formatting |
