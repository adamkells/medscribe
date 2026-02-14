# MedGemma Models

**Module:** `medscribe.src.models.medgemma`

MedScribe uses Google's MedGemma 4B-it as its LLM backbone for clinical reasoning. Four client implementations are available, all conforming to `MedGemmaClientProtocol`.

## Client Protocol

All clients implement two methods:

```python
class MedGemmaClientProtocol(Protocol):
    def reason_over_note(
        self, note_text: str, extracted_entities: list | None = None
    ) -> dict:
        """Analyze a clinical note and return structured diagnoses, procedures, medications."""
        ...

    def reason_with_resolution(
        self, note_text: str, discrepancies: list, patient_history: str = ""
    ) -> dict:
        """Re-analyze with discrepancy context and patient history evidence."""
        ...
```

### `reason_over_note` Output Schema

```json
{
    "diagnoses": [
        {"text": "Community-acquired pneumonia", "icd10": "J18.9", "confidence": "high", "gaps": ""}
    ],
    "procedures": [
        {"text": "Chest X-ray", "cpt": "71046"}
    ],
    "medications": [
        {"text": "Azithromycin 500mg IV", "status": "started"}
    ],
    "discrepancies": []
}
```

### `reason_with_resolution` Output Schema

```json
{
    "resolved": [
        {"text": "Type 2 diabetes", "code": "E11.9", "reasoning": "Confirmed by HbA1c 7.8%", "confidence": "high"}
    ]
}
```

## Client Factory

```python
from medscribe.src.models.medgemma import create_client

client = create_client()       # uses MEDGEMMA_MODE env var
client = create_client("demo") # explicit mode
```

The factory reads the `MEDGEMMA_MODE` environment variable and returns the appropriate client:

| Mode | Client Class | Requirements |
|------|-------------|-------------|
| `demo` | `DemoMedGemmaClient` | None |
| `endpoint` | `HFEndpointMedGemmaClient` | `HF_ENDPOINT_URL`, `HF_TOKEN` |
| `vertex` | `VertexMedGemmaClient` | GCP credentials, `VERTEX_ENDPOINT_ID` |
| `real` | `MedGemmaClient` | CUDA GPU, ~8GB VRAM |
| `auto` | (auto-detect) | Tries endpoint -> GPU -> demo |

## DemoMedGemmaClient

A pure-Python mock client for development and testing. No GPU, no API keys, no network calls.

Uses keyword matching against 20 hardcoded condition patterns, 11 medication patterns, and 6 procedure patterns. Each condition has a pre-assigned ICD-10 code and confidence level.

**Selected condition mappings:**

| Pattern | ICD-10 | Confidence | Gaps |
|---------|--------|------------|------|
| `pneumonia` | J18.9 | high | |
| `hypertension` | I10 | high | |
| `diabetes` | E11.9 | medium | "Consider specifying type and complications" |
| `type 2 diabetes` | E11.9 | high | |
| `heart failure` | I50.9 | medium | "Specify systolic vs diastolic" |
| `atrial fibrillation` | I48.91 | high | |
| `acute kidney injury` | N17.9 | high | |
| `sepsis` | A41.9 | high | |
| `anemia` | D64.9 | low | "Specify type: iron deficiency, chronic disease, etc." |
| `depression` | F32.9 | medium | "Specify severity" |

Unmatched NLP entities are assigned `R69` ("Unspecified condition") to ensure every entity produces output.

The `reason_with_resolution` method returns generic resolutions with `R69` codes — sufficient for demonstrating the pipeline flow.

## HFEndpointMedGemmaClient

Production client using [HuggingFace Inference Endpoints](https://huggingface.co/docs/inference-endpoints).

```bash
export HF_ENDPOINT_URL="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud"
export HF_TOKEN="hf_..."
export MEDGEMMA_MODE=endpoint
```

**Implementation details:**

- Manually applies Gemma's chat template (`<start_of_turn>user\n...<end_of_turn>`) since HF Inference Endpoints don't support chat-format messages directly
- System messages are mapped to the user role (Gemma has no dedicated system role)
- Generation parameters: `max_new_tokens=1024`, `temperature=0.3`, `return_full_text=False`
- 300-second timeout with 3 retries on `ReadTimeout`
- JSON parsing uses a two-stage strategy: tries `json.loads()` first, then regex extraction for embedded JSON blocks

## MedGemmaClient

Local inference with 4-bit NF4 quantization via bitsandbytes.

```bash
export MEDGEMMA_MODE=real
export MEDGEMMA_MODEL=google/medgemma-4b-it  # optional, this is the default
```

**Implementation details:**

- Loads model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- Uses `device_map="auto"` via accelerate for GPU placement
- `temperature=0.3` for consistent outputs
- Requires ~8GB VRAM

## VertexMedGemmaClient

Production client using Google Cloud Vertex AI endpoints with the OpenAI-compatible API.

```bash
export VERTEX_ENDPOINT_ID="your-endpoint-id"
export VERTEX_PROJECT_ID="your-project"
export VERTEX_REGION="us-central1"  # optional, default
export MEDGEMMA_MODE=vertex
```

**Implementation details:**

- Uses `google.auth.default()` for credential management
- Auto-refreshes GCP access token before each generation call
- Connects via the OpenAI client to the Vertex AI endpoint URL

## Prompt Templates

**File:** `medscribe/src/models/prompts/reasoning.yaml`

Three YAML-defined system prompts:

### `reasoning_system`

Used by `reason_over_note()`. Instructs the LLM to extract all diagnoses, procedures, and medications with ICD-10/CPT codes, confidence levels, and documentation gaps. Requires structured JSON output.

### `resolution_system`

Used by `reason_with_resolution()`. Instructs the LLM to review discrepancies between NLP and prior LLM analysis, considering patient history evidence. Returns resolved items with final codes and reasoning.

### `triage_system`

Defined but not currently wired into the pipeline. Classifies severity for CDS alerting (`critical` / `warning` / `info`).

## JSON Response Parsing

All clients use a shared `_parse_json_response(response, fallback)` function:

1. Try `json.loads(response)` directly
2. Try regex extraction: `re.search(r"\{[\s\S]*\}", response)` to find embedded JSON
3. Return the fallback dict on failure

This handles LLMs that wrap JSON in markdown code fences or include preamble text.
