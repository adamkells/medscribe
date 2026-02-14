# Agent

The agent module (`medscribe.src.agent`) contains the core orchestration logic: the `MedScribeAgent` orchestrator, `DualPathwayValidator`, and `ToolRegistry` for structured tool dispatch.

## MedScribeAgent

**Module:** `medscribe.src.agent.orchestrator`

The central orchestrator that runs the 5-step agentic pipeline.

### Constructor

```python
MedScribeAgent(
    coding_pipeline: MedicalCodingPipeline,  # scispaCy NER pipeline
    medgemma_client: MedGemmaClientProtocol,  # LLM inference client
    fhir_gateway=None,                        # optional live FHIR connection
    fhir_source: str = "epic",                # named FHIR source
    confidence_threshold: float = 0.7,
)
```

The agent internally creates a `DualPathwayValidator` and a `ToolRegistry` (passing through the `fhir_gateway` and `fhir_source` parameters).

### `run(clinical_note, patient_id) -> dict`

Executes the full pipeline and returns a result dictionary:

```python
{
    "fhir_bundle": Bundle,           # FHIR Bundle with Condition + MedicationStatement resources
    "cds_cards": list[Card],         # CDS Hooks cards for physician review
    "nlp_entities": list[dict],      # Raw NLP entities from scispaCy
    "llm_reasoning": dict,           # LLM structured output (diagnoses, procedures, medications)
    "resolved": dict,                # Final resolved result after gap filling
    "discrepancies": list[dict],     # NLP vs LLM discrepancies
    "tool_calls": list[dict],        # Serialized ToolResult records from tool dispatch
}
```

### Pipeline Steps

**Step 1 — NLP Extraction:** Wraps the clinical note in a `healthchain.io.Document`, runs it through the `MedicalCodingPipeline` (scispaCy NER), and extracts entities via `doc.nlp.get_entities()`.

**Step 2 — LLM Reasoning:** Calls `medgemma.reason_over_note(note, entities)` to get structured diagnoses (with ICD-10 codes, confidence, and documentation gaps), procedures (with CPT codes), and medications.

**Step 3 — Dual-Pathway Validation:** Calls `validator.compare(nlp_entities, llm_result)` to find discrepancies between the two pathways. Also checks for any low-confidence diagnoses.

**Step 4 — Tool-Based Gap Resolution:** If discrepancies or low-confidence items exist, calls `_resolve_with_tools()`. This is the agentic core — see [Tool Dispatch](#tool-dispatch) below.

**Step 5 — Output Generation:** Builds a FHIR Bundle using HealthChain utilities (`create_condition`, `create_medication_statement`, `create_bundle`, `add_resource`) and CDS Hooks cards with severity indicators.

### FHIR Bundle Construction

The `_build_fhir_bundle()` method creates a proper FHIR `collection` Bundle:

- Each diagnosis becomes a `Condition` resource (ICD-10-CM coded, `clinical_status="active"`)
- Each medication becomes a `MedicationStatement` resource (`status="recorded"`)
- All resources are linked to the patient via the `subject` reference

### CDS Card Generation

The `_build_cds_cards()` method creates one CDS Hooks card per diagnosis:

| Indicator | Condition |
|-----------|-----------|
| `warning` | Diagnosis has documentation gaps OR low confidence |
| `info` | Coding validated, no concerns |

Each card includes the condition text, ICD-10 code, and any resolution reasoning.

---

## DualPathwayValidator

**Module:** `medscribe.src.agent.validator`

Compares NLP-extracted entities against LLM-generated diagnoses to find disagreements.

### `compare(nlp_entities, llm_result) -> list[dict]`

Returns a list of discrepancy dictionaries. Two types:

**`llm_only`** — LLM found a diagnosis that NLP did not detect:

```python
{
    "type": "llm_only",
    "entity": "Community-acquired pneumonia",
    "icd10": "J18.9",
    "confidence": "high",
    "reason": "LLM diagnosis not found in NLP entities"
}
```

**`nlp_only`** — NLP detected an entity that the LLM did not include in diagnoses:

```python
{
    "type": "nlp_only",
    "entity": "metformin",
    "code": "C0025598",
    "reason": "NLP entity not confirmed by LLM"
}
```

### Matching Logic

The validator uses substring containment for fuzzy matching: entity A matches entity B if `a in b` or `b in a` (case-insensitive). This handles common variations like "diabetes" matching "type 2 diabetes mellitus".

---

## Tool Dispatch

**Module:** `medscribe.src.agent.tools`

The `ToolRegistry` provides structured, deterministic tool dispatch — the agent selects tools based on Python keyword classifiers, not LLM-generated tool calls.

### ToolResult

A dataclass representing the outcome of a tool execution:

```python
@dataclass
class ToolResult:
    tool_name: str        # e.g. "search_patient_conditions"
    entity: str           # the entity being investigated
    reason: str           # why this tool was called
    success: bool         # whether evidence was found
    data: list[dict]      # raw result data
    summary: str          # human-readable summary
```

### ToolRegistry

```python
registry = ToolRegistry(
    fhir_gateway=None,      # optional live FHIR gateway
    fhir_source="epic",     # named FHIR source for gateway searches
)
```

#### Available Tools

**Evidence tools** (search patient records):

| Tool Name | FHIR Resource | Mock Data Pool |
|-----------|--------------|----------------|
| `search_patient_conditions` | `Condition` | Hypertension, T2DM, COPD, CHF, AKI |
| `search_patient_medications` | `MedicationStatement` | Metformin, lisinopril, metoprolol, aspirin |
| `search_patient_allergies` | `AllergyIntolerance` | Penicillin, sulfa drugs |
| `search_patient_observations` | `Observation` | HbA1c, glucose, WBC, creatinine, BNP |
| `search_patient_procedures` | `Procedure` | Chest X-ray, echocardiogram, blood culture |

**Construction tools** (create FHIR resources):

| Tool Name | Creates |
|-----------|---------|
| `create_fhir_condition` | FHIR Condition via `healthchain.fhir.create_condition()` |
| `create_fhir_medication` | FHIR MedicationStatement via `healthchain.fhir.create_medication_statement()` |

#### Execution

```python
result = registry.execute("search_patient_conditions", entity="diabetes", patient_id="Patient/123")
# result.success = True
# result.summary = "Found 1 matching condition(s): Type 2 diabetes mellitus"
# result.data = [{"display": "Type 2 diabetes mellitus", "code": "44054006", ...}]
```

#### FHIR Gateway vs Mock Data

Each evidence tool has a dual path:

- **With gateway:** Executes a real FHIR search against the configured server (e.g. Epic). Dynamically imports the FHIR resource class and calls `fhir_gateway.search()`.
- **Without gateway:** Falls back to fuzzy substring matching against the mock data pools. This ensures the demo works without a live FHIR server.

### Discrepancy Classification

The orchestrator classifies each discrepancy using three keyword-based classifiers to decide which tools to dispatch:

**`_is_diagnosis_related`** — Keywords: `condition`, `disease`, `disorder`, `diagnosis`, `syndrome`, `injury`. Also matches all `llm_only` discrepancies (since those are diagnosis-level findings).

**`_is_medication_related`** — Keywords: `medication`, `drug`, `dose`, `tablet`, `capsule`, `mg`, `ml`.

**`_is_lab_related`** — Keywords: `lab`, `test`, `glucose`, `creatinine`, `hba1c`, `wbc`, `bnp`, `hemoglobin`, `observation`, `result`, `level`.

A single discrepancy can trigger multiple tools if it matches multiple classifiers.

### Resolution Flow

1. For each discrepancy, classify and dispatch appropriate evidence tools
2. For each low-confidence diagnosis, search patient conditions
3. Collect evidence summaries from successful tool results
4. Re-prompt the LLM with the clinical note + discrepancies + gathered evidence
5. Merge the LLM's resolution back into the main result

The evidence is formatted as a text block passed to the LLM's `reason_with_resolution()` method:

```
[search_patient_conditions] diabetes: Found 1 matching condition(s): Type 2 diabetes mellitus
[search_patient_observations] glucose: Found 2 matching observation(s): HbA1c=7.8 %, Glucose=185 mg/dL
```

---

## GapFiller (Legacy)

**Module:** `medscribe.src.agent.gap_filler`

An earlier version of gap resolution that queries a live FHIR server directly. This has been superseded by the `ToolRegistry` approach but remains in the codebase. It requires a `FHIRGateway` instance and searches `Observation` and `Condition` resources, scoring relevance via substring matching.
