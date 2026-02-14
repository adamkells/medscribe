# Streamlit Demo

**Module:** `medscribe.demo.app`

An interactive Streamlit application that visualizes the full MedScribe agent pipeline.

## Running the Demo

```bash
MEDGEMMA_MODE=demo uv run streamlit run medscribe/demo/app.py
```

Opens at `http://localhost:8501`. The demo forces `MEDGEMMA_MODE=demo` if not already set.

## Layout

### Sidebar

- **Mode indicator:** Shows the current `MEDGEMMA_MODE`
- **Sample note selector:** Dropdown with 4 pre-loaded clinical notes + "Custom" option
- **Clinical note input:** Text area (pre-filled for sample notes, editable)
- **Patient ID:** Text input, defaults to `Patient/demo-001`
- **Run Agent button:** Triggers the full pipeline

### Main Area

After clicking **Run Agent**, the main area shows:

1. **Pipeline Execution** — 5-step progressive visualization
2. **Results** — 7 tabbed panels

## Sample Clinical Notes

| Note | Scenario | Complexity |
|------|----------|------------|
| Pneumonia + Diabetes + Hypertension | 65M, inpatient, community-acquired pneumonia | 3 conditions + medications + labs |
| Heart Failure + Atrial Fibrillation | 72F, cardiology clinic, EF 35% | 2 conditions + rate control |
| Sepsis + AKI + UTI (Critical Care) | 78F, ICU transfer, sepsis from UTI | 3 conditions + critical labs |
| Complex Geriatric (6 Conditions) | 82M, SNF admit, functional decline | 6 conditions including new diagnosis |

## Pipeline Visualization

Five `st.status` blocks that expand sequentially with 300ms delays for visual effect:

**Step 1: NLP Entity Extraction** — Shows entity count and type breakdown (e.g., "ENTITY: 23").

**Step 2: MedGemma LLM Reasoning** — Shows diagnosis, procedure, and medication counts.

**Step 3: Dual-Pathway Validation** — Shows total discrepancies with NLP-only vs LLM-only breakdown.

**Step 4: Tool-Based Gap Resolution** — Shows tool dispatch count, successful evidence count, and resolved items. If no tools were dispatched, shows resolution count or "No gaps requiring resolution".

**Step 5: FHIR + CDS Output** — Shows CDS card count, FHIR Bundle status, and total pipeline latency.

## Result Tabs

### CDS Cards

Renders each CDS Hooks card as a bordered container with severity styling:

- `st.error()` for critical indicators
- `st.warning()` for warning indicators
- `st.info()` for info indicators

Each card shows the summary (condition -> ICD-10 code) and detail text.

### NLP Entities

A `st.dataframe` table with columns: Text, Label, Code, Start, End. Shows all entities extracted by scispaCy in Step 1.

### LLM Reasoning

Three sub-tables:

- **Diagnoses:** Condition, ICD-10, Confidence, Gaps
- **Procedures:** Procedure, CPT
- **Medications:** Medication, Status

### Discrepancies

Two-column layout:

- **Left: NLP-Only Findings** — Entities found by NLP but not confirmed by LLM
- **Right: LLM-Only Findings** — Diagnoses found by LLM but not detected by NLP

Shows a success message if no discrepancies exist.

### Agent Trace

Renders tool dispatch results from Step 4 as bordered cards. Each card shows:

- Tool name and entity (e.g., `search_patient_conditions — Entity: diabetes`)
- Reason for the tool call
- Result summary (e.g., "Found 1 matching condition(s): Type 2 diabetes mellitus")
- Expandable "Raw data" section with the full JSON response

Shows an info message if no tools were dispatched.

### FHIR Bundle

Renders the FHIR Bundle as formatted JSON via `st.json()`. The bundle contains:

- `Condition` resources for each diagnosis (ICD-10-CM coded)
- `MedicationStatement` resources for each medication

Shows a guidance message if no bundle is available (e.g., when using the simple pipeline without UMLS linking).

### Metrics

Two rows of metric cards:

| Row 1 | Row 2 |
|-------|-------|
| NLP Entities | Procedures |
| Diagnoses | Medications |
| Discrepancies | Coding Gaps |
| CDS Cards | Latency |

Plus a **Confidence Distribution** section showing high/medium/low confidence counts.

## Caching

The agent is loaded once via `@st.cache_resource` to avoid reloading the scispaCy model on every Streamlit rerun. The cached `load_agent()` function builds the pipeline, client, and agent together.
