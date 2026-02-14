# MedScribe Agent — Architecture

## Pipeline Overview

MedScribe implements a 5-step agentic pipeline for autonomous clinical documentation coding. The dual-pathway design (NLP + LLM) with validation and gap resolution ensures coding accuracy beyond either pathway alone.

```mermaid
flowchart TD
    EHR["EHR System<br/>(Epic, Cerner)"]
    GW["HealthChain Gateway<br/>NoteReader (SOAP/CDA) | CDS Hooks (REST)"]
    NLP["Step 1: NLP Extraction<br/>scispaCy NER → Medical Entities"]
    LLM["Step 2: MedGemma LLM Reasoning<br/>Diagnoses, Procedures, Medications<br/>+ ICD-10 / CPT Codes"]
    VAL["Step 3: Dual-Pathway Validation<br/>NLP vs LLM Comparison<br/>→ Discrepancy Detection"]
    DEC{{"Discrepancies or<br/>Low Confidence?"}}
    TOOLS["Tool Dispatch<br/>ToolRegistry → Evidence Search"]
    GAP["Step 4: Gap Resolution<br/>Evidence + LLM Re-prompt"]
    OUT["Step 5: Output Generation"]
    FHIR["FHIR Bundle<br/>(Conditions, Observations)"]
    CDS["CDS Hooks Cards<br/>(info / warning / critical)"]
    EHROUT["EHR System<br/>Physician Review"]

    EHR -->|"Clinical Note"| GW
    GW --> NLP
    NLP -->|"Entities"| LLM
    NLP -->|"Entities"| VAL
    LLM -->|"Diagnoses"| VAL
    VAL --> DEC
    DEC -->|"Yes"| TOOLS
    TOOLS -->|"Evidence"| GAP
    DEC -->|"No"| OUT
    GAP -->|"Resolved"| OUT
    OUT --> FHIR
    OUT --> CDS
    FHIR --> EHROUT
    CDS --> EHROUT

    style EHR fill:#e8d5b7,stroke:#333
    style GW fill:#b7d5e8,stroke:#333
    style NLP fill:#d5e8b7,stroke:#333
    style LLM fill:#d5e8b7,stroke:#333
    style VAL fill:#e8b7d5,stroke:#333
    style DEC fill:#f5e6cc,stroke:#333
    style TOOLS fill:#f5d5e8,stroke:#333
    style GAP fill:#e8b7d5,stroke:#333
    style OUT fill:#b7e8d5,stroke:#333
    style FHIR fill:#b7e8d5,stroke:#333
    style CDS fill:#b7e8d5,stroke:#333
    style EHROUT fill:#e8d5b7,stroke:#333
```

## Component Mapping

| Pipeline Step | Module | Key Class / Function |
|---|---|---|
| Gateway | `medscribe.src.gateway.app` | `create_app()` → HealthChainAPI |
| Step 1: NLP | `medscribe.src.pipeline.coding_pipeline` | `build_coding_pipeline_simple()` → scispaCy NER |
| Step 2: LLM | `medscribe.src.models.medgemma` | `MedGemmaClient` / `DemoMedGemmaClient` |
| Step 3: Validation | `medscribe.src.agent.validator` | `DualPathwayValidator.compare()` |
| Step 4: Tool Dispatch | `medscribe.src.agent.tools` | `ToolRegistry.execute()` → evidence search (FHIR or mock) |
| Step 4: Gap Resolution | `medscribe.src.agent.orchestrator` | `_resolve_with_tools()` → evidence + `reason_with_resolution()` |
| Step 5: Output | `medscribe.src.agent.orchestrator` | `MedScribeAgent._build_fhir_bundle()` / `_build_cds_cards()` |
| Orchestrator | `medscribe.src.agent.orchestrator` | `MedScribeAgent.run()` |

## Data Flow

1. **EHR → Gateway**: Clinical note arrives via SOAP/CDA (NoteReader) or REST (CDS Hooks)
2. **Gateway → NLP**: Raw text extracted, passed to scispaCy for entity recognition
3. **NLP + LLM → Validation**: Both pathways produce findings; validator detects discrepancies
4. **Validation → Tool Dispatch → Resolution**: Agent dispatches ToolRegistry tools (condition/medication/lab search) based on discrepancy type, gathers evidence, then re-prompts LLM
5. **Resolution → Output**: Final diagnoses converted to FHIR Bundle + CDS Hooks Cards for physician review
