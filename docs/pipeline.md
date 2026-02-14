# NLP Pipeline

**Module:** `medscribe.src.pipeline.coding_pipeline`

MedScribe uses HealthChain's `MedicalCodingPipeline` with scispaCy for biomedical named entity recognition. Two pipeline configurations are available depending on whether UMLS entity linking is installed.

## Pipeline Configurations

### Simple Pipeline (Default)

```python
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

pipeline = build_coding_pipeline_simple(
    model_name="en_core_sci_sm",     # scispaCy biomedical model
    patient_ref="Patient/123",        # FHIR patient reference
)
```

This is the default used throughout MedScribe. It runs NER only — no entity linking, no UMLS knowledge base required. Entities will have text spans and labels but no CUI codes.

**Components:**
1. `MedicalCodingPipeline(extract_problems=False)` — pipeline container
2. `SpacyNLP(nlp)` at stage `"ner"` — scispaCy NER

**Entity output format:**

```python
doc = Document("Patient has hypertension and diabetes.")
doc = pipeline(doc)
entities = doc.nlp.get_entities()
# [
#     {"text": "hypertension", "label": "ENTITY", "start": 12, "end": 24},
#     {"text": "diabetes", "label": "ENTITY", "start": 29, "end": 37},
# ]
```

### Full Pipeline (With UMLS Linking)

```python
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline

pipeline = build_coding_pipeline(
    model_name="en_core_sci_sm",
    linker_name="umls",                    # UMLS knowledge base
    resolve_abbreviations=True,
    min_linking_score=0.7,                  # CUI confidence threshold
    patient_ref="Patient/123",
)
```

Adds UMLS concept linking on top of NER. Entities get CUI codes which enable automatic FHIR problem list extraction.

**Components:**
1. `MedicalCodingPipeline(extract_problems=True, code_attribute="cui")` — pipeline with FHIR extraction
2. `SpacyNLP(nlp)` at stage `"ner+l"` — scispaCy NER + UMLS linker
3. `CUIExtractor(min_score=0.7)` at stage `"cui_extract"` — bridges `kb_ents` to `cui` attribute
4. `FHIRProblemListExtractor` (auto-added) — converts CUI-coded entities to FHIR Conditions

**Requirements:** The UMLS knowledge base (~3 GB) must be installed. scispaCy will download it on first use of the linker.

## CUIExtractor

A custom pipeline component that bridges scispaCy's entity linking output format to what HealthChain expects.

scispaCy stores linked concepts as `ent._.kb_ents = [(CUI, score), ...]`. HealthChain's `FHIRProblemListExtractor` expects `ent._.cui` as a string. `CUIExtractor` registers the `cui` Span extension and maps the top-scoring CUI (above `min_score`) to each entity.

```python
class CUIExtractor:
    def __init__(self, min_score: float = 0.7):
        ...

    def __call__(self, doc: Document) -> Document:
        # For each entity in doc.nlp spacy_doc:
        #   if ent._.kb_ents and top score >= min_score:
        #     ent._.cui = top_cui
        return doc
```

## scispaCy Model

MedScribe uses `en_core_sci_sm`, a small biomedical NER model trained on biomedical text. It recognizes medical entities but does not differentiate entity types (all entities get the generic `ENTITY` label). The LLM reasoning step (Step 2) handles entity classification and coding.

The model is installed as a pip dependency from the scispaCy S3 bucket (see `medscribe/pyproject.toml`).

## How the Pipeline Fits Into the Agent

The pipeline runs as Step 1 of the agentic loop:

```python
# Inside MedScribeAgent.run():
doc = Document(clinical_note)
doc = self.coding_pipeline(doc)        # Step 1: NLP extraction
nlp_entities = doc.nlp.get_entities()  # list[dict] with text, label, start, end
```

The extracted entities serve two purposes:
1. **Sent to MedGemma** (Step 2) as additional context for LLM reasoning
2. **Compared with LLM output** (Step 3) by the `DualPathwayValidator` to detect discrepancies
