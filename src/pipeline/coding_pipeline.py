"""MedicalCodingPipeline with scispaCy entity linking for UMLS/SNOMED codes."""

import logging

from healthchain.io import Document
from healthchain.pipeline import MedicalCodingPipeline
from healthchain.pipeline.components.integrations import SpacyNLP

logger = logging.getLogger(__name__)


class CUIExtractor:
    """Bridge component: extracts top CUI from scispaCy's kb_ents into ent._.cui.

    scispaCy's EntityLinker sets ent._.kb_ents = [(CUI, score), ...] but
    HealthChain's FHIRProblemListExtractor expects ent._.cui (a string).
    This component bridges the gap.
    """

    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score

    def __call__(self, doc: Document) -> Document:
        from spacy.tokens import Span

        spacy_doc = doc.nlp.get_spacy_doc()
        if spacy_doc is None:
            return doc

        # Register the cui extension if not already present
        if not Span.has_extension("cui"):
            Span.set_extension("cui", default=None)

        for ent in spacy_doc.ents:
            if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                top_cui, top_score = ent._.kb_ents[0]
                if top_score >= self.min_score:
                    ent._.cui = top_cui
                    logger.debug(
                        "Linked '%s' -> CUI %s (score=%.3f)",
                        ent.text,
                        top_cui,
                        top_score,
                    )

        return doc


def build_coding_pipeline(
    model_name: str = "en_core_sci_sm",
    linker_name: str = "umls",
    resolve_abbreviations: bool = True,
    min_linking_score: float = 0.7,
    patient_ref: str = "Patient/123",
) -> MedicalCodingPipeline:
    """Build a MedicalCodingPipeline with scispaCy UMLS entity linker.

    Args:
        model_name: scispaCy model to load.
        linker_name: Entity linker name ("umls" or "mesh").
        resolve_abbreviations: Whether to resolve abbreviations before linking.
        min_linking_score: Minimum score to accept a CUI link.
        patient_ref: FHIR patient reference for generated Conditions.

    Returns:
        Configured MedicalCodingPipeline with NER + entity linking + FHIR extraction.
    """
    import spacy

    nlp = spacy.load(model_name)
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": resolve_abbreviations,
            "linker_name": linker_name,
        },
    )

    pipeline = MedicalCodingPipeline(
        extract_problems=True,
        patient_ref=patient_ref,
        code_attribute="cui",
    )

    # Stage 1: NER + entity linking via scispaCy
    pipeline.add_node(SpacyNLP(nlp), stage="ner+l")

    # Stage 2: Bridge kb_ents -> cui attribute
    pipeline.add_node(CUIExtractor(min_score=min_linking_score), stage="cui_extract")

    # Stage 3: FHIRProblemListExtractor is auto-added by MedicalCodingPipeline

    return pipeline


def build_coding_pipeline_simple(
    model_name: str = "en_core_sci_sm",
    patient_ref: str = "Patient/123",
) -> MedicalCodingPipeline:
    """Build a simpler pipeline with NER only (no UMLS entity linking).

    Use this for development when the UMLS linker KB (~3GB) is not installed.
    Entities will not have CUI codes, so FHIR problem list extraction is skipped.

    Args:
        model_name: scispaCy model to load (just NER, no linker).
        patient_ref: FHIR patient reference for generated Conditions.
    """
    import spacy

    nlp = spacy.load(model_name)

    pipeline = MedicalCodingPipeline(
        extract_problems=False,
        patient_ref=patient_ref,
    )
    pipeline.add_node(SpacyNLP(nlp), stage="ner+l")

    return pipeline


def test_pipeline(use_linker: bool = True) -> Document:
    """Quick validation that the pipeline works end-to-end."""
    sample_note = (
        "Patient is a 65-year-old male presenting with community-acquired pneumonia. "
        "History of type 2 diabetes mellitus and hypertension. "
        "Currently on metformin 1000mg BID and lisinopril 20mg daily. "
        "Chest X-ray shows right lower lobe infiltrate. "
        "Started on azithromycin 500mg IV."
    )

    if use_linker:
        pipeline = build_coding_pipeline()
    else:
        pipeline = build_coding_pipeline_simple()

    doc = Document(sample_note)
    result = pipeline(doc)

    entities = result.nlp.get_entities()
    print(f"Found {len(entities)} entities:")
    for e in entities:
        print(f"  {e.get('text', 'N/A')} [{e.get('label', 'N/A')}]")

    spacy_doc = result.nlp.get_spacy_doc()
    if spacy_doc:
        for ent in spacy_doc.ents:
            cui = getattr(ent._, "cui", None) if hasattr(ent._, "cui") else None
            kb = getattr(ent._, "kb_ents", []) if hasattr(ent._, "kb_ents") else []
            cui_str = f" -> CUI: {cui}" if cui else ""
            kb_str = f" (top kb_ent: {kb[0]})" if kb else ""
            print(f"  {ent.text} [{ent.label_}]{cui_str}{kb_str}")

    problems = result.fhir.problem_list
    print(f"\nFHIR Problem List: {len(problems)} conditions")
    for cond in problems:
        code = cond.code
        if code and code.coding:
            print(f"  {code.coding[0].display} ({code.coding[0].code})")

    return result


if __name__ == "__main__":
    test_pipeline(use_linker=False)
