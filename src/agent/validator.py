"""Dual-pathway validation: NLP extraction vs LLM reasoning comparison."""


class DualPathwayValidator:
    """Compares NLP-extracted entities against LLM reasoning results.

    Identifies discrepancies between the two pathways and flags
    entities that need further review or resolution.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def compare(self, nlp_entities: list[dict], llm_result: dict) -> list[dict]:
        """Compare NLP and LLM pathways, return list of discrepancies."""
        discrepancies = []

        nlp_texts = {e.get("text", "").lower() for e in nlp_entities}
        llm_diagnoses = llm_result.get("diagnoses", [])

        # Check for LLM findings not in NLP
        for dx in llm_diagnoses:
            dx_text = dx.get("text", "").lower()
            if not self._fuzzy_match(dx_text, nlp_texts):
                discrepancies.append(
                    {
                        "type": "llm_only",
                        "entity": dx.get("text"),
                        "icd10": dx.get("icd10"),
                        "confidence": dx.get("confidence"),
                        "reason": "Found by LLM but not by NLP extraction",
                    }
                )

        # Check for NLP findings not confirmed by LLM
        llm_texts = {dx.get("text", "").lower() for dx in llm_diagnoses}
        for entity in nlp_entities:
            entity_text = entity.get("text", "").lower()
            if not self._fuzzy_match(entity_text, llm_texts):
                discrepancies.append(
                    {
                        "type": "nlp_only",
                        "entity": entity.get("text"),
                        "code": entity.get("code"),
                        "reason": "Found by NLP but not confirmed by LLM reasoning",
                    }
                )

        return discrepancies

    def _fuzzy_match(self, text: str, text_set: set[str]) -> bool:
        """Check if text approximately matches any string in the set."""
        if text in text_set:
            return True
        for candidate in text_set:
            if text in candidate or candidate in text:
                return True
        return False
