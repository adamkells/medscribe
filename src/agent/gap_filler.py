"""FHIR history search for missing evidence and documentation gap filling."""

from fhir.resources.observation import Observation
from fhir.resources.condition import Condition

from healthchain.gateway import FHIRGateway


class GapFiller:
    """Queries patient FHIR history to find supporting evidence for gaps.

    When the agent detects documentation gaps (e.g., a diagnosis mentioned
    but no supporting lab/imaging), this component searches FHIR resources
    for corroborating evidence.
    """

    def __init__(self, fhir_gateway: FHIRGateway, fhir_source: str = "epic"):
        self.fhir_gateway = fhir_gateway
        self.fhir_source = fhir_source

    def search_supporting_evidence(self, patient_id: str, gap: dict) -> list[dict]:
        """Search patient history for evidence supporting a documentation gap."""
        evidence = []

        # Search recent observations (labs, vitals)
        observations = self.fhir_gateway.search(
            Observation,
            {
                "patient": patient_id,
                "_sort": "-date",
                "_count": "20",
            },
            source=self.fhir_source,
        )
        if observations:
            for obs in observations:
                evidence.append(
                    {
                        "type": "observation",
                        "resource": obs,
                        "relevance": self._assess_relevance(obs, gap),
                    }
                )

        # Search existing conditions
        conditions = self.fhir_gateway.search(
            Condition,
            {"patient": patient_id},
            source=self.fhir_source,
        )
        if conditions:
            for cond in conditions:
                evidence.append(
                    {
                        "type": "condition",
                        "resource": cond,
                        "relevance": self._assess_relevance(cond, gap),
                    }
                )

        return [e for e in evidence if e["relevance"] > 0.5]

    def _assess_relevance(self, resource, gap: dict) -> float:
        """Score how relevant a FHIR resource is to a documentation gap.

        Extracts display text from the resource's code field and compares
        against the gap's entity text using case-insensitive matching.

        Returns:
            1.0 for exact match, 0.7 for substring match, 0.0 for no match.
        """
        gap_text = gap.get("entity", "")
        if not gap_text:
            return 0.0
        gap_lower = gap_text.lower()

        # Extract display texts from resource.code (works for Observation and Condition)
        resource_texts: list[str] = []
        code = getattr(resource, "code", None)
        if code is not None:
            if getattr(code, "text", None):
                resource_texts.append(code.text)
            for coding in getattr(code, "coding", None) or []:
                if getattr(coding, "display", None):
                    resource_texts.append(coding.display)

        if not resource_texts:
            return 0.0

        for text in resource_texts:
            text_lower = text.lower()
            if text_lower == gap_lower:
                return 1.0
            if gap_lower in text_lower or text_lower in gap_lower:
                return 0.7

        return 0.0
