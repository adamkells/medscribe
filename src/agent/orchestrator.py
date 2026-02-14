"""MedScribe Agent - Agentic orchestrator for clinical documentation and coding.

Implements the core agentic loop: NLP extraction → LLM reasoning → validation → gap filling → output.
"""

import logging

from healthchain.fhir import (
    create_condition,
    create_medication_statement,
    create_bundle,
    add_resource,
)
from healthchain.io import Document
from healthchain.models.responses.cdsresponse import Card, Source, IndicatorEnum
from healthchain.pipeline import MedicalCodingPipeline

from medscribe.src.agent.tools import ToolRegistry, ToolResult
from medscribe.src.agent.validator import DualPathwayValidator
from medscribe.src.models.medgemma import MedGemmaClientProtocol

logger = logging.getLogger(__name__)


class MedScribeAgent:
    """Agentic orchestrator for clinical documentation and coding.

    Pipeline: NLP extraction → LLM reasoning → dual-pathway validation →
    gap filling (optional) → FHIR output + CDS cards.
    """

    def __init__(
        self,
        coding_pipeline: MedicalCodingPipeline,
        medgemma_client: MedGemmaClientProtocol,
        fhir_gateway=None,
        fhir_source: str = "epic",
        confidence_threshold: float = 0.7,
    ):
        self.coding_pipeline = coding_pipeline
        self.medgemma = medgemma_client
        self.validator = DualPathwayValidator()
        self.fhir_gateway = fhir_gateway
        self.fhir_source = fhir_source
        self.confidence_threshold = confidence_threshold
        self.tool_registry = ToolRegistry(
            fhir_gateway=fhir_gateway, fhir_source=fhir_source
        )

    def run(self, clinical_note: str, patient_id: str = "Patient/123") -> dict:
        """Execute the full agentic pipeline on a clinical note."""
        logger.info("Starting MedScribe agent run for patient %s", patient_id)

        # Step 1: NLP extraction via HealthChain pipeline
        logger.info("Step 1: NLP extraction")
        doc = Document(clinical_note)
        doc = self.coding_pipeline(doc)
        nlp_entities = doc.nlp.get_entities()
        logger.info("NLP extracted %d entities", len(nlp_entities))

        # Step 2: LLM reasoning via MedGemma
        logger.info("Step 2: LLM reasoning")
        llm_result = self.medgemma.reason_over_note(clinical_note, nlp_entities)
        logger.info(
            "LLM found %d diagnoses, %d procedures, %d medications",
            len(llm_result.get("diagnoses", [])),
            len(llm_result.get("procedures", [])),
            len(llm_result.get("medications", [])),
        )

        # Step 3: Compare & validate (AGENTIC DECISION)
        logger.info("Step 3: Dual-pathway validation")
        discrepancies = self.validator.compare(nlp_entities, llm_result)
        has_low_conf = self._has_low_confidence(llm_result)
        logger.info(
            "Found %d discrepancies, low_confidence=%s",
            len(discrepancies),
            has_low_conf,
        )

        # Step 4: Structured tool dispatch for gap resolution
        resolved = llm_result
        tool_calls: list[dict] = []
        if discrepancies or has_low_conf:
            logger.info("Step 4: Tool-based gap resolution")
            resolved, tool_calls = self._resolve_with_tools(
                clinical_note, discrepancies, llm_result, patient_id
            )
        else:
            logger.info("Step 4: No gaps to resolve")

        # Step 5: Build FHIR output + CDS cards
        logger.info("Step 5: Building output")
        fhir_bundle = self._build_fhir_bundle(resolved, doc, patient_id)
        cds_cards = self._build_cds_cards(resolved)

        return {
            "fhir_bundle": fhir_bundle,
            "cds_cards": cds_cards,
            "nlp_entities": nlp_entities,
            "llm_reasoning": llm_result,
            "resolved": resolved,
            "discrepancies": discrepancies,
            "tool_calls": tool_calls,
        }

    def _has_low_confidence(self, llm_result: dict) -> bool:
        """Check if any LLM diagnosis has low confidence."""
        for dx in llm_result.get("diagnoses", []):
            if dx.get("confidence") == "low":
                return True
        return False

    def _resolve_with_tools(
        self,
        clinical_note: str,
        discrepancies: list,
        llm_result: dict,
        patient_id: str,
    ) -> tuple[dict, list[dict]]:
        """Dispatch tools based on discrepancy type, gather evidence, re-prompt LLM."""
        tool_results: list[ToolResult] = []

        for disc in discrepancies:
            entity = disc.get("entity", "")
            if not entity:
                continue

            # Classify discrepancy and dispatch appropriate tools
            if self._is_diagnosis_related(disc):
                result = self.tool_registry.execute(
                    "search_patient_conditions",
                    entity=entity,
                    patient_id=patient_id,
                )
                tool_results.append(result)

            if self._is_medication_related(disc):
                result = self.tool_registry.execute(
                    "search_patient_medications",
                    entity=entity,
                    patient_id=patient_id,
                )
                tool_results.append(result)

            if self._is_lab_related(disc):
                result = self.tool_registry.execute(
                    "search_patient_observations",
                    entity=entity,
                    patient_id=patient_id,
                )
                tool_results.append(result)

        # Also search for any low-confidence diagnoses
        for dx in llm_result.get("diagnoses", []):
            if dx.get("confidence") == "low":
                entity = dx.get("text", "")
                result = self.tool_registry.execute(
                    "search_patient_conditions",
                    entity=entity,
                    patient_id=patient_id,
                )
                tool_results.append(result)

        # Build evidence summary for LLM re-prompting
        evidence_parts = []
        for tr in tool_results:
            if tr.success:
                evidence_parts.append(f"[{tr.tool_name}] {tr.entity}: {tr.summary}")

        patient_history = "\n".join(evidence_parts) if evidence_parts else ""
        logger.info(
            "Tool dispatch: %d tools called, %d with evidence",
            len(tool_results),
            len(evidence_parts),
        )

        # Re-prompt LLM with gathered evidence
        resolution = self.medgemma.reason_with_resolution(
            clinical_note, discrepancies, patient_history
        )
        resolved = self._merge_resolution(llm_result, resolution)

        return resolved, [tr.to_dict() for tr in tool_results]

    @staticmethod
    def _is_diagnosis_related(disc: dict) -> bool:
        """Check if discrepancy relates to a diagnosis/condition."""
        keywords = {
            "condition",
            "disease",
            "disorder",
            "diagnosis",
            "syndrome",
            "injury",
        }
        text = (disc.get("entity", "") + " " + disc.get("type", "")).lower()
        return any(k in text for k in keywords) or disc.get("type") == "llm_only"

    @staticmethod
    def _is_medication_related(disc: dict) -> bool:
        """Check if discrepancy relates to a medication."""
        keywords = {"medication", "drug", "dose", "tablet", "capsule", "mg", "ml"}
        text = (disc.get("entity", "") + " " + disc.get("type", "")).lower()
        return any(k in text for k in keywords)

    @staticmethod
    def _is_lab_related(disc: dict) -> bool:
        """Check if discrepancy relates to a lab result or observation."""
        keywords = {
            "lab",
            "test",
            "glucose",
            "creatinine",
            "hba1c",
            "wbc",
            "bnp",
            "hemoglobin",
            "observation",
            "result",
            "level",
        }
        text = (disc.get("entity", "") + " " + disc.get("type", "")).lower()
        return any(k in text for k in keywords)

    def _merge_resolution(self, llm_result: dict, resolution: dict) -> dict:
        """Merge resolution results back into the main result."""
        merged = {**llm_result}
        resolved_items = resolution.get("resolved", [])

        for item in resolved_items:
            # Update existing diagnoses or add new ones
            found = False
            for dx in merged.get("diagnoses", []):
                if dx.get("text", "").lower() == item.get("text", "").lower():
                    dx["icd10"] = item.get("code", dx.get("icd10"))
                    dx["confidence"] = item.get("confidence", dx.get("confidence"))
                    dx["resolution_reasoning"] = item.get("reasoning", "")
                    found = True
                    break
            if not found:
                merged.setdefault("diagnoses", []).append(
                    {
                        "text": item.get("text", ""),
                        "icd10": item.get("code", ""),
                        "confidence": item.get("confidence", "medium"),
                        "gaps": "",
                        "resolution_reasoning": item.get("reasoning", ""),
                    }
                )

        return merged

    def _build_fhir_bundle(
        self, resolved: dict, doc: Document, patient_id: str = "Patient/123"
    ):
        """Build a FHIR Bundle from resolved results using HealthChain utilities."""
        bundle = create_bundle(bundle_type="collection")

        for dx in resolved.get("diagnoses", []):
            condition = create_condition(
                subject=patient_id,
                clinical_status="active",
                code=dx.get("icd10"),
                display=dx.get("text", ""),
                system="http://hl7.org/fhir/sid/icd-10-cm",
            )
            add_resource(bundle, condition)

        for med in resolved.get("medications", []):
            med_stmt = create_medication_statement(
                subject=patient_id,
                status="recorded",
                display=med.get("text", ""),
            )
            add_resource(bundle, med_stmt)

        return bundle

    def _build_cds_cards(self, resolved: dict) -> list[Card]:
        """Build CDS Hooks cards for physician review."""
        cards: list[Card] = []
        source = Source(label="MedScribe Agent")

        for dx in resolved.get("diagnoses", []):
            has_gaps = bool(dx.get("gaps"))
            low_conf = dx.get("confidence") == "low"

            if has_gaps or low_conf:
                indicator = IndicatorEnum.warning
            else:
                indicator = IndicatorEnum.info

            summary = f"{dx.get('text', 'Unknown')} -> {dx.get('icd10', 'N/A')}"

            detail = dx.get("gaps") or "Coding validated"
            if dx.get("resolution_reasoning"):
                detail += f" | Resolution: {dx['resolution_reasoning']}"

            cards.append(
                Card(
                    summary=summary[:140],
                    indicator=indicator,
                    source=source,
                    detail=detail,
                )
            )

        return cards
