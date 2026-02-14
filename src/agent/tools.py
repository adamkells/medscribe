"""Tool registry for MedScribe Agent structured tool dispatch.

Provides evidence-gathering tools that search patient records (via FHIR gateway
or mock data fallback) and construction tools that create FHIR resources using
HealthChain utilities.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Mock data pools for demo mode (no live FHIR server required)
MOCK_CONDITIONS = [
    {
        "display": "Essential hypertension",
        "code": "59621000",
        "system": "http://snomed.info/sct",
        "clinical_status": "active",
    },
    {
        "display": "Type 2 diabetes mellitus",
        "code": "44054006",
        "system": "http://snomed.info/sct",
        "clinical_status": "active",
    },
    {
        "display": "Chronic obstructive pulmonary disease",
        "code": "13645005",
        "system": "http://snomed.info/sct",
        "clinical_status": "active",
    },
    {
        "display": "Congestive heart failure",
        "code": "42343007",
        "system": "http://snomed.info/sct",
        "clinical_status": "active",
    },
    {
        "display": "Acute kidney injury",
        "code": "14669001",
        "system": "http://snomed.info/sct",
        "clinical_status": "active",
    },
]

MOCK_MEDICATIONS = [
    {
        "display": "Metformin 1000 MG",
        "code": "860975",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "status": "recorded",
    },
    {
        "display": "Lisinopril 20 MG",
        "code": "314076",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "status": "recorded",
    },
    {
        "display": "Metoprolol tartrate 25 MG",
        "code": "866924",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "status": "recorded",
    },
    {
        "display": "Aspirin 81 MG",
        "code": "243670",
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "status": "recorded",
    },
]

MOCK_ALLERGIES = [
    {
        "display": "Penicillin",
        "code": "91936005",
        "system": "http://snomed.info/sct",
    },
    {
        "display": "Sulfa drugs",
        "code": "387406002",
        "system": "http://snomed.info/sct",
    },
]

MOCK_OBSERVATIONS = [
    {
        "display": "HbA1c",
        "code": "4548-4",
        "system": "http://loinc.org",
        "value": "7.8",
        "unit": "%",
    },
    {
        "display": "Glucose",
        "code": "2345-7",
        "system": "http://loinc.org",
        "value": "185",
        "unit": "mg/dL",
    },
    {
        "display": "WBC",
        "code": "6690-2",
        "system": "http://loinc.org",
        "value": "14.2",
        "unit": "10*3/uL",
    },
    {
        "display": "Creatinine",
        "code": "2160-0",
        "system": "http://loinc.org",
        "value": "3.2",
        "unit": "mg/dL",
    },
    {
        "display": "BNP",
        "code": "42637-9",
        "system": "http://loinc.org",
        "value": "1450",
        "unit": "pg/mL",
    },
]

MOCK_PROCEDURES = [
    {
        "display": "Chest X-ray",
        "code": "399208008",
        "system": "http://snomed.info/sct",
    },
    {
        "display": "Echocardiogram",
        "code": "40701008",
        "system": "http://snomed.info/sct",
    },
    {
        "display": "Blood culture",
        "code": "30088009",
        "system": "http://snomed.info/sct",
    },
]


def _fuzzy_match(
    query: str, candidates: list[dict], key: str = "display"
) -> list[dict]:
    """Return candidates whose display name fuzzy-matches the query (substring)."""
    query_lower = query.lower()
    matches = []
    for item in candidates:
        display_lower = item[key].lower()
        # Match if any word in query appears in the display or vice versa
        query_words = query_lower.split()
        if any(w in display_lower for w in query_words if len(w) > 2):
            matches.append(item)
        elif any(w in query_lower for w in display_lower.split() if len(w) > 2):
            matches.append(item)
    return matches


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    entity: str
    reason: str
    success: bool
    data: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "entity": self.entity,
            "reason": self.reason,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
        }


class ToolRegistry:
    """Registry of tools available to the MedScribe agent for structured dispatch."""

    def __init__(self, fhir_gateway=None, fhir_source: str = "epic"):
        self.fhir_gateway = fhir_gateway
        self.fhir_source = fhir_source
        self._tools: dict[str, callable] = {
            "search_patient_conditions": self._search_conditions,
            "search_patient_medications": self._search_medications,
            "search_patient_allergies": self._search_allergies,
            "search_patient_observations": self._search_observations,
            "search_patient_procedures": self._search_procedures,
            "create_fhir_condition": self._create_condition,
            "create_fhir_medication": self._create_medication,
        }

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with keyword arguments."""
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                entity=kwargs.get("entity", ""),
                reason=f"Unknown tool: {tool_name}",
                success=False,
                summary=f"Tool '{tool_name}' not found in registry",
            )
        try:
            return self._tools[tool_name](**kwargs)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return ToolResult(
                tool_name=tool_name,
                entity=kwargs.get("entity", ""),
                reason=str(e),
                success=False,
                summary=f"Tool execution failed: {e}",
            )

    # --- Evidence tools ---

    def _search_conditions(
        self, entity: str = "", patient_id: str = "Patient/123", **kwargs
    ) -> ToolResult:
        """Search patient conditions via FHIR gateway or mock data."""
        if self.fhir_gateway is not None:
            return self._fhir_search(
                "search_patient_conditions",
                entity,
                patient_id,
                resource_type="Condition",
            )
        matches = _fuzzy_match(entity, MOCK_CONDITIONS)
        return ToolResult(
            tool_name="search_patient_conditions",
            entity=entity,
            reason="Searched patient condition history",
            success=bool(matches),
            data=matches,
            summary=(
                f"Found {len(matches)} matching condition(s): "
                + ", ".join(m["display"] for m in matches)
                if matches
                else "No matching conditions found in patient history"
            ),
        )

    def _search_medications(
        self, entity: str = "", patient_id: str = "Patient/123", **kwargs
    ) -> ToolResult:
        """Search patient medications via FHIR gateway or mock data."""
        if self.fhir_gateway is not None:
            return self._fhir_search(
                "search_patient_medications",
                entity,
                patient_id,
                resource_type="MedicationStatement",
            )
        matches = _fuzzy_match(entity, MOCK_MEDICATIONS)
        return ToolResult(
            tool_name="search_patient_medications",
            entity=entity,
            reason="Searched patient medication history",
            success=bool(matches),
            data=matches,
            summary=(
                f"Found {len(matches)} matching medication(s): "
                + ", ".join(m["display"] for m in matches)
                if matches
                else "No matching medications found in patient history"
            ),
        )

    def _search_allergies(
        self, entity: str = "", patient_id: str = "Patient/123", **kwargs
    ) -> ToolResult:
        """Search patient allergies via FHIR gateway or mock data."""
        if self.fhir_gateway is not None:
            return self._fhir_search(
                "search_patient_allergies",
                entity,
                patient_id,
                resource_type="AllergyIntolerance",
            )
        matches = _fuzzy_match(entity, MOCK_ALLERGIES)
        return ToolResult(
            tool_name="search_patient_allergies",
            entity=entity,
            reason="Searched patient allergy records",
            success=bool(matches),
            data=matches,
            summary=(
                f"Found {len(matches)} matching allergy record(s): "
                + ", ".join(m["display"] for m in matches)
                if matches
                else "No matching allergies found in patient history"
            ),
        )

    def _search_observations(
        self, entity: str = "", patient_id: str = "Patient/123", **kwargs
    ) -> ToolResult:
        """Search patient observations/labs via FHIR gateway or mock data."""
        if self.fhir_gateway is not None:
            return self._fhir_search(
                "search_patient_observations",
                entity,
                patient_id,
                resource_type="Observation",
            )
        matches = _fuzzy_match(entity, MOCK_OBSERVATIONS)
        return ToolResult(
            tool_name="search_patient_observations",
            entity=entity,
            reason="Searched patient lab/observation results",
            success=bool(matches),
            data=matches,
            summary=(
                f"Found {len(matches)} matching observation(s): "
                + ", ".join(
                    f"{m['display']}={m.get('value', '?')} {m.get('unit', '')}"
                    for m in matches
                )
                if matches
                else "No matching observations found in patient history"
            ),
        )

    def _search_procedures(
        self, entity: str = "", patient_id: str = "Patient/123", **kwargs
    ) -> ToolResult:
        """Search patient procedures via FHIR gateway or mock data."""
        if self.fhir_gateway is not None:
            return self._fhir_search(
                "search_patient_procedures",
                entity,
                patient_id,
                resource_type="Procedure",
            )
        matches = _fuzzy_match(entity, MOCK_PROCEDURES)
        return ToolResult(
            tool_name="search_patient_procedures",
            entity=entity,
            reason="Searched patient procedure history",
            success=bool(matches),
            data=matches,
            summary=(
                f"Found {len(matches)} matching procedure(s): "
                + ", ".join(m["display"] for m in matches)
                if matches
                else "No matching procedures found in patient history"
            ),
        )

    def _fhir_search(
        self,
        tool_name: str,
        entity: str,
        patient_id: str,
        resource_type: str,
    ) -> ToolResult:
        """Execute a real FHIR search via the gateway."""
        import importlib

        module = importlib.import_module(f"fhir.resources.{resource_type.lower()}")
        fhir_class = getattr(module, resource_type)

        result = self.fhir_gateway.search(
            fhir_class,
            {"patient": patient_id, "_count": "10"},
            source=self.fhir_source,
        )

        entries = []
        if result and hasattr(result, "entry") and result.entry:
            for entry in result.entry:
                res = entry.resource
                code_text = (
                    res.code.text if hasattr(res, "code") and res.code else "N/A"
                )
                entries.append({"display": code_text, "resource_type": resource_type})

        return ToolResult(
            tool_name=tool_name,
            entity=entity,
            reason=f"FHIR search for {resource_type} records",
            success=bool(entries),
            data=entries,
            summary=(
                f"Found {len(entries)} {resource_type} record(s) from FHIR server"
                if entries
                else f"No {resource_type} records found via FHIR"
            ),
        )

    # --- Construction tools ---

    def _create_condition(
        self,
        entity: str = "",
        patient_id: str = "Patient/123",
        code: str = None,
        system: str = "http://snomed.info/sct",
        **kwargs,
    ) -> ToolResult:
        """Create a FHIR Condition resource using HealthChain utilities."""
        from healthchain.fhir import create_condition

        create_condition(
            subject=patient_id,
            clinical_status="active",
            code=code,
            display=entity,
            system=system,
        )
        return ToolResult(
            tool_name="create_fhir_condition",
            entity=entity,
            reason="Created FHIR Condition resource",
            success=True,
            data=[{"resource_type": "Condition", "display": entity, "code": code}],
            summary=f"Created Condition: {entity}",
        )

    def _create_medication(
        self,
        entity: str = "",
        patient_id: str = "Patient/123",
        code: str = None,
        system: str = "http://www.nlm.nih.gov/research/umls/rxnorm",
        **kwargs,
    ) -> ToolResult:
        """Create a FHIR MedicationStatement resource using HealthChain utilities."""
        from healthchain.fhir import create_medication_statement

        create_medication_statement(
            subject=patient_id,
            status="recorded",
            code=code,
            display=entity,
            system=system,
        )
        return ToolResult(
            tool_name="create_fhir_medication",
            entity=entity,
            reason="Created FHIR MedicationStatement resource",
            success=True,
            data=[
                {
                    "resource_type": "MedicationStatement",
                    "display": entity,
                    "code": code,
                }
            ],
            summary=f"Created MedicationStatement: {entity}",
        )
