"""HealthChainAPI application with NoteReader + CDS Hooks services."""

import logging

from fhir.resources.documentreference import DocumentReference

from healthchain.fhir import read_content_attachment
from healthchain.gateway import HealthChainAPI, FHIRGateway
from healthchain.gateway.cds import CDSHooksService
from healthchain.gateway.soap import NoteReaderService
from healthchain.io import CdaAdapter
from healthchain.models.requests.cdarequest import CdaRequest
from healthchain.models.responses.cdaresponse import CdaResponse
from healthchain.models.requests.cdsrequest import CDSRequest
from healthchain.models.responses.cdsresponse import CDSResponse

from medscribe.src.agent.orchestrator import MedScribeAgent
from medscribe.src.models.medgemma import create_client
from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

logger = logging.getLogger(__name__)


def _extract_note_from_prefetch(prefetch: dict | None) -> str:
    """Extract clinical note text from CDS Hooks prefetch data.

    Looks for a DocumentReference resource under the "document" key
    and reads its attachment content.
    """
    if not prefetch:
        return ""

    doc_data = prefetch.get("document")
    if doc_data is None:
        return ""

    # Parse as DocumentReference if it's a dict
    if isinstance(doc_data, dict):
        doc_ref = DocumentReference.model_validate(doc_data)
    else:
        doc_ref = doc_data

    attachments = read_content_attachment(doc_ref)
    if attachments:
        return attachments[0].get("data", "")

    return ""


def create_app(
    fhir_gateway: FHIRGateway | None = None,
    fhir_source: str = "epic",
) -> HealthChainAPI:
    """Create the MedScribe HealthChainAPI application.

    Args:
        fhir_gateway: Optional FHIRGateway for patient history queries.
        fhir_source: FHIR source name to use for queries.

    Returns:
        Configured HealthChainAPI application.
    """
    # Build components
    pipeline = build_coding_pipeline_simple()
    medgemma_client = create_client()
    cda_adapter = CdaAdapter()

    agent = MedScribeAgent(
        coding_pipeline=pipeline,
        medgemma_client=medgemma_client,
        fhir_gateway=fhir_gateway,
        fhir_source=fhir_source,
    )

    # --- Create application ---
    app = HealthChainAPI(title="MedScribe Agent")

    # --- Register FHIR Gateway (if provided) ---
    if fhir_gateway is not None:
        app.register_gateway(fhir_gateway)

    # --- NoteReader Service (SOAP/CDA ingestion from Epic) ---
    notereader = NoteReaderService()

    @notereader.method("ProcessDocument")
    def process_document(request: CdaRequest) -> CdaResponse:
        """Called when a physician signs a note in Epic."""
        doc = cda_adapter.parse(request)
        note_text = doc.data

        if note_text:
            result = agent.run(note_text, patient_id="Patient/123")
            logger.info(
                "MedScribe processed note: %d CDS cards",
                len(result.get("cds_cards", [])),
            )

        return cda_adapter.format(doc)

    app.register_service(notereader)

    # --- CDS Hooks Service (alert output channel) ---
    cds_service = CDSHooksService()

    @cds_service.hook(
        hook_type="encounter-discharge",
        id="medscribe-coding-review",
        title="MedScribe Coding Review",
        description="AI-assisted coding validation and gap detection",
    )
    def coding_review(request: CDSRequest) -> CDSResponse:
        """Returns CDS cards with coding suggestions and alerts."""
        note_text = _extract_note_from_prefetch(request.prefetch)
        patient_id = request.context.patientId

        if not note_text:
            logger.warning("No note text found in CDS prefetch")
            return CDSResponse(cards=[])

        result = agent.run(note_text, patient_id=patient_id)
        cards = result.get("cds_cards", [])

        return CDSResponse(cards=cards)

    app.register_service(cds_service)

    return app


# Module-level app for `uvicorn medscribe.src.gateway.app:app`
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
