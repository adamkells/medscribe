"""End-to-end integration tests using HealthChain SandboxClient."""

import sys
import os

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from healthchain.sandbox import SandboxClient  # noqa: E402


# ---- Phase 1: Standalone end-to-end test (no live server) ----


def test_end_to_end_demo():
    """Run the full MedScribe agent pipeline in standalone demo mode.

    This test does not require a running server or FHIR gateway.
    It uses the simple NLP pipeline (no UMLS linker) and demo MedGemma client.
    """
    from medscribe.src.models.medgemma import DemoMedGemmaClient
    from medscribe.src.agent.orchestrator import MedScribeAgent

    # Build simple pipeline (NER only, no UMLS download needed)
    try:
        from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

        pipeline = build_coding_pipeline_simple()
    except Exception as e:
        print(f"Could not build pipeline: {e}")
        print("Make sure en_core_sci_sm is installed")
        return None

    client = DemoMedGemmaClient()

    agent = MedScribeAgent(
        coding_pipeline=pipeline,
        medgemma_client=client,
        fhir_gateway=None,
    )

    sample_note = (
        "Patient is a 65-year-old male admitted for community-acquired pneumonia. "
        "Past medical history significant for type 2 diabetes mellitus, hypertension, "
        "and hyperlipidemia. Currently on metformin 1000mg BID, lisinopril 20mg daily, "
        "and atorvastatin 40mg at bedtime. Chest X-ray shows right lower lobe infiltrate. "
        "Started on azithromycin 500mg IV and ceftriaxone 1g IV. "
        "Labs show WBC 14.2, glucose 185 mg/dL, HbA1c 7.8%. "
        "Assessment: Community-acquired pneumonia, uncontrolled type 2 diabetes."
    )

    output = agent.run(sample_note, patient_id="Patient/demo-001")

    # Validate output structure
    assert "fhir_bundle" in output, "Missing fhir_bundle"
    assert "cds_cards" in output, "Missing cds_cards"
    assert "nlp_entities" in output, "Missing nlp_entities"
    assert "llm_reasoning" in output, "Missing llm_reasoning"
    assert "discrepancies" in output, "Missing discrepancies"

    # Validate NLP extracted something
    assert len(output["nlp_entities"]) > 0, "NLP should extract at least one entity"

    # Validate LLM reasoning structure
    llm = output["llm_reasoning"]
    assert "diagnoses" in llm, "LLM should return diagnoses"
    assert len(llm["diagnoses"]) > 0, "LLM should find at least one diagnosis"

    # Validate CDS cards are proper Card models
    from healthchain.models.responses.cdsresponse import Card

    assert len(output["cds_cards"]) > 0, "Should generate at least one CDS card"
    for card in output["cds_cards"]:
        assert isinstance(card, Card), f"Expected Card model, got {type(card)}"
        assert card.indicator in ("info", "warning", "critical")

    print(f"NLP entities: {len(output['nlp_entities'])}")
    print(f"LLM diagnoses: {len(llm['diagnoses'])}")
    print(f"Discrepancies: {len(output['discrepancies'])}")
    print(f"CDS cards: {len(output['cds_cards'])}")

    print("\nCDS Cards:")
    for card in output["cds_cards"]:
        print(f"  [{card.indicator.value:8s}] {card.summary}")

    print("\nEnd-to-end demo test PASSED")
    return output


# ---- Phase 2: Gateway creation test ----


def test_gateway_creation():
    """Test that the MedScribe gateway app creates successfully."""
    from medscribe.src.gateway.app import create_app

    app = create_app()

    # Verify the app is a HealthChainAPI instance
    from healthchain.gateway import HealthChainAPI

    assert isinstance(app, HealthChainAPI), "create_app should return HealthChainAPI"

    # Verify routes are registered by checking the app has routes
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    print(f"Registered routes: {routes}")

    # NoteReader SOAP service should be registered
    notereader_routes = [r for r in routes if "notereader" in r.lower()]
    assert len(notereader_routes) > 0, "NoteReader service should be registered"

    # CDS Hooks service should be registered
    cds_routes = [r for r in routes if "cds" in r.lower()]
    assert len(cds_routes) > 0, "CDS Hooks service should be registered"

    print("Gateway creation test PASSED")
    return app


# ---- Phase 2: Server-based integration tests ----


def test_notereader_soap_flow(port: int = 8001):
    """Test NoteReader (SOAP) flow via live server."""
    import threading
    from time import sleep

    import uvicorn

    from medscribe.src.gateway.app import create_app

    app = create_app()

    def run_server():
        uvicorn.run(app, port=port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    sleep(2)

    soap_client = SandboxClient(
        url=f"http://localhost:{port}/notereader/?wsdl",
        workflow="sign-note-inpatient",
        protocol="soap",
    )

    # Use a minimal CDA document for testing
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    cda_file = os.path.join(data_path, "notereader_cda.xml")
    if os.path.exists(cda_file):
        soap_client.load_from_path(cda_file)
        responses = soap_client.send_requests()
        print(f"SOAP responses: {len(responses)}")
        print("NoteReader SOAP flow test PASSED")
        return responses
    else:
        print(f"Skipping SOAP test: {cda_file} not found")
        return None


def test_cds_hooks_rest_flow(port: int = 8002):
    """Test CDS Hooks (REST) flow via live server."""
    import threading
    from time import sleep

    import uvicorn

    from healthchain.fhir import create_document_reference
    from medscribe.src.gateway.app import create_app

    app = create_app()

    def run_server():
        uvicorn.run(app, port=port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    sleep(2)

    cds_client = SandboxClient(
        url=f"http://localhost:{port}/cds/cds-services/medscribe-coding-review",
        workflow="encounter-discharge",
    )

    # Build prefetch with a DocumentReference containing a clinical note
    note_text = (
        "Patient is a 72-year-old female with community-acquired pneumonia. "
        "History of COPD and hypertension. Started on levofloxacin 750mg IV."
    )
    doc_ref = create_document_reference(
        data=note_text, content_type="text/plain", status="current"
    )
    prefetch = {"document": doc_ref}
    cds_client._construct_request(prefetch)

    responses = cds_client.send_requests()
    print(f"CDS responses: {len(responses)}")
    print("CDS Hooks REST flow test PASSED")
    return responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedScribe Integration Tests")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run server-based tests (starts live servers)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MedScribe Agent - Integration Tests")
    print("=" * 60)

    print("\n--- Phase 1: Standalone Demo Test ---")
    test_end_to_end_demo()

    print("\n--- Phase 2: Gateway Creation Test ---")
    test_gateway_creation()

    if args.server:
        print("\n--- Phase 2: SOAP Integration Test ---")
        test_notereader_soap_flow()

        print("\n--- Phase 2: CDS Hooks Integration Test ---")
        test_cds_hooks_rest_flow()

    print("\nAll tests complete.")
