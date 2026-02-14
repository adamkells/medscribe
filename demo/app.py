"""Streamlit demo application for MedScribe Agent.

Run with: MEDGEMMA_MODE=demo uv run streamlit run medscribe/demo/app.py
"""

import os
import sys
import time

import streamlit as st

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force demo mode if not set
if "MEDGEMMA_MODE" not in os.environ:
    os.environ["MEDGEMMA_MODE"] = "demo"


# --- Sample clinical notes ---
SAMPLE_NOTES = {
    "Pneumonia + Diabetes + Hypertension": (
        "Patient is a 65-year-old male admitted for community-acquired pneumonia. "
        "Past medical history significant for type 2 diabetes mellitus and hypertension. "
        "Currently on metformin 1000mg BID and lisinopril 20mg daily. "
        "Chest X-ray shows right lower lobe infiltrate. "
        "Started on azithromycin 500mg IV. "
        "Labs show WBC 14.2, glucose 185 mg/dL, HbA1c 7.8%. "
        "Assessment: Community-acquired pneumonia, uncontrolled type 2 diabetes, hypertension."
    ),
    "Heart Failure + Atrial Fibrillation": (
        "72-year-old female presenting to the cardiology clinic with worsening dyspnea "
        "on exertion and lower extremity edema. Echocardiogram reveals EF 35% consistent "
        "with heart failure. EKG demonstrates atrial fibrillation with rapid ventricular "
        "response. Started on metoprolol 25mg twice daily for rate control and furosemide "
        "for volume management. BNP elevated at 1450 pg/mL. "
        "Assessment: Congestive heart failure with reduced ejection fraction, atrial fibrillation."
    ),
    "Sepsis + AKI + UTI (Critical Care)": (
        "78-year-old female transferred to the ICU with sepsis secondary to urinary tract "
        "infection. Urine culture growing E. coli. Creatinine acutely elevated from baseline "
        "0.9 to 3.2 mg/dL consistent with acute kidney injury. Past medical history includes "
        "type 2 diabetes managed with insulin. Blood culture obtained, started on "
        "piperacillin-tazobactam IV. Lactate 4.2 mmol/L, MAP 58 requiring norepinephrine. "
        "Assessment: Sepsis from urinary tract infection, acute kidney injury, diabetes."
    ),
    "Complex Geriatric (6 Conditions)": (
        "82-year-old male admitted from skilled nursing facility with increasing confusion "
        "and functional decline. Known history of type 2 diabetes on metformin 1000mg BID, "
        "hypertension on amlodipine 10mg daily, COPD on home oxygen 2L, and congestive heart "
        "failure with EF 30%. Labs notable for hemoglobin 9.1 g/dL indicating anemia and "
        "TSH 12.4 mIU/L consistent with hypothyroidism. Chest X-ray shows cardiomegaly with "
        "small bilateral pleural effusions. Started levothyroxine 25mcg daily. "
        "Assessment: Diabetes, hypertension, COPD, heart failure exacerbation, anemia, "
        "newly diagnosed hypothyroidism."
    ),
}


@st.cache_resource
def load_agent():
    """Initialize the MedScribe agent (cached to avoid reloading scispaCy model)."""
    from medscribe.src.agent.orchestrator import MedScribeAgent
    from medscribe.src.models.medgemma import create_client
    from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

    pipeline = build_coding_pipeline_simple()
    client = create_client()
    agent = MedScribeAgent(
        coding_pipeline=pipeline,
        medgemma_client=client,
        fhir_gateway=None,
    )
    return agent


def render_pipeline_visualization(output: dict, elapsed: float) -> None:
    """Render the 5-step pipeline visualization with progressive expansion."""
    nlp_entities = output["nlp_entities"]
    llm = output["llm_reasoning"]
    discrepancies = output["discrepancies"]
    cds_cards = output["cds_cards"]
    fhir_bundle = output["fhir_bundle"]

    # Step 1: NLP Extraction
    with st.status("Step 1: NLP Entity Extraction", expanded=True, state="complete"):
        st.write(f"Extracted **{len(nlp_entities)}** medical entities via scispaCy")
        if nlp_entities:
            labels = {}
            for e in nlp_entities:
                lbl = e.get("label", "ENTITY")
                labels[lbl] = labels.get(lbl, 0) + 1
            st.write(
                "Entity types: " + ", ".join(f"{k}: {v}" for k, v in labels.items())
            )
    time.sleep(0.3)

    # Step 2: LLM Reasoning
    with st.status("Step 2: MedGemma LLM Reasoning", expanded=True, state="complete"):
        n_dx = len(llm.get("diagnoses", []))
        n_proc = len(llm.get("procedures", []))
        n_med = len(llm.get("medications", []))
        st.write(
            f"Identified **{n_dx}** diagnoses, **{n_proc}** procedures, **{n_med}** medications"
        )
    time.sleep(0.3)

    # Step 3: Dual-Pathway Validation
    with st.status("Step 3: Dual-Pathway Validation", expanded=True, state="complete"):
        st.write(
            f"Found **{len(discrepancies)}** discrepancies between NLP and LLM pathways"
        )
        nlp_only = sum(1 for d in discrepancies if d.get("type") == "nlp_only")
        llm_only = sum(1 for d in discrepancies if d.get("type") == "llm_only")
        if discrepancies:
            st.write(f"NLP-only: {nlp_only} | LLM-only: {llm_only}")
    time.sleep(0.3)

    # Step 4: Tool-Based Gap Resolution
    with st.status(
        "Step 4: Tool-Based Gap Resolution", expanded=True, state="complete"
    ):
        resolved = output.get("resolved", {})
        tool_calls = output.get("tool_calls", [])
        resolved_count = sum(
            1 for dx in resolved.get("diagnoses", []) if dx.get("resolution_reasoning")
        )
        if tool_calls:
            successful = sum(1 for tc in tool_calls if tc.get("success"))
            st.write(
                f"Dispatched **{len(tool_calls)}** tools "
                f"(**{successful}** returned evidence), "
                f"resolved **{resolved_count}** items"
            )
        elif resolved_count > 0:
            st.write(f"Resolved **{resolved_count}** items via re-prompting")
        else:
            st.write("No gaps requiring resolution")
    time.sleep(0.3)

    # Step 5: Output Generation
    with st.status("Step 5: FHIR + CDS Output", expanded=True, state="complete"):
        st.write(f"Generated **{len(cds_cards)}** CDS Hooks cards")
        bundle_status = (
            "FHIR Bundle generated"
            if fhir_bundle
            else "No FHIR Bundle (UMLS linker not installed)"
        )
        st.write(bundle_status)
        st.write(f"Total pipeline time: **{elapsed:.2f}s**")


def render_cds_cards_tab(cds_cards: list) -> None:
    """Render CDS Hooks cards with appropriate severity indicators."""
    if not cds_cards:
        st.info("No CDS cards generated.")
        return

    for card in cds_cards:
        indicator = (
            card.indicator.value
            if hasattr(card.indicator, "value")
            else str(card.indicator)
        )

        if indicator == "critical":
            with st.container(border=True):
                st.error(f"**CRITICAL:** {card.summary}")
                if card.detail:
                    st.write(card.detail)
        elif indicator == "warning":
            with st.container(border=True):
                st.warning(f"**WARNING:** {card.summary}")
                if card.detail:
                    st.write(card.detail)
        else:
            with st.container(border=True):
                st.info(f"**INFO:** {card.summary}")
                if card.detail:
                    st.write(card.detail)


def render_entities_tab(nlp_entities: list) -> None:
    """Render NLP entities as a table."""
    if not nlp_entities:
        st.info("No entities extracted.")
        return

    rows = []
    for e in nlp_entities:
        rows.append(
            {
                "Text": e.get("text", ""),
                "Label": e.get("label", ""),
                "Code": e.get("code", "N/A"),
                "Start": e.get("start", ""),
                "End": e.get("end", ""),
            }
        )
    st.dataframe(rows, use_container_width=True)


def render_reasoning_tab(llm: dict) -> None:
    """Render LLM reasoning: diagnoses, procedures, medications."""
    # Diagnoses
    st.subheader("Diagnoses")
    diagnoses = llm.get("diagnoses", [])
    if diagnoses:
        dx_rows = []
        for dx in diagnoses:
            dx_rows.append(
                {
                    "Condition": dx.get("text", ""),
                    "ICD-10": dx.get("icd10", ""),
                    "Confidence": dx.get("confidence", ""),
                    "Gaps": dx.get("gaps", ""),
                }
            )
        st.dataframe(dx_rows, use_container_width=True)
    else:
        st.write("No diagnoses found.")

    # Procedures
    st.subheader("Procedures")
    procedures = llm.get("procedures", [])
    if procedures:
        proc_rows = []
        for p in procedures:
            proc_rows.append(
                {
                    "Procedure": p.get("text", ""),
                    "CPT": p.get("cpt", ""),
                }
            )
        st.dataframe(proc_rows, use_container_width=True)
    else:
        st.write("No procedures detected.")

    # Medications
    st.subheader("Medications")
    medications = llm.get("medications", [])
    if medications:
        med_rows = []
        for m in medications:
            med_rows.append(
                {
                    "Medication": m.get("text", ""),
                    "Status": m.get("status", ""),
                }
            )
        st.dataframe(med_rows, use_container_width=True)
    else:
        st.write("No medications detected.")


def render_discrepancies_tab(discrepancies: list) -> None:
    """Render NLP vs LLM discrepancy comparison."""
    if not discrepancies:
        st.success("No discrepancies found - NLP and LLM pathways agree.")
        return

    nlp_only = [d for d in discrepancies if d.get("type") == "nlp_only"]
    llm_only = [d for d in discrepancies if d.get("type") == "llm_only"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NLP-Only Findings")
        st.caption("Found by NLP but not confirmed by LLM")
        if nlp_only:
            for d in nlp_only:
                st.write(f"- **{d.get('entity', '')}** (code: {d.get('code', 'N/A')})")
        else:
            st.write("None")

    with col2:
        st.subheader("LLM-Only Findings")
        st.caption("Found by LLM but not detected by NLP")
        if llm_only:
            for d in llm_only:
                st.write(
                    f"- **{d.get('entity', '')}** (ICD-10: {d.get('icd10', 'N/A')}, "
                    f"confidence: {d.get('confidence', 'N/A')})"
                )
        else:
            st.write("None")


def render_agent_trace_tab(tool_calls: list) -> None:
    """Render tool dispatch trace as bordered cards."""
    if not tool_calls:
        st.info("No tools were dispatched during this run.")
        return

    st.write(f"**{len(tool_calls)}** tool(s) dispatched during gap resolution:")

    for i, tc in enumerate(tool_calls):
        with st.container(border=True):
            status_icon = "+" if tc.get("success") else "-"
            st.markdown(
                f"**{status_icon} {tc.get('tool_name', 'unknown')}** — "
                f"Entity: `{tc.get('entity', 'N/A')}`"
            )
            st.caption(tc.get("reason", ""))
            st.write(tc.get("summary", "No summary"))

            if tc.get("data"):
                with st.expander("Raw data"):
                    st.json(tc["data"])


def render_fhir_tab(fhir_bundle) -> None:
    """Render FHIR Bundle as JSON."""
    if fhir_bundle is None:
        st.info(
            "No FHIR Bundle available. The simple pipeline (no UMLS entity linking) "
            "does not generate FHIR Conditions. Install the UMLS knowledge base "
            "for full FHIR output."
        )
        return

    try:
        if hasattr(fhir_bundle, "model_dump"):
            bundle_dict = fhir_bundle.model_dump(exclude_none=True)
        elif hasattr(fhir_bundle, "dict"):
            bundle_dict = fhir_bundle.dict(exclude_none=True)
        else:
            bundle_dict = fhir_bundle
        st.json(bundle_dict)
    except Exception as e:
        st.error(f"Error rendering FHIR Bundle: {e}")


def render_metrics_tab(output: dict, elapsed: float) -> None:
    """Render summary metrics."""
    nlp_entities = output["nlp_entities"]
    llm = output["llm_reasoning"]
    discrepancies = output["discrepancies"]
    cds_cards = output["cds_cards"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("NLP Entities", len(nlp_entities))
    with col2:
        st.metric("Diagnoses", len(llm.get("diagnoses", [])))
    with col3:
        st.metric("Discrepancies", len(discrepancies))
    with col4:
        st.metric("CDS Cards", len(cds_cards))

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Procedures", len(llm.get("procedures", [])))
    with col6:
        st.metric("Medications", len(llm.get("medications", [])))
    with col7:
        gaps = sum(1 for dx in llm.get("diagnoses", []) if dx.get("gaps"))
        st.metric("Coding Gaps", gaps)
    with col8:
        st.metric("Latency", f"{elapsed:.2f}s")

    # Confidence breakdown
    st.subheader("Confidence Distribution")
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for dx in llm.get("diagnoses", []):
        conf = dx.get("confidence", "unknown")
        if conf in conf_counts:
            conf_counts[conf] += 1

    conf_col1, conf_col2, conf_col3 = st.columns(3)
    with conf_col1:
        st.metric("High Confidence", conf_counts["high"])
    with conf_col2:
        st.metric("Medium Confidence", conf_counts["medium"])
    with conf_col3:
        st.metric("Low Confidence", conf_counts["low"])


def main():
    st.set_page_config(
        page_title="MedScribe Agent Demo",
        page_icon="🏥",
        layout="wide",
    )

    st.title("MedScribe Agent")
    st.caption(
        "Agentic Clinical Documentation & Coding Assistant — powered by HealthChain + MedGemma"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        mode = os.environ.get("MEDGEMMA_MODE", "demo")
        st.info(f"Mode: **{mode}**")

        st.subheader("Clinical Note Input")

        sample_choice = st.selectbox(
            "Sample notes",
            options=["Custom"] + list(SAMPLE_NOTES.keys()),
        )

        if sample_choice == "Custom":
            note_text = st.text_area(
                "Enter clinical note",
                height=200,
                placeholder="Paste a clinical note here...",
            )
        else:
            note_text = st.text_area(
                "Clinical note",
                value=SAMPLE_NOTES[sample_choice],
                height=200,
            )

        patient_id = st.text_input("Patient ID", value="Patient/demo-001")

        run_button = st.button("Run Agent", type="primary", use_container_width=True)

    # --- Main area ---
    if run_button and note_text.strip():
        agent = load_agent()

        with st.spinner("Running MedScribe Agent pipeline..."):
            start = time.time()
            output = agent.run(note_text, patient_id)
            elapsed = time.time() - start

        # Pipeline visualization
        st.header("Pipeline Execution")
        render_pipeline_visualization(output, elapsed)

        st.divider()

        # Results tabs
        st.header("Results")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "CDS Cards",
                "NLP Entities",
                "LLM Reasoning",
                "Discrepancies",
                "Agent Trace",
                "FHIR Bundle",
                "Metrics",
            ]
        )

        with tab1:
            render_cds_cards_tab(output["cds_cards"])
        with tab2:
            render_entities_tab(output["nlp_entities"])
        with tab3:
            render_reasoning_tab(output["llm_reasoning"])
        with tab4:
            render_discrepancies_tab(output["discrepancies"])
        with tab5:
            render_agent_trace_tab(output.get("tool_calls", []))
        with tab6:
            render_fhir_tab(output["fhir_bundle"])
        with tab7:
            render_metrics_tab(output, elapsed)

    elif run_button:
        st.warning("Please enter a clinical note before running the agent.")
    else:
        st.info(
            "Select a sample note or enter a custom note in the sidebar, then click **Run Agent**."
        )


if __name__ == "__main__":
    main()
