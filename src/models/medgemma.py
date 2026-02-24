"""MedGemma client via HuggingFace Transformers, Vertex AI, or demo fallback."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Protocol

import yaml

logger = logging.getLogger(__name__)

# Load prompt templates from YAML
_PROMPTS_PATH = Path(__file__).parent / "prompts" / "reasoning.yaml"
with open(_PROMPTS_PATH) as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)

REASONING_SYSTEM_PROMPT = PROMPT_TEMPLATES["reasoning_system"]["template"]
RESOLUTION_SYSTEM_PROMPT = PROMPT_TEMPLATES["resolution_system"]["template"]


class MedGemmaClientProtocol(Protocol):
    """Protocol for MedGemma clients (real and demo)."""

    def reason_over_note(
        self, note_text: str, extracted_entities: list[dict] | None = None
    ) -> dict: ...

    def reason_with_resolution(
        self,
        note_text: str,
        discrepancies: list[dict],
        patient_history: str = "",
    ) -> dict: ...


def _parse_json_response(response: str, fallback: dict) -> dict:
    """Extract and parse JSON from model response.

    Handles raw JSON, markdown code blocks, JSON arrays, and JSON embedded in text.
    """
    # Strip markdown code fences if present
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        # If model returned an array, wrap it as diagnoses
        if isinstance(parsed, list):
            return {"diagnoses": parsed, "procedures": [], "medications": [], "discrepancies": []}
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON object from mixed text
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract JSON array from mixed text
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return {"diagnoses": parsed, "procedures": [], "medications": [], "discrepancies": []}
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse MedGemma response as JSON: %s", response[:500])
    return fallback


class MedGemmaClient:
    """Local MedGemma inference via HuggingFace Transformers with 4-bit quantization."""

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        device: str = "cuda",
        max_new_tokens: int = 2048,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info("Loading MedGemma model: %s (4-bit NF4 quantized)", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logger.info("MedGemma model loaded successfully on %s", self.model.device)

    def _generate(self, messages: list[dict]) -> str:
        """Generate a response from chat messages using the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
        )
        # Decode only the newly generated tokens
        return self.tokenizer.decode(
            outputs[0][inputs.shape[-1] :], skip_special_tokens=True
        )

    def reason_over_note(
        self, note_text: str, extracted_entities: list[dict] | None = None
    ) -> dict:
        """Send clinical note to MedGemma for structured reasoning."""
        entity_context = ""
        if extracted_entities:
            entity_context = (
                "\n\nNLP-extracted entities for cross-reference:\n"
                + "\n".join(
                    f"- {e['text']} ({e.get('code', 'no code')})"
                    for e in extracted_entities
                )
            )

        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Clinical Note:\n{note_text}{entity_context}",
            },
        ]

        raw = self._generate(messages)
        return _parse_json_response(
            raw,
            {"diagnoses": [], "procedures": [], "medications": [], "discrepancies": []},
        )

    def reason_with_resolution(
        self,
        note_text: str,
        discrepancies: list[dict],
        patient_history: str = "",
    ) -> dict:
        """Re-prompt MedGemma to resolve discrepancies with patient context."""
        user_content = (
            f"Original Clinical Note:\n{note_text}\n\n"
            f"Discrepancies to resolve:\n{json.dumps(discrepancies, indent=2)}"
        )
        if patient_history:
            user_content += f"\n\nRelevant Patient History (FHIR):\n{patient_history}"

        messages = [
            {"role": "system", "content": RESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages)
        return _parse_json_response(raw, {"resolved": []})


class VertexMedGemmaClient:
    """MedGemma inference via Vertex AI Model Garden endpoint (OpenAI-compatible API).

    Requires:
        - google-auth: for GCP credentials
        - openai: for the OpenAI-compatible chat completions API
        - A deployed MedGemma endpoint on Vertex AI Model Garden

    Environment variables:
        VERTEX_ENDPOINT_ID: The Vertex AI endpoint ID
        VERTEX_PROJECT_ID: Your GCP project ID
        VERTEX_REGION: Endpoint region (default: us-central1)
    """

    def __init__(
        self,
        endpoint_id: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        max_tokens: int = 2048,
    ):
        import google.auth
        import google.auth.transport.requests
        import openai

        self.max_tokens = max_tokens

        self.endpoint_id = endpoint_id or os.environ["VERTEX_ENDPOINT_ID"]
        self.project_id = project_id or os.environ["VERTEX_PROJECT_ID"]
        self.region = region or os.environ.get("VERTEX_REGION", "us-central1")

        self._credentials, _ = google.auth.default()
        self._auth_request = google.auth.transport.requests.Request()

        base_url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"endpoints/{self.endpoint_id}"
        )
        self._client = openai.OpenAI(base_url=base_url, api_key=self._get_token())

        logger.info(
            "Vertex MedGemma client: endpoint=%s region=%s",
            self.endpoint_id,
            self.region,
        )

    def _get_token(self) -> str:
        """Get a fresh GCP access token."""
        self._credentials.refresh(self._auth_request)
        return self._credentials.token

    def _generate(self, messages: list[dict]) -> str:
        """Generate a response via the Vertex AI endpoint."""
        # Refresh token before each call (tokens expire after ~1 hour)
        self._client.api_key = self._get_token()

        response = self._client.chat.completions.create(
            model="",
            messages=messages,
            max_completion_tokens=self.max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def reason_over_note(
        self, note_text: str, extracted_entities: list[dict] | None = None
    ) -> dict:
        """Send clinical note to MedGemma for structured reasoning."""
        entity_context = ""
        if extracted_entities:
            entity_context = (
                "\n\nNLP-extracted entities for cross-reference:\n"
                + "\n".join(
                    f"- {e['text']} ({e.get('code', 'no code')})"
                    for e in extracted_entities
                )
            )

        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Clinical Note:\n{note_text}{entity_context}",
            },
        ]

        raw = self._generate(messages)
        return _parse_json_response(
            raw,
            {"diagnoses": [], "procedures": [], "medications": [], "discrepancies": []},
        )

    def reason_with_resolution(
        self,
        note_text: str,
        discrepancies: list[dict],
        patient_history: str = "",
    ) -> dict:
        """Re-prompt MedGemma to resolve discrepancies with patient context."""
        user_content = (
            f"Original Clinical Note:\n{note_text}\n\n"
            f"Discrepancies to resolve:\n{json.dumps(discrepancies, indent=2)}"
        )
        if patient_history:
            user_content += f"\n\nRelevant Patient History (FHIR):\n{patient_history}"

        messages = [
            {"role": "system", "content": RESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages)
        return _parse_json_response(raw, {"resolved": []})


class HFEndpointMedGemmaClient:
    """MedGemma inference via HuggingFace Inference Endpoints.

    Uses the HF Inference API with Gemma chat template formatting.
    Supports both basic TGI endpoints (POST to root with inputs/parameters)
    and OpenAI-compatible endpoints (vLLM /v1/chat/completions).

    Environment variables:
        HF_ENDPOINT_URL: The full endpoint URL (e.g. https://{id}.{region}.aws.endpoints.huggingface.cloud)
        HF_TOKEN: HuggingFace API token for authentication
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_token: str | None = None,
        max_tokens: int = 512,
    ):
        import requests as _requests

        self._requests = _requests
        self.max_tokens = max_tokens

        self.endpoint_url = (endpoint_url or os.environ["HF_ENDPOINT_URL"]).rstrip("/")
        self._api_token = api_token or os.environ["HF_TOKEN"]
        self._headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }

        logger.info("HF Endpoint MedGemma client: url=%s", self.endpoint_url)

    @staticmethod
    def _apply_chat_template(messages: list[dict]) -> str:
        """Format messages using Gemma chat template.

        Gemma uses <start_of_turn>role\\n...content...<end_of_turn> format.
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # Gemma maps "system" to "user" turn with system instruction
            if role == "system":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        # Add generation prompt for model to complete
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    def _generate(self, messages: list[dict]) -> str:
        """Generate a response via the HF Inference Endpoint."""
        prompt = self._apply_chat_template(messages)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": 0.3,
                "return_full_text": False,
            },
        }

        for attempt in range(3):
            try:
                response = self._requests.post(
                    self.endpoint_url, headers=self._headers, json=payload, timeout=300
                )
                response.raise_for_status()

                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return ""
            except self._requests.exceptions.ReadTimeout:
                if attempt < 2:
                    logger.warning(
                        "HF Endpoint timed out (attempt %d/3), retrying...",
                        attempt + 1,
                    )
                    continue
                raise

    def reason_over_note(
        self, note_text: str, extracted_entities: list[dict] | None = None
    ) -> dict:
        """Send clinical note to MedGemma for structured reasoning."""
        entity_context = ""
        if extracted_entities:
            entity_context = (
                "\n\nNLP-extracted entities for cross-reference:\n"
                + "\n".join(
                    f"- {e['text']} ({e.get('code', 'no code')})"
                    for e in extracted_entities
                )
            )

        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Clinical Note:\n{note_text}{entity_context}",
            },
        ]

        raw = self._generate(messages)
        return _parse_json_response(
            raw,
            {"diagnoses": [], "procedures": [], "medications": [], "discrepancies": []},
        )

    def reason_with_resolution(
        self,
        note_text: str,
        discrepancies: list[dict],
        patient_history: str = "",
    ) -> dict:
        """Re-prompt MedGemma to resolve discrepancies with patient context."""
        user_content = (
            f"Original Clinical Note:\n{note_text}\n\n"
            f"Discrepancies to resolve:\n{json.dumps(discrepancies, indent=2)}"
        )
        if patient_history:
            user_content += f"\n\nRelevant Patient History (FHIR):\n{patient_history}"

        messages = [
            {"role": "system", "content": RESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages)
        return _parse_json_response(raw, {"resolved": []})


class DemoMedGemmaClient:
    """Demo client that returns realistic mock responses without Vertex AI."""

    def reason_over_note(
        self, note_text: str, extracted_entities: list[dict] | None = None
    ) -> dict:
        """Return a realistic mock coding response based on note content."""
        note_lower = note_text.lower()

        diagnoses = []
        procedures = []
        medications = []

        # Detect common clinical conditions in the note
        condition_map = {
            "pneumonia": ("J18.9", "high", ""),
            "community-acquired pneumonia": ("J18.9", "high", ""),
            "hypertension": ("I10", "high", ""),
            "diabetes": (
                "E11.9",
                "medium",
                "Consider specifying type and complications",
            ),
            "type 2 diabetes": ("E11.9", "high", ""),
            "type 1 diabetes": ("E10.9", "high", ""),
            "copd": ("J44.1", "high", ""),
            "chronic obstructive pulmonary disease": ("J44.1", "high", ""),
            "heart failure": ("I50.9", "medium", "Specify systolic vs diastolic"),
            "congestive heart failure": (
                "I50.9",
                "medium",
                "Specify systolic vs diastolic",
            ),
            "atrial fibrillation": ("I48.91", "high", ""),
            "chest pain": ("R07.9", "medium", "Consider specifying location"),
            "acute kidney injury": ("N17.9", "high", ""),
            "sepsis": ("A41.9", "high", ""),
            "urinary tract infection": ("N39.0", "high", ""),
            "anemia": (
                "D64.9",
                "low",
                "Specify type: iron deficiency, chronic disease, etc.",
            ),
            "hyperlipidemia": ("E78.5", "high", ""),
            "hypothyroidism": ("E03.9", "high", ""),
            "anxiety": ("F41.9", "high", ""),
            "depression": ("F32.9", "medium", "Specify severity"),
        }

        for condition, (icd10, confidence, gaps) in condition_map.items():
            if condition in note_lower:
                diagnoses.append(
                    {
                        "text": condition.title(),
                        "icd10": icd10,
                        "confidence": confidence,
                        "gaps": gaps,
                    }
                )

        # If nothing matched, provide a default response
        if not diagnoses:
            diagnoses = [
                {
                    "text": "Unspecified condition",
                    "icd10": "R69",
                    "confidence": "low",
                    "gaps": "Clinical note lacks specific diagnostic terms",
                }
            ]

        # Detect common medications
        med_map = {
            "metformin": "Metformin 500mg",
            "lisinopril": "Lisinopril 10mg",
            "amlodipine": "Amlodipine 5mg",
            "atorvastatin": "Atorvastatin 20mg",
            "aspirin": "Aspirin 81mg",
            "metoprolol": "Metoprolol 25mg",
            "insulin": "Insulin (type unspecified)",
            "levothyroxine": "Levothyroxine 50mcg",
            "omeprazole": "Omeprazole 20mg",
            "amoxicillin": "Amoxicillin 500mg",
            "azithromycin": "Azithromycin 250mg",
        }
        for med_key, med_name in med_map.items():
            if med_key in note_lower:
                medications.append({"text": med_name, "status": "active"})

        # Detect procedures
        proc_map = {
            "chest x-ray": ("71046", "Chest X-ray"),
            "ct scan": ("74177", "CT scan"),
            "echocardiogram": ("93306", "Echocardiogram"),
            "ekg": ("93000", "EKG"),
            "blood culture": ("87040", "Blood culture"),
            "urinalysis": ("81001", "Urinalysis"),
        }
        for proc_key, (cpt, proc_name) in proc_map.items():
            if proc_key in note_lower:
                procedures.append({"text": proc_name, "cpt": cpt})

        # Generate discrepancies if NLP entities were provided
        discrepancies = []
        if extracted_entities:
            nlp_texts = {e.get("text", "").lower() for e in extracted_entities}
            llm_texts = {d["text"].lower() for d in diagnoses}
            for nlp_text in nlp_texts - llm_texts:
                discrepancies.append(
                    {
                        "type": "nlp_only",
                        "entity": nlp_text,
                        "reason": "Found by NLP but not confirmed by LLM reasoning",
                    }
                )

        logger.info(
            "Demo response: %d diagnoses, %d procedures, %d medications",
            len(diagnoses),
            len(procedures),
            len(medications),
        )
        return {
            "diagnoses": diagnoses,
            "procedures": procedures,
            "medications": medications,
            "discrepancies": discrepancies,
        }

    def reason_with_resolution(
        self,
        note_text: str,
        discrepancies: list[dict],
        patient_history: str = "",
    ) -> dict:
        """Return mock resolution for discrepancies."""
        resolved = []
        for disc in discrepancies:
            resolved.append(
                {
                    "text": disc.get("entity", "unknown"),
                    "code": "R69",
                    "reasoning": f"Resolved via patient history review: {disc.get('reason', '')}",
                    "confidence": "medium",
                }
            )
        return {"resolved": resolved}


def create_client(mode: str | None = None) -> MedGemmaClientProtocol:
    """Factory to create the appropriate MedGemma client.

    Modes:
        - "demo": Always use DemoMedGemmaClient (no GPU needed)
        - "real": Always use MedGemmaClient (requires CUDA GPU + model access)
        - "endpoint": Use HFEndpointMedGemmaClient (HuggingFace Inference Endpoints)
        - "vertex": Use VertexMedGemmaClient (Vertex AI endpoint, no local GPU)
        - "auto" (default): Use endpoint if HF_ENDPOINT_URL is set, else real if
          CUDA is available, else demo

    Environment variables:
        MEDGEMMA_MODE: Override mode selection (demo/real/endpoint/vertex/auto)
        MEDGEMMA_MODEL: HuggingFace model ID (default: google/medgemma-4b-it)
        HF_ENDPOINT_URL: HuggingFace Inference Endpoint URL (for endpoint mode)
        HF_TOKEN: HuggingFace API token (for endpoint mode)
        VERTEX_ENDPOINT_ID: Vertex AI endpoint ID (required for vertex mode)
        VERTEX_PROJECT_ID: GCP project ID (required for vertex mode)
        VERTEX_REGION: Endpoint region (default: us-central1)
    """
    if mode is None:
        mode = os.environ.get("MEDGEMMA_MODE", "auto")

    if mode == "demo":
        logger.info("Using demo MedGemma client (mode=demo)")
        return DemoMedGemmaClient()

    if mode == "endpoint":
        logger.info("Creating HF Inference Endpoint MedGemma client")
        return HFEndpointMedGemmaClient()

    if mode == "vertex":
        logger.info("Creating Vertex AI MedGemma client")
        return VertexMedGemmaClient()

    if mode == "real":
        model_id = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
        logger.info("Creating real MedGemma client (model=%s)", model_id)
        return MedGemmaClient(model_id=model_id)

    # Auto mode: prefer endpoint if configured, then local GPU, then demo
    if os.environ.get("HF_ENDPOINT_URL"):
        logger.info("HF_ENDPOINT_URL found, creating HF Inference Endpoint client")
        return HFEndpointMedGemmaClient()

    try:
        import torch

        if torch.cuda.is_available():
            model_id = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
            logger.info(
                "CUDA available, creating real MedGemma client (model=%s)", model_id
            )
            return MedGemmaClient(model_id=model_id)
    except ImportError:
        pass

    logger.info("No CUDA available, falling back to demo MedGemma client")
    return DemoMedGemmaClient()
