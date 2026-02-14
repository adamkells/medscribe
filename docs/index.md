# MedScribe Agent Documentation

MedScribe is an agentic clinical documentation and ICD-10 coding system built on [HealthChain](https://github.com/dotimplement/HealthChain). It combines biomedical NLP (scispaCy), LLM reasoning (MedGemma 4B), and structured tool dispatch to automate medical coding with physician-reviewable output.

## How It Works

MedScribe runs a 5-step agentic pipeline on every clinical note:

1. **NLP Extraction** — scispaCy identifies medical entities (conditions, medications, procedures)
2. **LLM Reasoning** — MedGemma assigns ICD-10/CPT codes with confidence scores
3. **Dual-Pathway Validation** — compares NLP and LLM findings to detect discrepancies
4. **Tool-Based Gap Resolution** — dispatches HealthChain tools to gather patient evidence, then re-prompts the LLM
5. **Output Generation** — produces a FHIR Bundle and CDS Hooks cards for physician review

The dual-pathway design ensures coding accuracy beyond what either NLP or LLM achieves alone. The structured tool dispatch (Step 4) is the key differentiator — the agent uses Python logic (not LLM-chosen tool calls) to select which evidence tools to run based on discrepancy type.

## Documentation

| Page | Description |
|------|-------------|
| [Getting Started](getting-started.md) | Installation, environment setup, running the demo |
| [Architecture](architecture.md) | Pipeline flowchart, component mapping, data flow |
| [Agent](agent.md) | Orchestrator, validator, tool registry, gap resolution |
| [Pipeline](pipeline.md) | NLP pipeline configuration (scispaCy, entity linking) |
| [Models](models.md) | MedGemma client modes (demo, endpoint, local, Vertex) |
| [Gateway](gateway.md) | EHR integration via SOAP/CDA (NoteReader) and CDS Hooks |
| [Demo](demo.md) | Streamlit interactive demo application |
| [Evaluation](evaluation.md) | Benchmark framework, test cases, metrics |
| [Configuration](configuration.md) | Environment variables and prompt templates |

## Project Structure

```
medscribe/
├── demo/
│   └── app.py                  # Streamlit demo UI
├── docs/                       # This documentation
├── eval/
│   ├── benchmark.py            # Evaluation framework
│   └── test_cases.json         # 10 clinical test cases
├── sandbox_tests/
│   └── test_integration.py     # Integration test suite
├── src/
│   ├── agent/
│   │   ├── orchestrator.py     # MedScribeAgent (main entrypoint)
│   │   ├── validator.py        # DualPathwayValidator
│   │   ├── tools.py            # ToolRegistry + ToolResult
│   │   └── gap_filler.py       # Legacy FHIR gap filler
│   ├── gateway/
│   │   └── app.py              # HealthChainAPI factory (SOAP + CDS)
│   ├── models/
│   │   ├── medgemma.py         # LLM client implementations
│   │   └── prompts/
│   │       └── reasoning.yaml  # System prompts for LLM
│   └── pipeline/
│       └── coding_pipeline.py  # scispaCy NLP pipeline builders
├── pyproject.toml              # Dependencies and packaging
├── STATUS.md                   # Current project status
└── TASKS.md                    # Implementation roadmap
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `healthchain` | FHIR utilities, pipeline framework, gateway infrastructure |
| `scispacy` + `en_core_sci_sm` | Biomedical named entity recognition |
| `transformers` | MedGemma model loading (local mode) |
| `bitsandbytes` | 4-bit quantization (local GPU mode) |
| `streamlit` | Interactive demo UI |
| `pyyaml` | Prompt template loading |
