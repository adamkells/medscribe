# Evaluation

**Module:** `medscribe.eval`

MedScribe includes a benchmark framework for measuring ICD-10 coding accuracy across clinical scenarios.

## Running the Benchmark

```bash
# With demo client (keyword-matching baseline)
MEDGEMMA_MODE=demo uv run python -m medscribe.eval.benchmark

# With HuggingFace Inference Endpoint
MEDGEMMA_MODE=endpoint uv run python -m medscribe.eval.benchmark

# With local GPU
MEDGEMMA_MODE=real uv run python -m medscribe.eval.benchmark
```

## Test Cases

**File:** `medscribe/eval/test_cases.json`

10 test cases covering 7 clinical specialties:

| ID | Scenario | Ground Truth | Specialty |
|----|----------|-------------|-----------|
| TC-001 | Pneumonia + T2DM + HTN | J18.9, E11.9, I10 | Inpatient Medicine |
| TC-002 | Heart Failure + AFib | I50.9, I48.91 | Cardiology |
| TC-003 | COPD + Chest Pain + HTN + Hyperlipidemia | J44.1, R07.9, I10, E78.5 | Pulmonology |
| TC-004 | Sepsis + AKI + UTI + T2DM | A41.9, N17.9, N39.0, E11.9 | Critical Care |
| TC-005 | Depression + Anxiety + Hypothyroidism | F32.9, F41.9, E03.9 | Behavioral Health |
| TC-006 | Anemia + Hyperlipidemia | D64.9, E78.5 | Outpatient |
| TC-007 | UTI + T1DM + HTN | N39.0, E10.9, I10 | General Medicine |
| TC-008 | Pneumonia only | J18.9 | Single Condition |
| TC-009 | Complex Geriatric (6 conditions) | E11.9, I10, J44.1, I50.9, D64.9, E03.9 | Geriatrics |
| TC-010 | Minimal vague note | (empty) | Edge Case |

Each test case contains:

```json
{
    "note_id": "TC-001",
    "clinical_note": "Patient is a 65-year-old male admitted for...",
    "patient_id": "Patient/tc-001",
    "ground_truth_codes": ["J18.9", "E11.9", "I10"],
    "description": "Pneumonia + T2DM + HTN"
}
```

## Metrics

### Per-Case Metrics

For each test case, the benchmark computes:

- **Precision:** What fraction of predicted codes are correct
- **Recall:** What fraction of ground truth codes were found
- **F1 Score:** Harmonic mean of precision and recall
- **Latency:** Wall-clock time for `agent.run()`
- **Discrepancies:** Number of NLP vs LLM discrepancies detected
- **Gaps:** Number of diagnoses with documentation gaps

### Aggregate Metrics

- **Mean F1 / Precision / Recall** across all test cases
- **Mean Latency** per case
- **Total Cases** processed

### F1 Calculation

```python
def calculate_f1(predicted: list[str], ground_truth: list[str]) -> dict:
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
```

Special case: if both predicted and ground truth are empty, all metrics return 1.0 (correct prediction of no codes).

## BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    note_id: str
    predicted_codes: list[str]       # ICD-10 codes from llm_reasoning.diagnoses
    ground_truth_codes: list[str]    # From test_cases.json
    latency_seconds: float
    discrepancies_found: int
    gaps_detected: int               # Diagnoses with non-empty "gaps" field
```

## Output Format

The benchmark prints an 80-character summary table:

```
================================================================================
MedScribe Benchmark Results
================================================================================
Note ID  | F1    | Prec  | Rec   | Lat(s)| Disc | Gaps | Predicted          | GT
---------|-------|-------|-------|-------|------|------|--------------------|-----
TC-001   | 1.000 | 1.000 | 1.000 |  1.23 |    5 |    1 | E11.9,I10,J18.9    | E11.9,I10,J18.9
TC-002   | 0.667 | 0.500 | 1.000 |  0.98 |    3 |    1 | I48.91,I50.9,...    | I48.91,I50.9
...
---------|-------|-------|-------|-------|------|------|--------------------|-----
MEAN     | 0.727 | 0.650 | 0.850 |  1.05 |      |      |                    |
================================================================================
```

## Demo Baseline

With `MEDGEMMA_MODE=demo` (keyword matching, no real LLM), the benchmark achieves:

- **Mean F1:** ~0.727
- **Mean Precision:** ~0.650
- **Mean Recall:** ~0.850

The precision gap is expected — the demo client assigns codes to all NLP entities (including non-diagnostic ones like "patient", "admitted"), inflating the prediction set. The real MedGemma client produces significantly higher precision.

## Adding Test Cases

Add new entries to `medscribe/eval/test_cases.json`:

```json
{
    "note_id": "TC-011",
    "clinical_note": "Your clinical note text...",
    "patient_id": "Patient/tc-011",
    "ground_truth_codes": ["I10", "E11.9"],
    "description": "Brief description"
}
```

Ground truth codes should be the ICD-10-CM codes a human coder would assign.
