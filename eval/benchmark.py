"""Coding accuracy, latency metrics, and baseline comparison benchmarks."""

import json
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    note_id: str
    predicted_codes: list[str] = field(default_factory=list)
    ground_truth_codes: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    discrepancies_found: int = 0
    gaps_detected: int = 0


def calculate_f1(predicted: set[str], ground_truth: set[str]) -> dict:
    """Calculate precision, recall, and F1 for ICD-10 coding."""
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = predicted & ground_truth
    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def run_benchmark(agent, test_cases: list[dict]) -> list[BenchmarkResult]:
    """Run the agent on a set of test cases and collect metrics.

    Args:
        agent: MedScribeAgent instance.
        test_cases: List of dicts with keys: note_id, clinical_note, patient_id, ground_truth_codes.

    Returns:
        List of BenchmarkResult objects.
    """
    results = []
    for case in test_cases:
        start = time.time()
        output = agent.run(case["clinical_note"], case["patient_id"])
        elapsed = time.time() - start

        predicted_codes = [
            dx.get("icd10", "")
            for dx in output.get("llm_reasoning", {}).get("diagnoses", [])
        ]

        result = BenchmarkResult(
            note_id=case["note_id"],
            predicted_codes=predicted_codes,
            ground_truth_codes=case.get("ground_truth_codes", []),
            latency_seconds=elapsed,
            discrepancies_found=len(output.get("discrepancies", [])),
            gaps_detected=sum(
                1
                for dx in output.get("llm_reasoning", {}).get("diagnoses", [])
                if dx.get("gaps")
            ),
        )
        results.append(result)

    return results


def load_test_cases(path: str | Path | None = None) -> list[dict]:
    """Load test cases from a JSON file.

    Args:
        path: Path to test_cases.json. Defaults to eval/test_cases.json.

    Returns:
        List of test case dicts.
    """
    if path is None:
        path = Path(__file__).parent / "test_cases.json"
    else:
        path = Path(path)

    with open(path) as f:
        return json.load(f)


def calculate_aggregate_metrics(results: list[BenchmarkResult]) -> dict:
    """Calculate aggregate metrics across all benchmark results.

    Returns:
        Dict with mean_f1, mean_precision, mean_recall, mean_latency,
        total_cases, and per-case F1 scores.
    """
    per_case = []
    for r in results:
        metrics = calculate_f1(set(r.predicted_codes), set(r.ground_truth_codes))
        per_case.append(
            {
                "note_id": r.note_id,
                **metrics,
                "latency": r.latency_seconds,
                "discrepancies": r.discrepancies_found,
                "gaps": r.gaps_detected,
            }
        )

    n = len(per_case)
    if n == 0:
        return {
            "mean_f1": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_latency": 0.0,
            "total_cases": 0,
            "per_case": [],
        }

    return {
        "mean_f1": sum(c["f1"] for c in per_case) / n,
        "mean_precision": sum(c["precision"] for c in per_case) / n,
        "mean_recall": sum(c["recall"] for c in per_case) / n,
        "mean_latency": sum(c["latency"] for c in per_case) / n,
        "total_cases": n,
        "per_case": per_case,
    }


def print_benchmark_summary(results: list[BenchmarkResult], metrics: dict) -> None:
    """Print formatted benchmark results table."""
    print("\n" + "=" * 80)
    print("MedScribe Agent - Benchmark Results")
    print("=" * 80)

    # Per-case table
    header = f"{'Note ID':<10} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Lat(s)':>8} {'Disc':>5} {'Gaps':>5} {'Predicted':>10} {'GT':>4}"
    print(f"\n{header}")
    print("-" * 80)

    for i, case in enumerate(metrics["per_case"]):
        r = results[i]
        print(
            f"{case['note_id']:<10} "
            f"{case['f1']:>6.3f} "
            f"{case['precision']:>6.3f} "
            f"{case['recall']:>6.3f} "
            f"{case['latency']:>8.3f} "
            f"{case['discrepancies']:>5d} "
            f"{case['gaps']:>5d} "
            f"{len(r.predicted_codes):>10d} "
            f"{len(r.ground_truth_codes):>4d}"
        )

    # Aggregate summary
    print("-" * 80)
    print(
        f"{'MEAN':<10} "
        f"{metrics['mean_f1']:>6.3f} "
        f"{metrics['mean_precision']:>6.3f} "
        f"{metrics['mean_recall']:>6.3f} "
        f"{metrics['mean_latency']:>8.3f}"
    )
    print(f"\nTotal test cases: {metrics['total_cases']}")
    print("=" * 80)


if __name__ == "__main__":
    # Ensure project root is on path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from medscribe.src.agent.orchestrator import MedScribeAgent
    from medscribe.src.models.medgemma import create_client
    from medscribe.src.pipeline.coding_pipeline import build_coding_pipeline_simple

    print("Loading pipeline and model...")
    pipeline = build_coding_pipeline_simple()
    client = create_client()

    agent = MedScribeAgent(
        coding_pipeline=pipeline,
        medgemma_client=client,
        fhir_gateway=None,
    )

    print("Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")

    print("Running benchmark...")
    results = run_benchmark(agent, test_cases)

    metrics = calculate_aggregate_metrics(results)
    print_benchmark_summary(results, metrics)
