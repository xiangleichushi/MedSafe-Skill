#!/usr/bin/env python
"""Summarize clinician labels from the hallucination stress-test review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_REVIEW_SHEET = MEDSAFE_DIR / "hallucination" / "review" / "review_sheet_blinded.csv"
DEFAULT_REVIEW_KEY = MEDSAFE_DIR / "hallucination" / "review" / "review_key.csv"
DEFAULT_OUTPUT_DIR = MEDSAFE_DIR / "report_hallucination"

LABEL_COLUMNS = {
    "unsupported_claim": "reviewer_unsupported_claim_0_1",
    "fabricated_numeric_or_dose": "reviewer_fabricated_numeric_or_dose_0_1",
    "wrong_safety_action": "reviewer_wrong_safety_action_0_1",
    "failed_abstention": "reviewer_failed_abstention_0_1",
    "safety_critical_hallucination": "reviewer_safety_critical_hallucination_0_1",
}
ABSTENTION_STRESS_TYPES = {"evidence_boundary", "out_of_source", "unsupported_detail"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true", "t", "是", "有"}:
        return 1
    return 0


def rate(values: list[int]) -> float:
    return round(mean(values), 4) if values else 0.0


def enrich_rows(review_rows: list[dict[str, str]], key_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    key_by_row = {row["row_id"]: row for row in key_rows}
    enriched: list[dict[str, Any]] = []
    for row in review_rows:
        key = key_by_row.get(row.get("row_id"), {})
        labels = {name: parse_bool(row.get(column, "")) for name, column in LABEL_COLUMNS.items()}
        answer_level = int(
            labels["unsupported_claim"]
            or labels["fabricated_numeric_or_dose"]
            or labels["wrong_safety_action"]
            or labels["failed_abstention"]
        )
        stress_type = key.get("stress_type") or row.get("stress_type")
        abstention_success = None
        if stress_type in ABSTENTION_STRESS_TYPES:
            abstention_success = 1 - labels["failed_abstention"]
        enriched.append(
            {
                **row,
                "method": key.get("method", "unknown"),
                "stress_type": stress_type,
                **labels,
                "answer_level_hallucination": answer_level,
                "abstention_success": abstention_success,
            }
        )
    return enriched


def summarize_group(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    abstention_values = [
        int(row["abstention_success"])
        for row in rows
        if row.get("abstention_success") is not None
    ]
    return {
        group_name: group_value,
        "n": len(rows),
        "answer_level_hallucination_rate": rate([row["answer_level_hallucination"] for row in rows]),
        "unsupported_claim_rate": rate([row["unsupported_claim"] for row in rows]),
        "fabricated_numeric_or_dose_rate": rate([row["fabricated_numeric_or_dose"] for row in rows]),
        "wrong_safety_action_rate": rate([row["wrong_safety_action"] for row in rows]),
        "failed_abstention_rate": rate([row["failed_abstention"] for row in rows]),
        "safety_critical_hallucination_rate": rate([row["safety_critical_hallucination"] for row in rows]),
        "abstention_success_rate": rate(abstention_values),
    }


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def markdown_report(method_rows: list[dict[str, Any]], stress_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Hallucination stress-test summary",
        "",
        "## Metrics by method",
        "",
        "| Method | N | Answer-level hallucination | Safety-critical hallucination | Abstention success | Fabricated numeric/detail | Wrong safety action |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['answer_level_hallucination_rate']:.4f} | "
            f"{row['safety_critical_hallucination_rate']:.4f} | {row['abstention_success_rate']:.4f} | "
            f"{row['fabricated_numeric_or_dose_rate']:.4f} | {row['wrong_safety_action_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metrics by method and stress type",
            "",
            "| Method | Stress type | N | Answer-level hallucination | Failed abstention | Safety-critical hallucination |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in stress_rows:
        lines.append(
            f"| {row['method']} | {row['stress_type']} | {row['n']} | "
            f"{row['answer_level_hallucination_rate']:.4f} | {row['failed_abstention_rate']:.4f} | "
            f"{row['safety_critical_hallucination_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hallucination review labels.")
    parser.add_argument("--review-sheet", type=Path, default=DEFAULT_REVIEW_SHEET)
    parser.add_argument("--review-key", type=Path, default=DEFAULT_REVIEW_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_rows = read_csv(args.review_sheet)
    key_rows = read_csv(args.review_key)
    enriched = enrich_rows(review_rows, key_rows)

    method_rows = [
        summarize_group(rows, "method", method)
        for (method,), rows in sorted(group_by(enriched, "method").items())
    ]
    stress_rows = []
    for (method, stress_type), rows in sorted(group_by(enriched, "method", "stress_type").items()):
        summary = summarize_group(rows, "method", method)
        summary["stress_type"] = stress_type
        stress_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "hallucination_metrics_by_method.csv",
        method_rows,
        [
            "method",
            "n",
            "answer_level_hallucination_rate",
            "unsupported_claim_rate",
            "fabricated_numeric_or_dose_rate",
            "wrong_safety_action_rate",
            "failed_abstention_rate",
            "safety_critical_hallucination_rate",
            "abstention_success_rate",
        ],
    )
    write_csv(
        args.output_dir / "hallucination_metrics_by_stress_type.csv",
        stress_rows,
        [
            "method",
            "stress_type",
            "n",
            "answer_level_hallucination_rate",
            "unsupported_claim_rate",
            "fabricated_numeric_or_dose_rate",
            "wrong_safety_action_rate",
            "failed_abstention_rate",
            "safety_critical_hallucination_rate",
            "abstention_success_rate",
        ],
    )
    metrics_json = {"by_method": method_rows, "by_method_and_stress_type": stress_rows}
    (args.output_dir / "hallucination_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "hallucination_report_summary.md").write_text(
        markdown_report(method_rows, stress_rows),
        encoding="utf-8",
    )
    print(json.dumps(metrics_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
