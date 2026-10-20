#!/usr/bin/env python
"""Build an independent review sheet for the draft Skill benchmark key."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def instruction_text() -> str:
    return """# Skill benchmark key review

This review is completed before running or inspecting the final answer experiment. It establishes an independent key for routing and expected behavior.

Two perspectives are recommended:

- A clinical reviewer validates the reproductive context, target drug(s), missing clinical information, unsafe premise, and expected behavior.
- A system-method reviewer validates the required executable skills and requested evidence fact types.

For each row:

1. Enter `labels_acceptable_0_1 = 1` when all proposed labels are acceptable.
2. If any label is wrong, enter 0 and complete the corresponding `reviewer_*` fields.
3. Array/object fields must use JSON, for example `["incidence","sample_size"]`.
4. `reviewer_context` must be pregnancy, preconception, lactation, pregnancy_and_lactation, or unspecified.
5. `reviewer_required_skills_json` may use only the following IDs: context_extraction, pregnancy_assessment, lactation_assessment, missing_information, evidence_sufficiency, safety_escalation, claim_citation.
6. `clinical_expected_behavior_acceptable_0_1` assesses whether the expected behavior is clinically appropriate within the displayed guide evidence boundary.

The draft key must not be used as an independently annotated gold standard until this review is finalized.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=MEDSAFE_DIR / "experiments" / "skill_benchmark_public.json")
    parser.add_argument("--draft-key", type=Path, default=MEDSAFE_DIR / "experiments" / "skill_benchmark_key_draft.json")
    parser.add_argument("--output-dir", type=Path, default=MEDSAFE_DIR / "experiments" / "benchmark_key_review")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public = {item["id"]: item for item in load_json(args.public)}
    rows = []
    for item in load_json(args.draft_key):
        qid = item["id"]
        rows.append({
            "id": qid,
            "query": public[qid]["query"],
            "scenario": item.get("scenario") or item.get("stress_type"),
            "target_drugs_json": compact_json(item.get("target_drugs") or item.get("gold_drug") or []),
            "proposed_context": item.get("context"),
            "proposed_requested_facts_json": compact_json(item.get("requested_facts") or []),
            "proposed_required_skills_json": compact_json(item.get("required_skills") or []),
            "proposed_missing_information_json": compact_json(item.get("missing_information") or []),
            "proposed_unsafe_premise_0_1": int(bool(item.get("unsafe_premise_detected"))),
            "proposed_acceptable_response_modes_json": compact_json(item.get("acceptable_response_modes") or []),
            "expected_behavior": item.get("expected_behavior"),
            "labels_acceptable_0_1": "",
            "reviewer_context": "",
            "reviewer_requested_facts_json": "",
            "reviewer_required_skills_json": "",
            "reviewer_missing_information_json": "",
            "reviewer_unsafe_premise_0_1": "",
            "reviewer_acceptable_response_modes_json": "",
            "clinical_expected_behavior_acceptable_0_1": "",
            "review_comment": "",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "benchmark_key_review.csv", rows)
    (args.output_dir / "benchmark_key_review_instructions.md").write_text(instruction_text(), encoding="utf-8")
    print(json.dumps({"row_count": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
