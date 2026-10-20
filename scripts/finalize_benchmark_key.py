#!/usr/bin/env python
"""Finalize a reviewed Skill benchmark key without silently accepting blanks."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
ALLOWED_CONTEXTS = {"pregnancy", "preconception", "lactation", "pregnancy_and_lactation", "unspecified"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_json_field(value: str, field: str, qid: str) -> Any:
    try:
        return json.loads(value)
    except Exception as exc:
        raise ValueError(f"Invalid JSON in {field} for id={qid}: {value}") from exc


def parse_binary(value: str, field: str, qid: str) -> int:
    value = str(value).strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1 for id={qid}")
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-key", type=Path, default=MEDSAFE_DIR / "experiments" / "skill_benchmark_key_draft.json")
    parser.add_argument("--review-sheet", type=Path, default=MEDSAFE_DIR / "experiments" / "benchmark_key_review" / "benchmark_key_review.csv")
    parser.add_argument("--output", type=Path, default=MEDSAFE_DIR / "experiments" / "skill_benchmark_key_reviewed.json")
    parser.add_argument("--reviewer-id", required=True, help="De-identified reviewer code, not a personal name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draft = {item["id"]: item for item in load_json(args.draft_key)}
    reviews = {row["id"]: row for row in load_csv(args.review_sheet)}
    if set(draft) != set(reviews):
        raise ValueError("Review sheet ids do not match the draft key")

    finalized = []
    for qid, item in draft.items():
        row = reviews[qid]
        accepted = parse_binary(row.get("labels_acceptable_0_1", ""), "labels_acceptable_0_1", qid)
        clinical_ok = parse_binary(
            row.get("clinical_expected_behavior_acceptable_0_1", ""),
            "clinical_expected_behavior_acceptable_0_1",
            qid,
        )
        if not clinical_ok:
            raise ValueError(f"Expected behavior requires correction before finalization for id={qid}")

        result = dict(item)
        if not accepted:
            context = row.get("reviewer_context", "").strip()
            if context not in ALLOWED_CONTEXTS:
                raise ValueError(f"Invalid reviewer_context for id={qid}: {context}")
            required = {
                "requested_facts": "reviewer_requested_facts_json",
                "required_skills": "reviewer_required_skills_json",
                "missing_information": "reviewer_missing_information_json",
                "acceptable_response_modes": "reviewer_acceptable_response_modes_json",
            }
            result["context"] = context
            for target, source in required.items():
                value = row.get(source, "").strip()
                if not value:
                    raise ValueError(f"Missing {source} for corrected id={qid}")
                result[target] = parse_json_field(value, source, qid)
            result["unsafe_premise_detected"] = bool(parse_binary(
                row.get("reviewer_unsafe_premise_0_1", ""),
                "reviewer_unsafe_premise_0_1",
                qid,
            ))

        result["annotation_status"] = "independently_reviewed"
        result["annotation_review"] = {
            "reviewer_id": args.reviewer_id,
            "labels_accepted_without_change": bool(accepted),
            "clinical_expected_behavior_accepted": bool(clinical_ok),
            "comment": row.get("review_comment", "").strip(),
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        finalized.append(result)

    save_json(args.output, finalized)
    print(json.dumps({
        "finalized_count": len(finalized),
        "reviewer_id": args.reviewer_id,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
