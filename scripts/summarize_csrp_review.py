#!/usr/bin/env python
"""Summarize completed CSRP clinical validation sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
ACTIONS = ["continue", "caution", "avoid", "insufficient", "ask_clarification"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, default=MEDSAFE_DIR / "experiments" / "csrp_clinical_review" / "csrp_review_sheet.csv")
    parser.add_argument("--output-dir", type=Path, default=MEDSAFE_DIR / "report_csrp_clinical")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.review_sheet)
    for row in rows:
        if row.get("expert_action") not in ACTIONS:
            raise ValueError(f"Invalid or missing expert_action for {row.get('case_id')}")
        if row.get("action_agreement_0_1") not in {"0", "1"}:
            raise ValueError(f"Invalid or missing action agreement for {row.get('case_id')}")
        if row.get("review_level_agreement_0_1") not in {"0", "1"}:
            raise ValueError(f"Invalid or missing review-level agreement for {row.get('case_id')}")

    by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_context[row["context"]].append(row)
    summaries = []
    for context, values in sorted(by_context.items()):
        summaries.append({
            "context": context,
            "n": len(values),
            "action_exact_agreement": round(sum(int(row["action_agreement_0_1"]) for row in values) / len(values), 4),
            "review_level_exact_agreement": round(sum(int(row["review_level_agreement_0_1"]) for row in values) / len(values), 4),
            "system_too_permissive_rate": round(sum(row.get("direction_if_disagree") == "system_too_permissive" for row in values) / len(values), 4),
            "system_too_conservative_rate": round(sum(row.get("direction_if_disagree") == "system_too_conservative" for row in values) / len(values), 4),
        })
    summaries.append({
        "context": "overall",
        "n": len(rows),
        "action_exact_agreement": round(sum(int(row["action_agreement_0_1"]) for row in rows) / len(rows), 4),
        "review_level_exact_agreement": round(sum(int(row["review_level_agreement_0_1"]) for row in rows) / len(rows), 4),
        "system_too_permissive_rate": round(sum(row.get("direction_if_disagree") == "system_too_permissive" for row in rows) / len(rows), 4),
        "system_too_conservative_rate": round(sum(row.get("direction_if_disagree") == "system_too_conservative" for row in rows) / len(rows), 4),
    })

    confusion = Counter((row["system_action"], row["expert_action"]) for row in rows)
    confusion_rows = [
        {"system_action": system, "expert_action": expert, "count": confusion.get((system, expert), 0)}
        for system in ACTIONS for expert in ACTIONS
    ]
    directions = Counter(row.get("direction_if_disagree") or "agreement_or_blank" for row in rows)
    report = {"summary": summaries, "disagreement_directions": dict(directions), "confusion_matrix": confusion_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "csrp_clinical_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_csv(args.output_dir / "csrp_agreement_summary.csv", summaries)
    write_csv(args.output_dir / "csrp_confusion_matrix.csv", confusion_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
