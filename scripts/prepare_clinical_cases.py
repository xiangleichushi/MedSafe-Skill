#!/usr/bin/env python
"""Convert a clinical consultation XLSX into model-facing and held-reference JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.clinical_case_import import load_clinical_cases_xlsx  # noqa: E402


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--sheet")
    parser.add_argument("--id-column")
    parser.add_argument("--query-column")
    parser.add_argument("--reference-column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_cases, reference_cases, summary = load_clinical_cases_xlsx(
        args.xlsx,
        sheet_name=args.sheet,
        id_column=args.id_column,
        query_column=args.query_column,
        reference_column=args.reference_column,
    )
    summary_output = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    save_json(args.output, public_cases)
    save_json(args.reference_output, reference_cases)
    save_json(summary_output, summary)
    print(json.dumps({
        "case_count": len(public_cases),
        "model_input": str(args.output),
        "held_reference": str(args.reference_output),
        "summary": str(summary_output),
        "reference_answers_excluded_from_model_input": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
