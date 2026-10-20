#!/usr/bin/env python
"""Revalidate and render saved V7.2 structured answers without calling a model."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from medsafe_skill.core.consultation_v7 import VERSION, digest, render_answer, require, validate_answer
from medsafe_skill.scripts.run_consultation_v7 import export, load, now, save, sources_for, version_hash


def rerender(source_run_dir: Path, output_run_dir: Path, case_ids: list[str] | None = None) -> Path:
    require(not output_run_dir.exists(), "Use a new --output-run-dir to preserve previous results")
    stored = load(source_run_dir / "consultation_cases_v7.json")
    selected = [deepcopy(r) for r in stored if case_ids is None or str(r["id"]) in case_ids]
    require(bool(selected), "No saved cases selected")
    if case_ids is not None:
        require({str(r["id"]) for r in selected} == set(case_ids), "Some requested case IDs are missing")
    for record in selected:
        final = record.get("final") or {}
        packet = final.get("decision_packet")
        require(packet is not None and packet.get("schema_version") == VERSION,
                f"case {record['id']}: saved answer needs a current-schema decision packet")
        structured = final.get("structured_answer")
        require(structured is not None, f"case {record['id']}: no saved structured answer")
        original_answer_hash = digest(final.get("answer"))
        sources = sources_for(record)
        data = validate_answer(structured, packet, sources)
        answer, ledger = render_answer(data, packet, sources)
        final.update({"answer": answer, "structured_answer": data, "claim_ledger": ledger,
                      "rerendered_at": now()})
        record["api_audit"].append({"at": now(), "action": "local_rerender_no_api",
                                    "source_run_dir": str(source_run_dir),
                                    "source_answer_sha256": original_answer_hash,
                                    "implementation_sha256": version_hash()})
    save(output_run_dir / "consultation_cases_v7.json", selected)
    export(output_run_dir, selected)
    return output_run_dir / "final_answers_readable.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    print(rerender(args.source_run_dir, args.output_run_dir, args.case_ids))


if __name__ == "__main__":
    main()
