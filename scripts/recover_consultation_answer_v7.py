#!/usr/bin/env python
"""Recover a V7 draft from a saved final response without another API call."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from medsafe_skill.core.consultation_v7 import digest, render_answer, require, validate_answer
from medsafe_skill.scripts.run_consultation_v7 import (
    export, load, now, parse_json, save, sources_for, version_hash,
)


def recover(source_run_dir: Path, output_run_dir: Path, case_id: str) -> Path:
    require(not output_run_dir.exists(), "Use a new --output-run-dir; the failed run is preserved")
    records = deepcopy(load(source_run_dir / "consultation_cases_v7.json"))
    matches = [record for record in records if str(record["id"]) == case_id]
    require(len(matches) == 1, f"Expected one stored case {case_id}")
    record = matches[0]
    require(record.get("status") == "finalize_error", "Recovery requires a failed final answer")
    packet = next((p for p in record.get("packets", [])
                   if p.get("packet_id") == record.get("final_packet_id")), None)
    require(packet is not None, "No saved final decision packet; API reassessment is required")
    candidates = [(i, event) for i, event in enumerate(record["api_audit"])
                  if event.get("prompt") == "consultation_answer_v7.txt" and event.get("raw_response")]
    require(bool(candidates), "No stored final model response to recover")
    index, event = candidates[-1]
    require("unsupported absolute safety assurance" in event.get("validation", ""),
            "The latest model response did not fail on the absolute-safety wording check")
    request_packet = (event.get("request") or {}).get("decision_packet")
    if request_packet is not None:
        require(request_packet.get("packet_id") == packet["packet_id"],
                "Saved response belongs to a different decision packet")
    raw = event["raw_response"]
    sources = sources_for(record)
    data = validate_answer(parse_json(raw), packet, sources)
    edits = [item for item in data.get("_validation_adjustments", [])
             if item.get("rule") == "absolute_safety_wording_downgrade"]
    require(bool(edits), "No narrow safety wording adjustment was possible")
    answer, ledger = render_answer(data, packet, sources)
    record["final"] = {
        "fingerprint": digest({"input": record["input_fingerprint"], "supplements": record["supplements"]}),
        "answer": answer, "structured_answer": data, "claim_ledger": ledger,
        "decision_packet": packet,
        "validation": {"schema_and_source_spans_passed": True,
                       "clinical_accuracy_evaluated": False,
                       "local_wording_adjustments": edits},
        "at": now(),
    }
    record["api_audit"].append({
        "at": now(), "action": "local_answer_recovery_no_api",
        "source_event_index": index, "source_raw_sha256": digest(raw),
        "implementation_sha256": version_hash(), "wording_adjustments": edits,
    })
    record["status"] = "draft_generated"
    record.pop("error", None)
    record["state_history"].append({"at": now(), "state": "draft_generated_local_recovery",
                                    "packet_id": packet["packet_id"]})
    save(output_run_dir / "consultation_cases_v7.json", records)
    export(output_run_dir, records)
    return output_run_dir / "final_answers_readable.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(recover(args.source_run_dir, args.output_run_dir, args.case_id))


if __name__ == "__main__":
    main()
