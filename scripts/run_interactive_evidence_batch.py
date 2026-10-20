#!/usr/bin/env python
"""Run terminal evidence-supplementation sessions for every eligible case."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
SESSION_SCRIPT = SCRIPT_DIR / "run_interactive_evidence_session.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.supplement_request import build_unified_supplement_request  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--reopen-complete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def eligible_cases(records: list[dict[str, Any]], case_ids: set[str] | None) -> list[dict[str, Any]]:
    return [
        item
        for item in records
        if item.get("group") == "medsafe_full"
        and build_unified_supplement_request(item)
        and (case_ids is None or str(item.get("id")) in case_ids)
    ]


def session_path(answers: Path, case_id: Any) -> Path:
    return answers.with_name(f"interactive_evidence_case_{case_id}.json")


def session_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        session = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        session.get("collection_closed_by_user")
        or str(session.get("session_status") or "").startswith("supplementation_complete")
    )


def main() -> None:
    args = parse_args()
    if not args.answers.exists():
        raise FileNotFoundError(args.answers)
    if not (os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or CHAT_API_KEY.")

    records = load_json(args.answers)
    if not isinstance(records, list):
        raise ValueError("--answers must contain a JSON list")
    failed = [item.get("id") for item in records if item.get("error")]
    invalid = [
        item.get("id")
        for item in records
        if not item.get("error")
        and (
            not item.get("decision_packet")
            or not item.get("reasoning_trace")
            or not item.get("answer")
        )
    ]
    if failed or invalid:
        raise RuntimeError(
            "Stage 1 is incomplete; interactive supplementation is unsafe. "
            f"error_records={len(failed)} invalid_records={len(invalid)} "
            f"examples={(failed + invalid)[:10]}"
        )
    selected_ids = set(args.case_ids) if args.case_ids else None
    selected = eligible_cases(records, selected_ids)
    selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"[InteractiveBatch] Eligible cases: {len(selected)}")
    print("[InteractiveBatch] IDs: " + " ".join(str(item.get("id")) for item in selected))
    if not selected:
        print("[InteractiveBatch] No cases require interactive evidence supplementation.")
        return

    completed = 0
    skipped = 0
    errors = []
    for index, item in enumerate(selected, start=1):
        case_id = item.get("id")
        output = session_path(args.answers, case_id)
        if not args.reopen_complete and session_is_complete(output):
            print(f"[InteractiveBatch] Skip complete id={case_id}")
            skipped += 1
            continue

        print(f"\n[InteractiveBatch] Case {index}/{len(selected)} id={case_id}")
        command = [
            sys.executable,
            str(SESSION_SCRIPT),
            "--answers", str(args.answers),
            "--case-id", str(case_id),
            "--output", str(output),
            "--timeout", str(args.timeout),
            "--api-max-retries", str(args.api_max_retries),
            "--max-turns", str(args.max_turns),
            "--resume",
            "--no-generate-answer",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            errors.append({"id": case_id, "returncode": result.returncode})
            print(f"[InteractiveBatch][ERROR] id={case_id} returncode={result.returncode}")
            if not args.continue_on_error:
                raise subprocess.CalledProcessError(result.returncode, command)
        else:
            completed += 1

    print(json.dumps({
        "eligible_count": len(selected),
        "completed_this_run": completed,
        "skipped_complete": skipped,
        "error_count": len(errors),
        "errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
