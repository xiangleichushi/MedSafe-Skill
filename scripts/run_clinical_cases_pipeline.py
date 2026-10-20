#!/usr/bin/env python
"""Run real clinical cases from XLSX through the complete MedSafe pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_PROJECT_ROOT = MEDSAFE_DIR.parent
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    call_chat_api,
    parse_model_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=MEDSAFE_DIR / "output" / "clinical_cases",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--answer-max-tokens", type=int, default=3200)
    parser.add_argument("--json-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--resume-model-policy",
        choices=["exact", "pool", "preserve"],
        default="pool",
    )
    parser.add_argument("--groups", nargs="+", default=["medsafe_full"])
    parser.add_argument("--sheet")
    parser.add_argument("--id-column")
    parser.add_argument("--query-column")
    parser.add_argument("--reference-column")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Stop after observation extraction and HRC retrieval; use run_consultation_v7.py next")
    return parser.parse_args()


def run(command: list[str], log_path: Path) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[ClinicalCases] $ {printable}", flush=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def add_optional(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def require_environment(project_root: Path) -> None:
    missing = []
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("CHAT_API_KEY")):
        missing.append("DASHSCOPE_API_KEY (or CHAT_API_KEY)")
    if not os.getenv("EMBEDDING_MODEL_PATH"):
        missing.append("EMBEDDING_MODEL_PATH")
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    required_paths = [
        project_root / "retriever_critic.py",
        MEDSAFE_DIR / "resources" / "evidence_cards.jsonl",
    ]
    absent = [str(path) for path in required_paths if not path.exists()]
    if absent:
        raise FileNotFoundError("Required project files not found: " + ", ".join(absent))


def preflight_chat_api(timeout: int) -> None:
    api_url = (
        os.getenv("CHAT_API_URL")
        or os.getenv("DASHSCOPE_CHAT_API_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    model = os.getenv("CHAT_MODEL") or "qwen3.7-max"
    models = parse_model_pool(model, os.getenv("CHAT_MODELS") or "")
    print(f"[ClinicalCases] Chat model pool: {', '.join(models)}", flush=True)
    try:
        call_chat_api(
            api_url=api_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": "Reply with OK only."}],
            max_tokens=4,
            timeout=min(timeout, 60),
            temperature=0.0,
            max_retries=0,
            models=models,
        )
    except Exception as exc:
        raise RuntimeError(
            "Chat API preflight failed before the clinical run. Check whether the API key "
            f"is active and whether any configured model {models!r} is enabled for this account. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def require_complete_observations(path: Path, expected_count: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    errors = [item for item in records.values() if item.get("error")]
    successes = [item for item in records.values() if not item.get("error")]
    if errors or len(successes) < expected_count:
        raise RuntimeError(
            "Observation stage incomplete; downstream retrieval was not started. "
            f"successful={len(successes)} expected={expected_count} errors={len(errors)}. "
            "Fix the API issue and rerun the same command; compatible successes will be reused."
        )


def require_complete_retrieval(path: Path, expected_count: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    errors = [item for item in records if item.get("error")]
    unavailable = [
        item for item in records
        if (item.get("retrieval_quality") or {}).get("status") == "critic_unavailable"
    ]
    if errors or unavailable or len(records) < expected_count:
        raise RuntimeError(
            "Retrieval stage incomplete; answer generation was not started. "
            f"records={len(records)} expected={expected_count} errors={len(errors)} "
            f"critic_unavailable={len(unavailable)}. Rerun after restoring chat API access."
        )


def require_complete_answers(path: Path, expected_count: int) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    errors = [item for item in records if item.get("error")]
    invalid = [
        item for item in records
        if not item.get("error")
        and (
            not item.get("answer")
            or not item.get("decision_packet")
            or not item.get("reasoning_trace")
        )
    ]
    if errors or invalid or len(records) < expected_count:
        raise RuntimeError(
            "Answer stage incomplete. "
            f"records={len(records)} expected={expected_count} errors={len(errors)} "
            f"invalid={len(invalid)}. Do not run supplementation or finalization yet."
        )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    xlsx = args.xlsx.resolve()
    if not xlsx.exists():
        raise FileNotFoundError(xlsx)
    if args.smoke_limit < 1:
        raise ValueError("--smoke-limit must be at least 1")
    require_environment(project_root)
    print("[ClinicalCases] Checking chat API access...", flush=True)
    preflight_chat_api(args.api_timeout)
    print("[ClinicalCases] Chat API preflight passed.", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = run_dir / "pipeline.log"
    with pipeline_log.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== {args.mode} run {datetime.now(timezone.utc).isoformat()} "
            f"xlsx={xlsx} ===\n"
        )

    public_queries = run_dir / "clinical_cases_public.json"
    held_references = run_dir / "clinical_cases_reference_held_out.json"
    import_summary = run_dir / "clinical_cases_import_summary.json"
    observations = run_dir / "clinical_observations_cache.json"
    retrieval_raw = run_dir / "hrc_retrieval_raw.json"
    retrieval_enriched = run_dir / "hrc_retrieval_with_cards.json"
    answers = run_dir / "clinical_case_answers.json"
    artifacts = run_dir / "artifacts"

    prepare = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_clinical_cases.py"),
        "--xlsx", str(xlsx),
        "--output", str(public_queries),
        "--reference-output", str(held_references),
        "--summary-output", str(import_summary),
    ]
    add_optional(prepare, "--sheet", args.sheet)
    add_optional(prepare, "--id-column", args.id_column)
    add_optional(prepare, "--query-column", args.query_column)
    add_optional(prepare, "--reference-column", args.reference_column)
    run(prepare, pipeline_log)

    with import_summary.open("r", encoding="utf-8") as file:
        imported = json.load(file)
    case_count = int(imported["case_count"])
    selected_count = min(args.smoke_limit, case_count) if args.mode == "smoke" else case_count
    common_limit = ["--limit", str(args.smoke_limit)] if args.mode == "smoke" else []

    observation_command = [
        sys.executable,
        str(SCRIPT_DIR / "extract_clinical_observations.py"),
        "--query-path", str(public_queries),
        "--output", str(observations),
        "--save-each",
        "--resume",
        "--timeout", str(args.api_timeout),
        "--api-max-retries", str(args.api_max_retries),
        *common_limit,
    ]
    if args.continue_on_error:
        observation_command.append("--continue-on-error")
    run(observation_command, pipeline_log)
    require_complete_observations(observations, selected_count)

    retrieval_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_skill_batch_from_hrc.py"),
        "--project-root", str(project_root),
        "--query-path", str(public_queries),
        "--observation-cache", str(observations),
        "--raw-output", str(retrieval_raw),
        "--output", str(retrieval_enriched),
        "--save-each",
        "--resume",
        "--api-timeout", str(args.api_timeout),
        *common_limit,
    ]
    if args.continue_on_error:
        retrieval_command.append("--continue-on-error")
    run(retrieval_command, pipeline_log)
    require_complete_retrieval(retrieval_enriched, selected_count)

    if args.retrieval_only:
        print(json.dumps({"stage": "retrieval_complete", "selected_case_count": selected_count,
                          "source_run_dir_for_v7": str(run_dir),
                          "next": "run_consultation_v7.py prepare --source-run-dir <this directory> --run-dir <new directory>"},
                         ensure_ascii=False, indent=2))
        return

    answer_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_answer_experiment.py"),
        "--query-path", str(public_queries),
        "--retrieval-output", str(retrieval_enriched),
        "--observation-cache", str(observations),
        "--groups", *args.groups,
        "--output", str(answers),
        "--artifact-dir", str(artifacts),
        "--temperature", "0",
        "--save-each",
        "--resume",
        "--timeout", str(args.api_timeout),
        "--api-max-retries", str(args.api_max_retries),
        "--max-tokens", str(args.answer_max_tokens),
        "--json-repair-attempts", str(args.json_repair_attempts),
        "--resume-model-policy", args.resume_model_policy,
        *common_limit,
    ]
    if args.mode == "smoke":
        if "medsafe_full" not in args.groups:
            raise ValueError("Smoke mode requires medsafe_full in --groups")
        answer_command.append("--strict-smoke")
    if args.continue_on_error:
        answer_command.append("--continue-on-error")
    run(answer_command, pipeline_log)
    require_complete_answers(answers, selected_count * len(args.groups))

    print(json.dumps({
        "mode": args.mode,
        "detected_case_count": case_count,
        "selected_case_count": selected_count,
        "groups": args.groups,
        "answers": str(answers),
        "readable_answers": str(artifacts / "answers_readable.md"),
        "reasoning_traces": str(artifacts / "reasoning_traces.jsonl"),
        "review_csv": str(artifacts / "clinical_answer_review.csv"),
        "run_summary": str(artifacts / "run_summary.json"),
        "pipeline_log": str(pipeline_log),
        "held_reference_not_used_by_model": str(held_references),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
