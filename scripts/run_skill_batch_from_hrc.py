#!/usr/bin/env python
"""Run live HRC-RAG batch retrieval and attach MedSafe-Skill outputs.

Typical AutoDL usage from the project root:

python medsafe_skill/scripts/run_skill_batch_from_hrc.py \
  --query-path query.json \
  --output medsafe_skill/output/answer_skill_pipeline_top_10.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[1]
RETRIEVAL_GATE_VERSION = "2.0.0"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attach_cards_to_hcr_results import attach_cards  # noqa: E402
from load_card_store import EvidenceCardStore  # noqa: E402
from run_skill_pipeline_from_hrc import (  # noqa: E402
    build_config,
    import_retriever_module,
    maybe_set_chat_config,
)


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def classify_retrieval_quality(critic_history: list[dict[str, Any]]) -> dict[str, Any]:
    final_critic = critic_history[-1] if critic_history else {}
    verdict = str(final_critic.get("verdict") or "").upper()
    reason = str(final_critic.get("reason") or "")
    critic_unavailable = (
        not critic_history
        or "Critic API 失败" in reason
        or "Critic API failed" in reason
    )
    if critic_unavailable:
        status = "critic_unavailable"
    elif verdict == "PASS":
        status = "passed"
    else:
        status = "failed_after_retries"
    return {
        "status": status,
        "final_verdict": final_critic.get("verdict"),
        "reason": final_critic.get("reason"),
        "critic_rounds": len(critic_history),
    }


def run_batch(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.project_root = args.project_root.resolve()
    retriever_module = import_retriever_module(args.project_root)
    maybe_set_chat_config(retriever_module, args)
    config = build_config(retriever_module, args)
    critic_prompt_sha256 = hashlib.sha256(
        str(retriever_module.CRITIC_PROMPT).encode("utf-8")
    ).hexdigest()

    with args.query_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
    if args.start:
        queries = queries[args.start :]
    if args.limit is not None:
        queries = queries[: args.limit]

    existing_raw = load_existing_results(args.raw_output) if args.resume else []
    def reusable(item: dict[str, Any]) -> bool:
        return (
            "error" not in item
            and item.get("critic_prompt_sha256") == critic_prompt_sha256
            and item.get("retrieval_gate_version") == RETRIEVAL_GATE_VERSION
            and (item.get("retrieval_quality") or {}).get("status") != "critic_unavailable"
        )

    # Errors and unavailable-Critic records must remain retryable on --resume.
    done_ids = {
        item.get("id")
        for item in existing_raw
        if reusable(item)
    }
    raw_results = [item for item in existing_raw if reusable(item)]
    invalidated = len(existing_raw) - len(raw_results)
    if invalidated:
        print(f"[MedSafe] Invalidated {invalidated} stale/error/critic-unavailable retrieval records")

    print("[MedSafe] Loading Evidence Cards...")
    store = EvidenceCardStore.from_jsonl(args.cards)
    print(f"[MedSafe] Loaded {len(store.cards)} Evidence Cards.")

    observation_cache = {}
    if args.observation_cache:
        if not args.observation_cache.exists():
            raise FileNotFoundError(f"Observation cache not found: {args.observation_cache}")
        with args.observation_cache.open("r", encoding="utf-8") as f:
            observation_cache = json.load(f)
        if not isinstance(observation_cache, dict):
            raise ValueError("--observation-cache must contain a JSON object")

    print("[MedSafe] Loading HRC-RAG retriever...")
    retriever = retriever_module.HybridRetriever(config)

    total = len(queries)
    for i, item in enumerate(queries, start=1):
        qid = item.get("id", i)
        if args.resume and qid in done_ids:
            print(f"[MedSafe] Skip existing id={qid}")
            continue

        query = item["query"]
        print(f"[MedSafe] Query {i}/{total} id={qid}: {query}")
        started = time.time()
        route_start = len(getattr(retriever_module, "MEDSAFE_MODEL_ROUTE_EVENTS", []))
        try:
            result = retriever.retrieve_single(query)
            result["id"] = qid
            result["critic_prompt_file"] = "hrc_critic_clinical_v2.txt"
            result["critic_prompt_sha256"] = critic_prompt_sha256
            result["retrieval_gate_version"] = RETRIEVAL_GATE_VERSION
            critic_history = result.get("critic_history") or []
            result["retrieval_quality"] = classify_retrieval_quality(critic_history)
            result["api_model_routes"] = list(
                getattr(retriever_module, "MEDSAFE_MODEL_ROUTE_EVENTS", [])[route_start:]
            )
            raw_results.append(result)
            print(f"[MedSafe] Done id={qid} elapsed={time.time() - started:.1f}s")
        except Exception as exc:
            print(f"[MedSafe][ERROR] id={qid} failed: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise
            raw_results.append(
                {
                    "id": qid,
                    "query": query,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

        if args.save_each:
            save_json(args.raw_output, raw_results)

    save_json(args.raw_output, raw_results)

    valid_raw = [item for item in raw_results if "error" not in item]
    enriched, summary = attach_cards(
        valid_raw,
        store=store,
        top_k=args.skill_top_k,
        observation_cache=observation_cache,
    )
    if any("error" in item for item in raw_results):
        enriched.extend(item for item in raw_results if "error" in item)

    save_json(args.output, enriched)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    save_json(summary_path, summary)
    return enriched, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live HRC-RAG batch with MedSafe-Skill enrichment.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--query-path", type=Path, default=DEFAULT_PROJECT_ROOT / "query.json")
    parser.add_argument("--output", type=Path, default=MEDSAFE_DIR / "output" / "answer_skill_pipeline_top_10.json")
    parser.add_argument("--raw-output", type=Path, default=MEDSAFE_DIR / "output" / "answer_skill_pipeline_raw_hrc.json")
    parser.add_argument("--cards", type=Path, default=MEDSAFE_DIR / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--observation-cache", type=Path)
    parser.add_argument("--skill-top-k", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-each", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--metadata-path")
    parser.add_argument("--faiss-category-path")
    parser.add_argument("--faiss-indication-path")
    parser.add_argument("--retriever-output-dir")
    parser.add_argument("--chat-api-url")
    parser.add_argument("--chat-api-key")
    parser.add_argument("--chat-model")
    parser.add_argument(
        "--chat-models",
        default=None,
        help="Comma-separated fallback models; --chat-model/CHAT_MODEL is primary.",
    )
    parser.add_argument("--api-timeout", type=int, default=60)
    parser.add_argument("--api-max-tokens", type=int, default=512)
    parser.add_argument("--critic-api-max-tokens", type=int, default=1024)
    parser.add_argument("--critic-max-rounds", type=int, default=2)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=40)
    parser.add_argument("--warn-print-llm-output", action="store_true")
    parser.add_argument("--warn-llm-output-max-chars", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched, summary = run_batch(args)
    print(f"[MedSafe] Saved enriched output: {args.output}")
    print(f"[MedSafe] Enriched items: {len(enriched)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
