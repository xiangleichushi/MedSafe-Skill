#!/usr/bin/env python
"""Extract, semantically verify, and cache clinical observations with the chat API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    extract_clinical_observations,
    observation_record_is_compatible,
    parse_model_pool,
    query_hash,
    save_json,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=MEDSAFE_DIR / "output" / "clinical_observations_cache.json",
    )
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument(
        "--models",
        default=os.getenv("CHAT_MODELS") or "",
        help="Comma-separated fallback models; --model/CHAT_MODEL is primary.",
    )
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--semantic-max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-each", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models = ",".join(parse_model_pool(args.model, args.models))
    os.environ["CHAT_MODELS"] = args.models
    print(f"[ChatAPI] Configured model pool: {args.models}", flush=True)
    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")
    queries = load_json(args.query_path)[args.start :]
    if args.limit is not None:
        queries = queries[: args.limit]
    cache = load_json(args.output) if args.resume and args.output.exists() else {}
    if not isinstance(cache, dict):
        raise ValueError("Observation cache must be a JSON object")

    api_args = SimpleNamespace(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        observation_max_tokens=args.max_tokens,
        observation_semantic_max_tokens=args.semantic_max_tokens,
        timeout=args.timeout,
        api_max_retries=args.api_max_retries,
    )
    errors = 0
    for index, item in enumerate(queries, start=1):
        query = str(item.get("query") or item.get("query_text") or "").strip()
        qid = item.get("id", index)
        key = query_hash(query)
        if args.resume and observation_record_is_compatible(cache.get(key)):
            print(f"[Observation] Skip {index}/{len(queries)} id={qid}")
            continue
        if args.resume and key in cache:
            print(f"[Observation] Rebuild stale span-only cache {index}/{len(queries)} id={qid}")
        print(f"[Observation] Run {index}/{len(queries)} id={qid}")
        try:
            record = extract_clinical_observations(query, api_args)
            record["case_id"] = qid
            cache[key] = record
        except Exception as exc:
            errors += 1
            print(f"[Observation][ERROR] id={qid}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise
            cache[key] = {
                "case_id": qid,
                "query_sha256": key,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        if args.save_each:
            save_json(args.output, cache)
    save_json(args.output, cache)
    print(json.dumps({"query_count": len(queries), "cache_count": len(cache), "error_count": errors, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
