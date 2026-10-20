#!/usr/bin/env python
"""Run an LLM-only baseline for the hallucination stress test.

The script intentionally does not retrieve Evidence Cards. It provides a
reasonable clinical-assistant prompt and records the direct model answer, so
the result can be compared with the MedSafe-Skill evidence-grounded output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_QUERY_PATH = MEDSAFE_DIR / "hallucination" / "query_hallucination.json"
DEFAULT_OUTPUT = MEDSAFE_DIR / "output" / "hallucination_catalog_llm_only.json"

SYSTEM_PROMPT = """你是一个谨慎的妊娠期和哺乳期用药问答助手。
请用中文回答用户问题，给出简洁但可审核的临床安全建议。
如果你不确定、缺少证据、药物名称无法确认或问题要求具体发生率/剂量/乳汁浓度但你没有可靠依据，请明确说明证据不足，不要编造数字、研究出处或指南结论。
回答不能替代医生或药师的个体化判断。"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_api_key(api_key: str, api_url: str) -> str:
    if not api_key:
        return api_key
    if "dashscope.aliyuncs.com" in api_url and not api_key.lower().startswith("bearer "):
        return f"Bearer {api_key}"
    return api_key


def call_chat_api(
    *,
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    temperature: float,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": normalize_api_key(api_key, api_url),
        "Content-Type": "application/json",
        "User-Agent": "medsafe-hallucination-baseline/1.0",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_answer(response_json: dict[str, Any]) -> str:
    try:
        return str(response_json["choices"][0]["message"]["content"]).strip()
    except Exception as exc:  # pragma: no cover - defensive for remote API variants
        raise RuntimeError(f"Unexpected chat response shape: {response_json}") from exc


def existing_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    if not isinstance(data, list):
        return {}
    return {item.get("id"): item for item in data}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-only baseline on hallucination stress queries.")
    parser.add_argument("--query-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY_ID") or "")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="When used with --resume, retry previous items that have error but no answer.")
    parser.add_argument("--save-each", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")

    queries = load_json(args.query_path)
    if args.start:
        queries = queries[args.start :]
    if args.limit is not None:
        queries = queries[: args.limit]

    previous = existing_by_id(args.output) if args.resume else {}
    results = []
    done_ids = set()
    for key, previous_item in previous.items():
        should_retry = args.retry_errors and previous_item.get("error") and not previous_item.get("answer")
        if should_retry:
            print(f"[LLM-only] Retry previous error id={key}")
            continue
        results.append(previous_item)
        done_ids.add(key)

    total = len(queries)
    for idx, item in enumerate(queries, start=1):
        qid = item.get("id", idx)
        if qid in done_ids:
            print(f"[LLM-only] Skip existing id={qid}")
            continue

        query = item["query"]
        print(f"[LLM-only] Query {idx}/{total} id={qid}: {query}")
        started = time.time()
        output_item: dict[str, Any] = {
            "id": qid,
            "method": "llm_only",
            "query": query,
            "model": args.model,
            "system_prompt": SYSTEM_PROMPT,
        }
        try:
            response_json = call_chat_api(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                temperature=args.temperature,
            )
            output_item["answer"] = extract_answer(response_json)
            output_item["elapsed_seconds"] = round(time.time() - started, 3)
            output_item["raw_response"] = response_json
            print(f"[LLM-only] Done id={qid} elapsed={output_item['elapsed_seconds']}s")
        except Exception as exc:
            output_item["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"[LLM-only][ERROR] id={qid}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

        results.append(output_item)
        if args.save_each:
            save_json(args.output, results)

    save_json(args.output, results)
    print(f"[LLM-only] Saved: {args.output}")
    print(json.dumps({"input_count": total, "saved_count": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
