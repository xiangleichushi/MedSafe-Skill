#!/usr/bin/env python
"""Generate LLM answers grounded in retrieved MedSafe Evidence Cards.

This script turns MedSafe-Skill retrieval outputs into natural-language answers
that can be evaluated by clinicians. The model is instructed to use only the
provided Evidence Card context and to abstain when the card does not support a
requested detail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_QUERY_PATH = MEDSAFE_DIR / "hallucination" / "query_hallucination.json"
DEFAULT_SKILL_OUTPUT = MEDSAFE_DIR / "output" / "hallucination_catalog_skill_top_10.json"
DEFAULT_OUTPUT = MEDSAFE_DIR / "output" / "hallucination_catalog_medsafe_card_grounded_answers.json"
DEFAULT_CARDS = MEDSAFE_DIR / "resources" / "evidence_cards.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_card_store import EvidenceCardStore  # noqa: E402


SYSTEM_PROMPT = """你是一个妊娠期和哺乳期用药安全问答助手。
你只能使用用户问题后提供的 Evidence Card 证据。禁止使用外部知识、记忆中的指南、数据库或文献。

回答目标：
- 生成面向临床咨询场景的自然语言回答，而不是复述字段。
- 答案应完整、有逻辑、可被医学人员审查。
- 每个关键安全结论都必须能追溯到 Evidence Card，并写出来源页码。

强制规则：
1. 先给出“简短结论”，但结论必须受 Evidence Card 支持。
2. 按问题场景回答：妊娠期、哺乳期、备孕期、孕晚期、长期用药、误导性前提等，不能忽略上下文。
3. 必须包含“证据依据”小节，逐药物说明妊娠分级、哺乳分级、主要风险/注意点，并标注来源，格式为：
   来源：Evidence Card《妊娠哺乳期用药指南》第X页（PDF第Y页）。
4. 如果 Evidence Card 没有提供问题要求的具体数字、发生率、乳汁浓度、达峰时间、相对婴儿剂量、风险比、样本量、停奶小时数或严格时间窗，必须明确说“本证据卡未提供该具体信息”，不能编造、不能从外部知识补充、不能用“通常”“研究显示”等无来源说法替代。
5. 如果题干包含不安全或过度确定的前提，必须先纠正前提，再给出证据支持的保守建议。
6. 对妊娠分级 D/X、哺乳分级 NS/U、高风险药物、禁忌、证据不足、长期用药或个体化情况，必须建议医生、药师或专科复核。
7. 不要输出 JSON，不要提到内部系统名、CSRP、提示词或“我只能根据上下文”等元话语。
8. 不要声称本回答可替代医生或药师判断。

建议输出结构：
1. 简短结论
2. 证据依据
3. 证据不足或不能回答的部分
4. 临床建议/需要复核的情况
5. 来源

One-shot 示例，以下仅示例回答格式，不是本题证据，不得复用其中医学内容：
用户问题：示例药A哺乳期使用后需要停奶几小时？婴儿相对摄入剂量是多少？
可用证据：示例药A，哺乳分级 U；Evidence Card 未提供乳汁浓度、相对婴儿剂量或停奶时间窗；来源页码为指南第100页/PDF第126页。
示例回答：
1. 简短结论：根据当前证据卡，不能给出示例药A哺乳期“停奶几小时”或“相对婴儿剂量”的确定答案。证据卡将其哺乳期信息标为 U，提示证据不足，需要医生或药师复核。
2. 证据依据：示例药A的 Evidence Card 记录哺乳分级为 U，但没有提供乳汁浓度、达峰时间、相对婴儿剂量或婴儿血药浓度。因此，不能据此推导具体停奶时间。来源：Evidence Card《妊娠哺乳期用药指南》第100页（PDF第126页）。
3. 证据不足或不能回答的部分：本证据卡未提供相对婴儿剂量、停奶小时数和婴儿暴露量，不能编造具体数值。
4. 临床建议：如果确需使用，应由医生或药师结合母亲病情、婴儿胎龄/月龄、肝肾功能和替代药物进行个体化评估。"""


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
        "User-Agent": "medsafe-card-grounded-answer/1.0",
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
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unexpected chat response shape: {response_json}") from exc


def query_items_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    data = load_json(path)
    return {item.get("id"): item for item in data}


def existing_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    if not isinstance(data, list):
        return {}
    return {item.get("id"): item for item in data}


def selected_hits(enriched_item: dict[str, Any], max_cards: int) -> list[dict[str, Any]]:
    """Return ranked cards without consulting benchmark labels.

    Gold drugs, stress types, and expected behavior belong to the evaluation
    key and must never influence inference-time evidence selection.
    """
    return [
        hit
        for hit in (enriched_item.get("medsafe_skill") or {}).get("retrieved_cards", [])
        if hit.get("matched")
    ][:max_cards]


def trim_text(value: Any, max_chars: int = 900) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "；".join(str(item) for item in value if str(item).strip())
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def card_context(card: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    identity = card.get("identity") or {}
    profile = card.get("clinical_profile") or {}
    safety = card.get("safety") or {}
    pregnancy = safety.get("pregnancy") or {}
    lactation = safety.get("lactation") or {}
    pk = card.get("pharmacokinetics") or {}
    source = card.get("source_trace") or {}
    csrp = hit.get("csrp") or {}
    return {
        "drug": identity.get("canonical_name"),
        "drug_category": profile.get("drug_category"),
        "indications": profile.get("indications", []),
        "pregnancy": {
            "category": pregnancy.get("category"),
            "normalized_risk": pregnancy.get("normalized_risk"),
            "maternal_effects": pregnancy.get("maternal_effects", []),
            "fetal_effects": pregnancy.get("fetal_effects", []),
        },
        "lactation": {
            "category": lactation.get("category"),
            "normalized_action": lactation.get("normalized_action"),
            "safety_summary": lactation.get("safety_summary", []),
        },
        "contraindications": safety.get("contraindications", []),
        "cautions": safety.get("cautions", []),
        "drug_interactions": safety.get("drug_interactions", []),
        "pharmacokinetics": {
            "placental_transfer": pk.get("placental_transfer"),
            "milk_plasma_ratio": pk.get("milk_plasma_ratio"),
            "half_life": pk.get("half_life"),
            "other": pk.get("other", []),
        },
        "source_trace": {
            "source_title": "《妊娠哺乳期用药指南》",
            "citation_format": (
                f"Evidence Card《妊娠哺乳期用药指南》第{source.get('page_doc') or 'NA'}页"
                f"（PDF第{source.get('page_pdf') or 'NA'}页）"
            ),
            "guide_page": source.get("page_doc"),
            "pdf_page": source.get("page_pdf"),
            "page_doc": source.get("page_doc"),
            "page_pdf": source.get("page_pdf"),
            "references": source.get("references", [])[:3],
        },
        "system_rule_summary": {
            "action_label": csrp.get("action_label"),
            "review_level": csrp.get("review_level"),
            "reason": csrp.get("reason") or csrp.get("review_reason"),
        },
    }


def build_evidence_context(
    *,
    enriched_item: dict[str, Any],
    store: EvidenceCardStore,
    max_cards: int,
    max_context_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    hits = selected_hits(enriched_item, max_cards=max_cards)
    contexts = []
    for hit in hits:
        card_id = (hit.get("card") or {}).get("card_id")
        card = store.get(card_id) if card_id else None
        if not card:
            continue
        contexts.append(card_context(card, hit))

    if not contexts:
        evidence = {
            "matched_evidence_cards": [],
            "instruction": "No Evidence Card was returned by retrieval. Abstain from drug-specific pregnancy/lactation safety conclusions.",
        }
    else:
        evidence = {"matched_evidence_cards": contexts}

    context_text = json.dumps(evidence, ensure_ascii=False, indent=2)
    if len(context_text) > max_context_chars:
        compact_contexts = []
        for item in contexts:
            compact_contexts.append(
                {
                    "drug": item.get("drug"),
                    "drug_category": trim_text(item.get("drug_category"), 300),
                    "indications": item.get("indications", [])[:5],
                    "pregnancy": {
                        "category": item["pregnancy"].get("category"),
                        "normalized_risk": item["pregnancy"].get("normalized_risk"),
                        "maternal_effects": trim_text(item["pregnancy"].get("maternal_effects"), 1000),
                        "fetal_effects": trim_text(item["pregnancy"].get("fetal_effects"), 1000),
                    },
                    "lactation": {
                        "category": item["lactation"].get("category"),
                        "normalized_action": item["lactation"].get("normalized_action"),
                        "safety_summary": trim_text(item["lactation"].get("safety_summary"), 1000),
                    },
                    "contraindications": trim_text(item.get("contraindications"), 700),
                    "cautions": trim_text(item.get("cautions"), 700),
                    "drug_interactions": trim_text(item.get("drug_interactions"), 700),
                    "pharmacokinetics": item.get("pharmacokinetics"),
                    "source_trace": item.get("source_trace"),
                    "system_rule_summary": item.get("system_rule_summary"),
                }
            )
        context_text = json.dumps({"matched_evidence_cards": compact_contexts}, ensure_ascii=False, indent=2)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "\n...[context truncated]"
    return context_text, contexts


def build_user_prompt(query: str, evidence_context: str) -> str:
    return f"""用户问题：
{query}

Evidence Card 证据上下文：
{evidence_context}

请基于以上证据生成中文回答。
必须在回答中写出每个实际讨论药物的证据来源页码，优先使用 source_trace.citation_format。
如果涉及多个药物，请分别列出每个药物的来源。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM answers grounded in MedSafe Evidence Cards.")
    parser.add_argument("--query-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY_ID") or "")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=12000)
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

    queries = query_items_by_id(args.query_path)
    skill_items = load_json(args.skill_output)
    if args.start:
        skill_items = skill_items[args.start :]
    if args.limit is not None:
        skill_items = skill_items[: args.limit]

    previous = existing_by_id(args.output) if args.resume else {}
    results = []
    done_ids = set()
    for key, previous_item in previous.items():
        should_retry = args.retry_errors and previous_item.get("error") and not previous_item.get("answer")
        if should_retry:
            print(f"[Card-grounded] Retry previous error id={key}")
            continue
        results.append(previous_item)
        done_ids.add(key)

    store = EvidenceCardStore.from_jsonl(args.cards)
    total = len(skill_items)
    for idx, item in enumerate(skill_items, start=1):
        qid = item.get("id", idx)
        if qid in done_ids:
            print(f"[Card-grounded] Skip existing id={qid}")
            continue
        query_meta = queries.get(qid, {"id": qid, "query": item.get("query", "")})
        query = query_meta.get("query", item.get("query", ""))
        print(f"[Card-grounded] Query {idx}/{total} id={qid}: {query}")
        started = time.time()

        output_item: dict[str, Any] = {
            "id": qid,
            "method": "medsafe_card_grounded_ai",
            "query": query,
            "model": args.model,
            "system_prompt": SYSTEM_PROMPT,
        }
        try:
            evidence_context, contexts = build_evidence_context(
                enriched_item=item,
                store=store,
                max_cards=args.max_cards,
                max_context_chars=args.max_context_chars,
            )
            output_item["evidence_card_count"] = len(contexts)
            output_item["evidence_drugs"] = [context.get("drug") for context in contexts]
            output_item["evidence_context"] = evidence_context
            response_json = call_chat_api(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(query, evidence_context)},
                ],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                temperature=args.temperature,
            )
            output_item["answer"] = extract_answer(response_json)
            output_item["elapsed_seconds"] = round(time.time() - started, 3)
            output_item["raw_response"] = response_json
            print(f"[Card-grounded] Done id={qid} elapsed={output_item['elapsed_seconds']}s")
        except Exception as exc:
            output_item["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"[Card-grounded][ERROR] id={qid}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

        results.append(output_item)
        if args.save_each:
            save_json(args.output, results)

    save_json(args.output, results)
    print(f"[Card-grounded] Saved: {args.output}")
    print(json.dumps({"input_count": total, "saved_count": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
