#!/usr/bin/env python
"""Render MedSafe-Skill hallucination-test outputs into reviewable answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_QUERY_PATH = MEDSAFE_DIR / "hallucination" / "query_hallucination.json"
DEFAULT_SKILL_OUTPUT = MEDSAFE_DIR / "output" / "hallucination_skill_top_10.json"
DEFAULT_OUTPUT = MEDSAFE_DIR / "output" / "hallucination_medsafe_answers.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_card_store import normalize_text  # noqa: E402


ACTION_RANK = {
    "continue": 0,
    "caution": 1,
    "ask_clarification": 2,
    "insufficient": 3,
    "avoid": 4,
}
REVIEW_RANK = {
    "standard_clinician_confirmation": 0,
    "pharmacist_review": 1,
    "specialist_review": 2,
}
ACTION_ZH = {
    "continue": "可考虑使用/继续，但仍需结合个体情况确认",
    "caution": "需谨慎使用，应由医生或药师评估",
    "avoid": "应避免使用或视为高风险",
    "insufficient": "证据不足，不能给出确定安全结论",
    "ask_clarification": "需要补充关键信息后再判断",
}
REVIEW_ZH = {
    "standard_clinician_confirmation": "常规临床确认",
    "pharmacist_review": "药师复核",
    "specialist_review": "专科复核",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def query_items_by_id(path: Path) -> dict[Any, dict[str, Any]]:
    data = load_json(path)
    return {item.get("id"): item for item in data}


def target_drugs(item: dict[str, Any]) -> list[str]:
    value = item.get("target_drugs") or item.get("target_drug") or []
    if isinstance(value, str):
        return [value]
    return [str(drug) for drug in value if str(drug).strip()]


def matched_hits(enriched_item: dict[str, Any]) -> list[dict[str, Any]]:
    skill = enriched_item.get("medsafe_skill") or {}
    return [hit for hit in skill.get("retrieved_cards", []) if hit.get("matched")]


def hit_drug(hit: dict[str, Any]) -> str:
    card = hit.get("card") or {}
    return str(card.get("drug") or hit.get("drug") or "")


def is_target_hit(hit: dict[str, Any], targets: list[str]) -> bool:
    if not targets:
        return True
    hit_name = normalize_text(hit_drug(hit))
    return any(hit_name == normalize_text(target) for target in targets)


def strongest(values: list[str], rank: dict[str, int], default: str) -> str:
    if not values:
        return default
    return max(values, key=lambda value: rank.get(value, -1))


def card_line(hit: dict[str, Any]) -> str:
    card = hit.get("card") or {}
    safety = card.get("safety") or {}
    source = card.get("source_trace") or {}
    csrp = hit.get("csrp") or {}
    return (
        f"- {card.get('drug')}：妊娠分级={safety.get('pregnancy_category') or '未报告'}，"
        f"哺乳分级={safety.get('breastfeeding_category') or '未报告'}，"
        f"系统安全倾向={ACTION_ZH.get(csrp.get('action_label'), csrp.get('action_label'))}，"
        f"建议复核={REVIEW_ZH.get(csrp.get('review_level'), csrp.get('review_level'))}，"
        f"来源页码=文档{source.get('page_doc') or 'NA'}/PDF{source.get('page_pdf') or 'NA'}。"
    )


def render_answer(query_meta: dict[str, Any], enriched_item: dict[str, Any]) -> dict[str, Any]:
    qid = query_meta.get("id")
    query = query_meta.get("query", enriched_item.get("query", ""))
    stress_type = query_meta.get("stress_type")
    targets = target_drugs(query_meta)

    base: dict[str, Any] = {
        "id": qid,
        "method": "medsafe_skill",
        "query": query,
        "stress_type": stress_type,
        "difficulty": query_meta.get("difficulty"),
        "target_drugs": targets,
        "expected_behavior": query_meta.get("expected_behavior"),
    }

    if "error" in enriched_item:
        base["answer"] = (
            "本题未能生成可靠的证据约束回答，因此在重新运行或人工复核前，"
            "不应给出妊娠期或哺乳期用药安全结论。"
        )
        base["error"] = enriched_item["error"]
        return base

    hits = matched_hits(enriched_item)
    target_hits = [hit for hit in hits if is_target_hit(hit, targets)]
    selected = target_hits if target_hits else ([] if targets else hits[:3])

    retrieved_names = [hit_drug(hit) for hit in hits[:10]]
    base["retrieved_cards"] = retrieved_names
    base["selected_cards"] = [
        {
            "rank": hit.get("rank"),
            "drug": hit_drug(hit),
            "card_id": (hit.get("card") or {}).get("card_id"),
            "action_label": (hit.get("csrp") or {}).get("action_label"),
            "review_level": (hit.get("csrp") or {}).get("review_level"),
        }
        for hit in selected
    ]
    base["unmatched_target_drugs"] = [] if selected else targets

    if targets and not selected:
        base["answer"] = (
            f"问题：{query}\n"
            f"回答：未在本研究证据库中确认与目标药物/产品 "
            f"{'、'.join(targets)} 完全匹配的记录，因此不能给出妊娠期或哺乳期安全结论。\n"
            "处理建议：证据不足，需要人工核对药品通用名、具体成分和来源后再判断。\n"
            "注意：不能用其他相似候选药物替代该目标药物，也不应编造妊娠分级、乳汁转运、剂量间隔、发生率或指南推荐。"
        )
        return base

    action = strongest(
        [(hit.get("csrp") or {}).get("action_label") for hit in selected],
        ACTION_RANK,
        "insufficient",
    )
    review = strongest(
        [(hit.get("csrp") or {}).get("review_level") for hit in selected],
        REVIEW_RANK,
        "pharmacist_review",
    )

    intro = "回答基于已匹配的药物证据记录生成。"
    if stress_type == "unsupported_detail":
        intro = (
            "可报告源证据支持的一般安全信息；但题目要求的精确发生率、浓度曲线、"
            "风险比、严格时间窗等细节，如源证据未提供，则不应外推或编造。"
        )
    elif stress_type == "misleading_premise":
        intro = "不能直接接受题干中的安全性前提；应按源证据进行保守判定。"
    else:
        intro = "回答基于已匹配的药物证据记录生成。"

    lines = [
        f"问题：{query}",
        f"回答：{intro}",
        f"总体安全倾向：{ACTION_ZH.get(action, action)}；建议复核：{REVIEW_ZH.get(review, review)}。",
        "证据摘要：",
    ]
    lines.extend(card_line(hit) for hit in selected[:5])
    lines.append(
        "结论：仅可使用上述源证据明确支持的信息。若问题要求的具体数值、研究样本量、"
        "停奶小时数或个体化方案未出现在源证据中，应回答证据不足并交由医生/药师复核。"
    )
    base["answer"] = "\n".join(lines)
    base["overall_action_label"] = action
    base["overall_review_level"] = review
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MedSafe-Skill outputs into answers for blinded review.")
    parser.add_argument("--query-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = query_items_by_id(args.query_path)
    enriched_items = load_json(args.skill_output)

    rendered = []
    for item in enriched_items:
        qid = item.get("id")
        meta = queries.get(qid, {"id": qid, "query": item.get("query", "")})
        rendered.append(render_answer(meta, item))

    save_json(args.output, rendered)
    print(json.dumps({"rendered_count": len(rendered), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
