#!/usr/bin/env python
"""Run a deterministic MedSafe-Skill demo for one query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.orchestrator import execute_skill_plan  # noqa: E402
from medsafe_skill.core.qasp_state_machine import finalize_after_claim_verification  # noqa: E402
from apply_csrp_rules import detect_context  # noqa: E402
from load_card_store import EvidenceCardStore  # noqa: E402
from verify_claim_support import split_claims, verify_claims  # noqa: E402


def make_draft_answer(query: str, cards: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> str:
    lines = [f"问题：{query}"]
    for card, decision in zip(cards, decisions):
        drug = card.get("identity", {}).get("canonical_name")
        source = card.get("source_trace", {})
        pregnancy = card.get("safety", {}).get("pregnancy", {})
        lactation = card.get("safety", {}).get("lactation", {})
        lines.append(
            f"{drug}：CSRP标签为{decision['action_label']}（{decision['color_label']}），"
            f"妊娠分级为{pregnancy.get('category') or '未知'}，"
            f"哺乳分级为{lactation.get('category') or '未知'}，"
            f"建议复核级别为{decision['review_level']}，"
            f"证据页码为文档第{source.get('page_doc')}页/PDF第{source.get('page_pdf')}页。"
        )
    lines.append("本回答仅为基于指南证据的草稿，高风险或个体化用药需由医生或药师确认。")
    return "\n".join(lines)


def is_verifiable_claim(claim: str) -> bool:
    return not (
        claim.startswith("问题：")
        or claim.startswith("本回答仅")
        or claim.startswith("高风险或个体化用药")
    )


def run_demo(query: str, drug_names: list[str], limit: int) -> dict[str, Any]:
    store = EvidenceCardStore.from_jsonl()
    cards = []
    for name in drug_names:
        card = store.lookup_name(name)
        if card:
            cards.append(card)
    if not cards:
        cards = store.search_text(query, limit=limit)

    context = detect_context(query)
    hits = [
        {"matched": True, "card": {"card_id": card.get("card_id")}}
        for card in cards
    ]
    plan = execute_skill_plan(
        query=query,
        retrieved_hits=hits,
        store=store,
        max_cards=limit,
    )
    decisions = plan["csrp"]
    draft_answer = make_draft_answer(query, cards, decisions)
    claims = [claim for claim in split_claims(draft_answer) if is_verifiable_claim(claim)]
    verification = verify_claims(claims, cards)
    workflow = finalize_after_claim_verification(
        plan["clinical_workflow"],
        all_supported=all(item.get("supported") for item in verification),
    )

    return {
        "query": query,
        "query_context": context,
        "retrieved_cards": [store.card_brief(card) for card in cards],
        "csrp_decisions": decisions,
        "execution_plan": plan,
        "clinical_workflow": workflow,
        "draft_answer": draft_answer,
        "claim_verification": verification,
        "unsupported_claims": [
            item["claim"] for item in verification if not item.get("supported")
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MedSafe-Skill demo.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--drug", action="append", default=[], help="Optional drug name. Can repeat.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args.query, args.drug, args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
