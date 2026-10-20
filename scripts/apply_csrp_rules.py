#!/usr/bin/env python
"""Apply Clinical Safety Reasoning Protocol (CSRP) rules to Evidence Cards."""

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

from medsafe_skill.core.action_policy import (  # noqa: E402
    ACTION_COLOR,
    evaluate_card_action,
    worst_action,
)
from medsafe_skill.core.clinical_state import build_consultation_case  # noqa: E402
from medsafe_skill.core.router import detect_context as route_detect_context  # noqa: E402
from medsafe_skill.core.rulebook import load_rulebook  # noqa: E402
from load_card_store import EvidenceCardStore  # noqa: E402


ACTION_RANK = load_rulebook()["action_rank"]


def detect_context(query: str) -> str:
    return route_detect_context(query)[0]


def pregnancy_action(normalized_risk: str) -> str:
    return load_rulebook()["pregnancy_action_mapping"].get(normalized_risk, "undetermined")


def lactation_action(normalized_action: str) -> str:
    return load_rulebook()["lactation_action_mapping"].get(normalized_action, "undetermined")


def decision_rationale(card: dict[str, Any], context: str, action: str) -> list[str]:
    safety = card.get("safety", {})
    pregnancy = safety.get("pregnancy", {})
    lactation = safety.get("lactation", {})
    reasons = []
    if context in {"pregnancy", "pregnancy_and_lactation", "unspecified"}:
        reasons.append(
            f"pregnancy_category={pregnancy.get('category') or 'unknown'}; "
            f"pregnancy_risk={pregnancy.get('normalized_risk') or 'unknown'}"
        )
    if context in {"lactation", "pregnancy_and_lactation", "unspecified"}:
        reasons.append(
            f"breastfeeding_category={lactation.get('category') or 'unknown'}; "
            f"lactation_action={lactation.get('normalized_action') or 'unknown'}"
        )
    flags = safety.get("risk_flags", {})
    if flags.get("review_reasons"):
        reasons.extend(flags["review_reasons"])
    reasons.append(f"bounded_action={action}")
    return reasons


def apply_csrp_to_card(card: dict[str, Any], query: str = "", context: str = "auto") -> dict[str, Any]:
    if context == "auto":
        context = detect_context(query)
    case = build_consultation_case(
        query=query,
        reproductive_context=context,
        cards=[card],
    )
    return evaluate_card_action(card=card, case=case, all_cards=[card])


def apply_csrp(cards: list[dict[str, Any]], query: str = "", context: str = "auto") -> list[dict[str, Any]]:
    return [apply_csrp_to_card(card, query=query, context=context) for card in cards]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply CSRP rules to Evidence Cards.")
    parser.add_argument("--cards", type=Path, default=SCRIPT_DIR.parent / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--query", default="")
    parser.add_argument("--drug", action="append", default=[], help="Drug name. Can be repeated.")
    parser.add_argument(
        "--context",
        choices=["auto", "pregnancy", "lactation", "pregnancy_and_lactation", "unspecified"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EvidenceCardStore.from_jsonl(args.cards)
    cards = []
    for drug in args.drug:
        card = store.lookup_name(drug)
        if card:
            cards.append(card)
    if not cards and args.query:
        cards = store.search_text(args.query, limit=5)
    print(json.dumps(apply_csrp(cards, args.query, args.context), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
