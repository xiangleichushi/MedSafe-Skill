#!/usr/bin/env python
"""Run QASP without FAISS, using a validated observation cache or fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.orchestrator import execute_skill_plan  # noqa: E402
from medsafe_skill.scripts.load_card_store import EvidenceCardStore  # noqa: E402
from medsafe_skill.scripts.run_answer_experiment import observation_record_is_compatible  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--drug", action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--cards",
        type=Path,
        default=MEDSAFE_DIR / "resources" / "evidence_cards.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--observation-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EvidenceCardStore.from_jsonl(args.cards)
    cards = []
    for name in args.drug:
        card = store.lookup_name(name)
        if card and card.get("card_id") not in {item.get("card_id") for item in cards}:
            cards.append(card)
    if not cards:
        cards = store.search_text(args.query, limit=args.limit)

    hits = [
        {"rank": rank, "matched": True, "card": {"card_id": card.get("card_id")}}
        for rank, card in enumerate(cards, start=1)
    ]
    validated_observations = None
    observation_cache_hit = False
    if args.observation_cache:
        with args.observation_cache.open("r", encoding="utf-8") as f:
            cache = json.load(f)
        key = hashlib.sha256(args.query.encode("utf-8")).hexdigest()
        record = cache.get(key) if isinstance(cache, dict) else None
        if observation_record_is_compatible(record):
            validated_observations = record.get("validated_observations")
            observation_cache_hit = True

    plan = execute_skill_plan(
        query=args.query,
        retrieved_hits=hits,
        store=store,
        max_cards=args.limit,
        validated_observations=validated_observations,
    )
    payload = {
        "query": args.query,
        "retrieved_card_ids": [card.get("card_id") for card in cards],
        "observation_cache_hit": observation_cache_hit,
        "execution_plan": plan,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
