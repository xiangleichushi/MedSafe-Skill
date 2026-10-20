#!/usr/bin/env python
"""Attach Evidence Cards and CSRP labels to existing HRC-RAG result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
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
from apply_csrp_rules import detect_context  # noqa: E402
from load_card_store import EvidenceCardStore  # noqa: E402
from medsafe_skill.scripts.run_answer_experiment import observation_record_is_compatible  # noqa: E402


DEFAULT_INPUT = Path("answer_critic_top_20.json")
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "output" / "answer_critic_with_cards.json"


def result_drug_list(result: dict[str, Any], top_k: int) -> list[str]:
    merged: list[str] = []
    for key in [f"final_reranked_top_{top_k}", "final_reranked_top_20", "rerank_top_20"]:
        value = result.get(key)
        if isinstance(value, list) and value:
            merged.extend(str(item) for item in value)
            break

    history = result.get("critic_history") or []
    if isinstance(history, list) and history:
        last_round = history[-1]
        if isinstance(last_round, dict):
            merged.extend(str(item) for item in last_round.get("rrf_drugs", []) or [])

    for key in ["rrf_top_20", "name_match_top_20", "category_top_20", "indication_top_20"]:
        value = result.get(key)
        if isinstance(value, list) and value:
            merged.extend(str(item) for item in value)

    deduped = []
    seen = set()
    for drug in merged:
        if not drug or drug in seen:
            continue
        seen.add(drug)
        deduped.append(drug)
        if len(deduped) >= top_k:
            break
    return deduped


def attach_cards(
    hcr_results: list[dict[str, Any]],
    store: EvidenceCardStore,
    top_k: int,
    observation_cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched = []
    unmatched = []
    review_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    case_action_counter: Counter[str] = Counter()
    final_state_counter: Counter[str] = Counter()
    observation_source_counter: Counter[str] = Counter()

    for item in hcr_results:
        query = item.get("query", "")
        query_context = detect_context(query)
        drugs = result_drug_list(item, top_k=top_k)
        card_hits = []
        for rank, drug in enumerate(drugs, start=1):
            card = store.lookup_name(drug)
            if not card:
                unmatched.append({"query_id": item.get("id"), "query": query, "drug": drug})
                card_hits.append({"rank": rank, "drug": drug, "matched": False})
                continue

            brief = store.card_brief(card)
            card_hits.append(
                {
                    "rank": rank,
                    "drug": drug,
                    "matched": True,
                    "card": brief,
                }
            )

        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        observation_record = (observation_cache or {}).get(cache_key) or {}
        validated_observations = (
            observation_record.get("validated_observations")
            if observation_record_is_compatible(observation_record)
            else None
        )

        execution_plan = execute_skill_plan(
            query=query,
            retrieved_hits=card_hits,
            store=store,
            max_cards=top_k,
            validated_observations=validated_observations,
        )
        decisions_by_card = {
            decision.get("card_id"): decision
            for decision in execution_plan.get("csrp", [])
        }
        for hit in card_hits:
            if not hit.get("matched"):
                continue
            csrp = decisions_by_card.get(hit["card"].get("card_id"))
            if csrp:
                hit["csrp"] = csrp
                review_counter[csrp["review_level"]] += 1
                action_counter[csrp["action_label"]] += 1

        workflow = execution_plan["clinical_workflow"]
        action_plan = workflow["qasp"]["P_action_plan"]
        if action_plan.get("final_action"):
            case_action_counter[str(action_plan["final_action"])] += 1
        final_state_counter[str(workflow.get("current_state"))] += 1
        observation_source_counter[str(
            workflow["qasp"]["Q_question_clarification"].get("observation_source")
        )] += 1

        enriched_item = dict(item)
        enriched_item["medsafe_skill"] = {
            "query_context": query_context,
            "top_k": top_k,
            "retrieved_cards": card_hits,
            "retrieved_card_count": sum(1 for hit in card_hits if hit.get("matched")),
            "unmatched_drugs": [hit["drug"] for hit in card_hits if not hit.get("matched")],
            "execution_plan": execution_plan,
            "clinical_workflow": workflow,
        }
        enriched.append(enriched_item)

    summary = {
        "input_count": len(hcr_results),
        "top_k": top_k,
        "total_card_hits": sum(
            item["medsafe_skill"]["retrieved_card_count"] for item in enriched
        ),
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
        "action_labels": dict(action_counter),
        "review_levels": dict(review_counter),
        "case_action_labels": dict(case_action_counter),
        "workflow_final_states": dict(final_state_counter),
        "observation_sources": dict(observation_source_counter),
    }
    return enriched, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach MedSafe Evidence Cards to HRC-RAG outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cards", type=Path, default=SCRIPT_DIR.parent / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--observation-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        hcr_results = json.load(f)
    if not isinstance(hcr_results, list):
        raise ValueError(f"Expected list in {args.input}")

    store = EvidenceCardStore.from_jsonl(args.cards)
    observation_cache = {}
    if args.observation_cache:
        with args.observation_cache.open("r", encoding="utf-8") as f:
            observation_cache = json.load(f)
        if not isinstance(observation_cache, dict):
            raise ValueError("--observation-cache must contain a JSON object")
    enriched, summary = attach_cards(
        hcr_results,
        store,
        args.top_k,
        observation_cache=observation_cache,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
