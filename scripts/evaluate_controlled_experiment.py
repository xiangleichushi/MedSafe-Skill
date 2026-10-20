#!/usr/bin/env python
"""Compute technical metrics for controlled MedSafe answer experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.claim_ledger import numeric_tokens, normalize_span  # noqa: E402
from medsafe_skill.scripts.load_card_store import EvidenceCardStore  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def card_text(card: dict[str, Any]) -> str:
    return json.dumps({
        "identity": card.get("identity"),
        "clinical_profile": card.get("clinical_profile"),
        "safety": card.get("safety"),
        "pharmacokinetics": card.get("pharmacokinetics"),
        "source_trace": card.get("source_trace"),
    }, ensure_ascii=False)


def numeric_support(answer: str, query: str, cards: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    corpus = normalize_span(query + " " + " ".join(card_text(card) for card in cards))
    tokens = numeric_tokens(answer)
    unsupported = [token for token in tokens if normalize_span(token) not in corpus]
    return len(tokens), len(unsupported), unsupported


def f1_counts(predicted: list[str], expected: list[str]) -> tuple[int, int, int]:
    pred = set(predicted)
    gold = set(expected)
    return len(pred & gold), len(pred - gold), len(gold - pred)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate_item(
    result: dict[str, Any],
    key: dict[str, Any],
    store: EvidenceCardStore,
) -> dict[str, Any]:
    target_cards = [store.lookup_name(str(drug), fuzzy=False) for drug in key.get("gold_drug") or []]
    target_ids = {card.get("card_id") for card in target_cards if card}
    retrieved_ids = set(result.get("retrieved_card_ids") or [])
    retrieved_cards = [store.get(card_id) for card_id in retrieved_ids]
    retrieved_cards = [card for card in retrieved_cards if card]
    answer = str(result.get("answer") or "")
    numeric_count, unsupported_numeric_count, unsupported_numeric = numeric_support(
        answer, str(result.get("query") or ""), retrieved_cards
    )
    usage = result.get("usage") or {}
    row: dict[str, Any] = {
        "id": result.get("id"),
        "group": result.get("group"),
        "scenario": key.get("scenario") or key.get("stress_type"),
        "error": int(bool(result.get("error"))),
        "answer_chars": len(answer),
        "retrieval_target_recall": ratio(len(target_ids & retrieved_ids), len(target_ids)),
        "citation_present": int(bool(re.search(r"指南.{0,30}第?.{0,15}页", answer))),
        "numeric_claim_count": numeric_count,
        "unsupported_numeric_count": unsupported_numeric_count,
        "unsupported_numeric_rate": ratio(unsupported_numeric_count, numeric_count),
        "unsupported_numeric_tokens": ";".join(unsupported_numeric),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "annotation_status": key.get("annotation_status"),
    }

    plan = result.get("plan") or {}
    route = plan.get("route") or {}
    selected = plan.get("selected_skills") or []
    expected = key.get("required_skills") or []
    if plan:
        tp, fp, fn = f1_counts(selected, expected)
        row.update({
            "skill_tp": tp,
            "skill_fp": fp,
            "skill_fn": fn,
            "skill_exact_match": int(set(selected) == set(expected)),
            "router_context_correct": int(route.get("context") == key.get("context")),
            "response_mode": plan.get("response_mode"),
            "response_mode_acceptable": int(plan.get("response_mode") in (key.get("acceptable_response_modes") or [])),
            "evidence_unsupported_fact_count": len((plan.get("evidence_sufficiency") or {}).get("unsupported_facts") or []),
        })
    verification = result.get("claim_verification") or {}
    if verification:
        row.update({
            "ledger_claim_count": verification.get("claim_count"),
            "ledger_supported_count": verification.get("supported_count"),
            "ledger_unsupported_count": verification.get("unsupported_count"),
            "ledger_support_rate": ratio(
                int(verification.get("supported_count") or 0),
                int(verification.get("claim_count") or 0),
            ),
            "revision_count": result.get("revision_count", 0),
        })
    return row


def aggregate(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row.get("skill_tp") or 0) for row in rows)
    fp = sum(int(row.get("skill_fp") or 0) for row in rows)
    fn = sum(int(row.get("skill_fn") or 0) for row in rows)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) not in (None, "")]

    numeric_total = sum(int(row.get("numeric_claim_count") or 0) for row in rows)
    unsupported_numeric = sum(int(row.get("unsupported_numeric_count") or 0) for row in rows)
    ledger_total = sum(int(row.get("ledger_claim_count") or 0) for row in rows)
    ledger_supported = sum(int(row.get("ledger_supported_count") or 0) for row in rows)
    return {
        "group": group,
        "n": len(rows),
        "error_rate": ratio(sum(int(row["error"]) for row in rows), len(rows)),
        "mean_answer_chars": mean(values("answer_chars")),
        "mean_retrieval_target_recall": mean(values("retrieval_target_recall")),
        "citation_coverage": ratio(sum(int(row["citation_present"]) for row in rows), len(rows)),
        "unsupported_numeric_rate": ratio(unsupported_numeric, numeric_total),
        "mean_elapsed_seconds": mean(values("elapsed_seconds")),
        "mean_total_tokens": mean(values("total_tokens")),
        "skill_selection_micro_precision": precision,
        "skill_selection_micro_recall": recall,
        "skill_selection_micro_f1": f1,
        "skill_exact_match_rate": mean(values("skill_exact_match")),
        "router_context_accuracy": mean(values("router_context_correct")),
        "response_mode_accuracy": mean(values("response_mode_acceptable")),
        "claim_ledger_support_rate": ratio(ledger_supported, ledger_total),
        "mean_revision_count": mean(values("revision_count")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=MEDSAFE_DIR / "output" / "controlled_answer_experiment.json")
    parser.add_argument("--key", type=Path, default=MEDSAFE_DIR / "experiments" / "skill_benchmark_key_draft.json")
    parser.add_argument("--cards", type=Path, default=MEDSAFE_DIR / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--output-dir", type=Path, default=MEDSAFE_DIR / "report_controlled_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_items = {item.get("id"): item for item in load_json(args.key)}
    store = EvidenceCardStore.from_jsonl(args.cards)
    rows = []
    for result in load_json(args.answers):
        key = key_items.get(result.get("id"))
        if key:
            rows.append(evaluate_item(result, key, store))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group"))].append(row)
    summaries = [aggregate(group, group_rows) for group, group_rows in sorted(groups.items())]
    statuses = sorted({str(row.get("annotation_status")) for row in rows})
    report = {
        "key_annotation_statuses": statuses,
        "warning": (
            "Router and response-mode metrics are provisional until the hidden key is independently reviewed."
            if any("draft" in status for status in statuses)
            else None
        ),
        "by_group": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "technical_metrics.json", report)
    write_csv(args.output_dir / "technical_metrics_by_group.csv", summaries)
    write_csv(args.output_dir / "technical_metrics_per_answer.csv", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
