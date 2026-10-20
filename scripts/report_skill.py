#!/usr/bin/env python
"""Self-contained MedSafe-Skill report generator.

This version lives inside medsafe_skill/ so the folder can be copied to AutoDL
as a standalone package. It evaluates enriched outputs produced by
run_skill_batch_from_hrc.py or attach_cards_to_hcr_results.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
DEFAULT_ANSWERS = MEDSAFE_DIR / "output" / "answer_skill_pipeline_top_10.json"
DEFAULT_QUERIES = SCRIPT_DIR.parents[1] / "query.json"
DEFAULT_OUTPUT_DIR = MEDSAFE_DIR / "report_skill"

ALLOWED_ACTION_LABELS = {"continue", "caution", "avoid", "insufficient", "ask_clarification"}
ALLOWED_REVIEW_LEVELS = {
    "standard_clinician_confirmation",
    "pharmacist_review",
    "specialist_review",
}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def has_citation(hit: dict[str, Any]) -> bool:
    csrp_citation = (hit.get("csrp") or {}).get("citation") or {}
    card_source = ((hit.get("card") or {}).get("source_trace") or {})
    page_doc = csrp_citation.get("page_doc") or card_source.get("page_doc")
    page_pdf = csrp_citation.get("page_pdf") or card_source.get("page_pdf")
    return bool(page_doc and page_pdf)


def claim_support_items(hit: dict[str, Any], query_context: str) -> list[dict[str, Any]]:
    card = hit.get("card") or {}
    csrp = hit.get("csrp") or {}
    safety = card.get("safety") or {}
    source = card.get("source_trace") or {}
    drug = card.get("drug") or hit.get("drug")

    items = [
        {
            "claim_type": "csrp_action_label",
            "claim": f"{drug}: action_label={csrp.get('action_label')}",
            "supported": bool(
                csrp.get("action_label") in ALLOWED_ACTION_LABELS
                and csrp.get("rationale")
                and has_citation(hit)
            ),
        },
        {
            "claim_type": "review_routing",
            "claim": f"{drug}: review_level={csrp.get('review_level')}",
            "supported": bool(csrp.get("review_level") in ALLOWED_REVIEW_LEVELS and has_citation(hit)),
        },
        {
            "claim_type": "source_trace",
            "claim": f"{drug}: page_doc={source.get('page_doc')}; page_pdf={source.get('page_pdf')}",
            "supported": has_citation(hit),
        },
    ]

    if query_context in {"pregnancy", "pregnancy_and_lactation", "unspecified"}:
        items.append(
            {
                "claim_type": "pregnancy_category",
                "claim": f"{drug}: pregnancy_category={safety.get('pregnancy_category')}",
                "supported": bool(
                    safety.get("pregnancy_category")
                    and safety.get("pregnancy_risk")
                    and has_citation(hit)
                ),
            }
        )
    if query_context in {"lactation", "pregnancy_and_lactation", "unspecified"}:
        items.append(
            {
                "claim_type": "lactation_category",
                "claim": f"{drug}: breastfeeding_category={safety.get('breastfeeding_category')}",
                "supported": bool(
                    safety.get("breastfeeding_category")
                    and safety.get("lactation_action")
                    and has_citation(hit)
                ),
            }
        )
    return items


def evaluate_query(item: dict[str, Any], query_meta: dict[str, Any]) -> dict[str, Any]:
    skill = item.get("medsafe_skill") or {}
    hits = skill.get("retrieved_cards") or []
    matched_hits = [hit for hit in hits if hit.get("matched")]
    top_k = int(skill.get("top_k") or len(hits) or 0)
    query_context = skill.get("query_context") or "unspecified"

    action_labels = []
    review_levels = []
    citation_flags = []
    claim_items = []
    for hit in matched_hits:
        csrp = hit.get("csrp") or {}
        action_labels.append(csrp.get("action_label"))
        review_levels.append(csrp.get("review_level"))
        citation_flags.append(has_citation(hit))
        claim_items.extend(claim_support_items(hit, query_context))

    supported_claims = sum(1 for claim in claim_items if claim["supported"])
    total_claims = len(claim_items)
    valid_actions = sum(1 for label in action_labels if label in ALLOWED_ACTION_LABELS)
    valid_reviews = sum(1 for level in review_levels if level in ALLOWED_REVIEW_LEVELS)
    review_required = sum(1 for level in review_levels if level != "standard_clinician_confirmation")
    specialist_review = sum(1 for level in review_levels if level == "specialist_review")
    insufficient = sum(1 for label in action_labels if label in {"insufficient", "ask_clarification"})

    return {
        "id": item.get("id"),
        "query": item.get("query"),
        "type_id": query_meta.get("type_id"),
        "type_name": query_meta.get("type_name"),
        "difficulty": query_meta.get("difficulty"),
        "query_context": query_context,
        "top_k": top_k,
        "matched_card_count": len(matched_hits),
        "card_match_rate": safe_div(len(matched_hits), top_k),
        "citation_coverage": mean([1.0 if flag else 0.0 for flag in citation_flags]),
        "action_label_coverage": safe_div(valid_actions, len(matched_hits)),
        "review_routing_coverage": safe_div(valid_reviews, len(matched_hits)),
        "claim_support_rate": safe_div(supported_claims, total_claims),
        "unsupported_claim_rate": 1.0 - safe_div(supported_claims, total_claims),
        "review_required_rate": safe_div(review_required, len(matched_hits)),
        "specialist_review_rate": safe_div(specialist_review, len(matched_hits)),
        "insufficient_or_abstain_rate": safe_div(insufficient, len(matched_hits)),
        "action_labels": dict(Counter(action_labels)),
        "review_levels": dict(Counter(review_levels)),
        "claim_counts": {"supported": supported_claims, "total": total_claims},
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "card_match_rate",
        "citation_coverage",
        "action_label_coverage",
        "review_routing_coverage",
        "claim_support_rate",
        "unsupported_claim_rate",
        "review_required_rate",
        "specialist_review_rate",
        "insufficient_or_abstain_rate",
    ]
    return {name: mean([float(row.get(name, 0.0)) for row in rows]) for name in metric_names}


def aggregate_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get(field) or {})
    return dict(counter)


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return dict(groups)


def evaluate(answers: list[dict[str, Any]], queries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    skipped = []
    for item in answers:
        if "error" in item:
            continue
        qid = item.get("id")
        meta = queries.get(qid, {})
        if not meta:
            skipped.append(qid)
        rows.append(evaluate_query(item, meta))

    per_difficulty = {
        name: {
            "query_count": len(group),
            "metrics": aggregate_metrics(group),
            "action_labels": aggregate_counts(group, "action_labels"),
            "review_levels": aggregate_counts(group, "review_levels"),
        }
        for name, group in group_rows(rows, "difficulty").items()
    }

    total_claims = sum(row["claim_counts"]["total"] for row in rows)
    supported_claims = sum(row["claim_counts"]["supported"] for row in rows)

    return {
        "overall": {
            "query_count": len(rows),
            "skipped_query_ids": skipped,
            "matched_card_hits": sum(row["matched_card_count"] for row in rows),
            "field_level_claims": {
                "supported": supported_claims,
                "total": total_claims,
                "support_rate": safe_div(supported_claims, total_claims),
            },
            "metrics": aggregate_metrics(rows),
            "action_labels": aggregate_counts(rows, "action_labels"),
            "review_levels": aggregate_counts(rows, "review_levels"),
        },
        "per_difficulty": per_difficulty,
        "per_query": rows,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skill_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "overall_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in report["overall"]["metrics"].items():
            writer.writerow([key, f"{value:.6f}"])

    fields = [
        "id",
        "type_name",
        "difficulty",
        "query_context",
        "matched_card_count",
        "card_match_rate",
        "citation_coverage",
        "action_label_coverage",
        "review_routing_coverage",
        "claim_support_rate",
        "review_required_rate",
        "specialist_review_rate",
        "insufficient_or_abstain_rate",
    ]
    with (output_dir / "per_query_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["per_query"]:
            writer.writerow({field: row.get(field) for field in fields})

    metrics = report["overall"]["metrics"]
    claims = report["overall"]["field_level_claims"]
    lines = [
        "# MedSafe-Skill Safety Evaluation Summary",
        "",
        f"- Query count: {report['overall']['query_count']}",
        f"- Matched card hits: {report['overall']['matched_card_hits']}",
        f"- Evidence Card match rate: {metrics['card_match_rate']:.4f}",
        f"- Citation coverage: {metrics['citation_coverage']:.4f}",
        f"- CSRP action-label coverage: {metrics['action_label_coverage']:.4f}",
        f"- Review-routing coverage: {metrics['review_routing_coverage']:.4f}",
        f"- Field-level claim support rate: {claims['support_rate']:.4f} ({claims['supported']}/{claims['total']})",
        f"- Specialist-review routing rate: {metrics['specialist_review_rate']:.4f}",
        f"- Insufficient/abstention label rate: {metrics['insufficient_or_abstain_rate']:.4f}",
        "",
        "## CSRP Action Labels",
    ]
    for label, count in report["overall"]["action_labels"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Human Review Routing"])
    for level, count in report["overall"]["review_levels"].items():
        lines.append(f"- {level}: {count}")
    (output_dir / "skill_report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[MedSafe] matplotlib unavailable; skipped plots.")
        return

    metrics = report["overall"]["metrics"]
    keys = [
        "card_match_rate",
        "citation_coverage",
        "action_label_coverage",
        "review_routing_coverage",
        "claim_support_rate",
    ]
    labels = ["Card match", "Citation", "Action label", "Review routing", "Claim support"]
    values = [metrics.get(key, 0.0) for key in keys]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("MedSafe-Skill Overall Coverage Metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_skill_rates.png", bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MedSafe-Skill report.")
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers = load_json(args.answers)
    queries_raw = load_json(args.queries) if args.queries.exists() else []
    queries = {item["id"]: item for item in queries_raw if isinstance(item, dict) and "id" in item}
    report = evaluate(answers, queries)
    write_outputs(report, args.output_dir)
    if not args.no_plots:
        maybe_plot(report, args.output_dir)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"[MedSafe] Report saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
