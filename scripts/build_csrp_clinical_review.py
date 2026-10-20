#!/usr/bin/env python
"""Build a stratified clinical review sheet for CSRP action validation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.scripts.apply_csrp_rules import apply_csrp_to_card  # noqa: E402
from medsafe_skill.scripts.load_card_store import EvidenceCardStore  # noqa: E402


QUOTAS = {
    ("pregnancy", "continue"): 15,
    ("pregnancy", "caution"): 15,
    ("pregnancy", "avoid"): 9,
    ("pregnancy", "undetermined"): 1,
    ("lactation", "continue"): 10,
    ("lactation", "caution"): 10,
    ("lactation", "avoid"): 10,
    ("lactation", "undetermined"): 10,
}


def compact(values: Any, limit: int = 650) -> str:
    if isinstance(values, list):
        text = "；".join(str(value) for value in values if str(value).strip())
    elif isinstance(values, dict):
        text = json.dumps(values, ensure_ascii=False)
    else:
        text = str(values or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def evidence_text(card: dict[str, Any], context: str) -> str:
    safety = card.get("safety", {})
    if context == "pregnancy":
        pregnancy = safety.get("pregnancy", {})
        return compact({
            "maternal_effects": pregnancy.get("maternal_effects", []),
            "fetal_effects": pregnancy.get("fetal_effects", []),
            "contraindications": safety.get("contraindications", []),
            "cautions": safety.get("cautions", []),
        })
    lactation = safety.get("lactation", {})
    return compact({
        "safety_summary": lactation.get("safety_summary", []),
        "contraindications": safety.get("contraindications", []),
        "cautions": safety.get("cautions", []),
    })


def build_rows(store: EvidenceCardStore, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in QUOTAS}
    for card in store.cards:
        for context in ("pregnancy", "lactation"):
            result = apply_csrp_to_card(card, context=context)
            key = (context, result["action_label"])
            if key not in buckets:
                continue
            safety = card.get("safety", {})
            pregnancy = safety.get("pregnancy", {})
            lactation = safety.get("lactation", {})
            source = card.get("source_trace", {})
            buckets[key].append({
                "case_id": "",
                "drug": card.get("identity", {}).get("canonical_name"),
                "context": context,
                "pregnancy_category": pregnancy.get("category"),
                "pregnancy_normalized_risk": pregnancy.get("normalized_risk"),
                "lactation_category": lactation.get("category"),
                "lactation_normalized_action": lactation.get("normalized_action"),
                "evidence_excerpt": evidence_text(card, context),
                "guide_page": source.get("page_doc"),
                "pdf_page": source.get("page_pdf"),
                "system_action": result.get("action_label"),
                "system_review_level": result.get("review_level"),
                "system_rationale": " | ".join(result.get("rationale") or []),
                "expert_action": "",
                "action_agreement_0_1": "",
                "direction_if_disagree": "",
                "expert_review_level": "",
                "review_level_agreement_0_1": "",
                "clinical_comment": "",
            })

    selected = []
    for key, quota in QUOTAS.items():
        candidates = buckets[key]
        if len(candidates) < quota:
            raise ValueError(f"Not enough cases for {key}: requested={quota}, available={len(candidates)}")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"CSRP-{index:03d}"
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def instruction_text() -> str:
    return """# CSRP clinical validation instructions

This sheet validates the deterministic CSRP mapping, not the final AI answer.
For each row, review the source-derived evidence excerpt and the system action.

Fill the following fields:

- `expert_action`: continue / caution / avoid / undetermined.
- `action_agreement_0_1`: 1 if the system action is acceptable; otherwise 0.
- `direction_if_disagree`: system_too_permissive / system_too_conservative / wrong_context / other.
- `expert_review_level`: standard_clinician_confirmation / pharmacist_review / specialist_review.
- `review_level_agreement_0_1`: 1 if the system review level is acceptable; otherwise 0.
- `clinical_comment`: briefly state the reason for disagreement or important nuance.

Interpretation of actions:

- `continue`: evidence permits continuation/use in the stated context, subject to ordinary clinical confirmation.
- `caution`: use requires benefit-risk assessment, monitoring, stage/dose qualification, or pharmacist review.
- `avoid`: the evidence supports avoidance/contraindication or specialist-only exceptional use.
- `undetermined`: the card does not contain enough evidence for a source-bound medication-safety action.

Do not infer information from outside the displayed guide evidence when judging whether the deterministic mapping correctly represents this source. External clinical knowledge may be noted separately in `clinical_comment`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=MEDSAFE_DIR / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--output-dir", type=Path, default=MEDSAFE_DIR / "experiments" / "csrp_clinical_review")
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(EvidenceCardStore.from_jsonl(args.cards), args.seed)
    sheet = args.output_dir / "csrp_review_sheet.csv"
    instructions = args.output_dir / "csrp_review_instructions.md"
    write_csv(sheet, rows)
    instructions.write_text(instruction_text(), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['context']}:{row['system_action']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({
        "case_count": len(rows),
        "strata": counts,
        "review_sheet": str(sheet),
        "instructions": str(instructions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
