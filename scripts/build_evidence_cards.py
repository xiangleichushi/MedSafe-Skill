#!/usr/bin/env python
"""Build MedSafe Evidence Cards from extracted medication records.

The source file is expected to follow the current project format:

[
  {
    "drug": "...",
    "start_page_doc": 1,
    "end_page_doc": 2,
    "start_page_pdf": 27,
    "end_page_pdf": 28,
    "extracted": {
      "drug": "...",
      "drug_category": "...",
      ...
      "evidence": {"source": "...", "page_doc": "...", "page_pdf": "..."}
    }
  }
]

Outputs:
- resources/evidence_cards.jsonl
- resources/evidence_card_index.json
- evidence_cards/<card_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
DEFAULT_SOURCE = Path("output/extracted_all_repaired.json")
DEFAULT_SKILL_DIR = Path("medsafe_skill")


def as_list(value: Any) -> list[Any]:
    """Return a list while preserving meaningful scalar values."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, tuple):
        return [item for item in value if item not in (None, "")]
    if value == "":
        return []
    return [value]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_list(value: Any) -> list[str]:
    return [clean_text(item) for item in as_list(value) if clean_text(item)]


def normalize_dosage(value: Any) -> list[dict[str, str]]:
    rows = []
    for item in as_list(value):
        if not isinstance(item, dict):
            text = clean_text(item)
            if text:
                rows.append({"indication": "", "route": "", "dose": text})
            continue
        rows.append(
            {
                "indication": clean_text(item.get("indication")),
                "route": clean_text(item.get("route")),
                "dose": clean_text(item.get("dose")),
            }
        )
    return rows


def normalize_pk(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "molecular_weight": clean_text(value.get("molecular_weight")),
        "protein_binding": clean_text(value.get("protein_binding")),
        "half_life": clean_text(value.get("half_life")),
        "volume_of_distribution": clean_text(value.get("volume_of_distribution")),
        "placental_transfer": clean_text(value.get("placental_transfer")),
        "milk_plasma_ratio": clean_text(value.get("milk_plasma_ratio")),
        "other": normalize_list(value.get("other")),
    }


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def make_card_id(record_index: int, drug_name: str) -> str:
    digest = short_hash(drug_name or str(record_index))
    return f"medsafe-guide2-{record_index + 1:04d}-{digest}"


def split_compound_components(name: str) -> list[str]:
    """Heuristically split combination products without over-normalizing."""
    if not name:
        return []
    separators = ["/", "＋", "+", "和", "加"]
    pattern = "|".join(re.escape(sep) for sep in separators)
    parts = [part.strip() for part in re.split(pattern, name) if part.strip()]
    if len(parts) <= 1:
        return []
    return parts


def normalize_pregnancy_risk(category: str) -> str:
    """Rule-derived pregnancy risk bucket, not a clinical final decision."""
    cat = category.strip().upper()
    if not cat:
        return "unknown"
    if "X" in cat:
        return "contraindicated_or_high_risk"
    if "D" in cat:
        return "known_risk_or_specialist_use"
    if "C" in cat:
        return "caution"
    if "A" in cat or "B" in cat:
        return "lower_risk_in_category_system"
    return "context_dependent"


def normalize_lactation_action(category: str) -> str:
    """Map source breastfeeding category to a bounded action label.

    Labels are conservative and intended for downstream CSRP rules:
    - continue
    - caution
    - avoid
    - insufficient
    - context_dependent
    """
    raw = category.strip()
    cat = raw.upper().replace(" ", "")
    if not cat:
        return "insufficient"

    starts_s = cat.startswith("S")
    starts_ns = cat.startswith("NS")
    starts_u = cat.startswith("U")

    if starts_ns:
        return "avoid"
    if starts_u:
        return "insufficient"
    if starts_s:
        cautious_markers = ["可能", "POSSIB", "LIKELY", "PROBAB", "似乎", "局部", "仅当"]
        if any(marker.upper() in cat for marker in cautious_markers):
            return "caution"
        return "continue"
    if "NS" in cat and "S" not in cat:
        return "avoid"
    if "U" in cat and "S" not in cat:
        return "insufficient"
    if cat in {"C", "V", "NC"}:
        return "caution"
    return "context_dependent"


def derive_risk_flags(
    pregnancy_category: str,
    breastfeeding_category: str,
    contraindications: list[str],
    cautions: list[str],
    interactions: list[str],
) -> dict[str, Any]:
    preg_risk = normalize_pregnancy_risk(pregnancy_category)
    lact_action = normalize_lactation_action(breastfeeding_category)

    high_risk_terms = [
        "妊娠",
        "肝功能不全",
        "肾功能不全",
        "心力衰竭",
        "出血",
        "骨髓",
        "免疫",
        "新生儿",
        "早产",
        "致畸",
        "毒性",
        "化疗",
        "细胞毒",
        "呼吸抑制",
    ]
    joined = " ".join(contraindications + cautions + interactions)
    matched_terms = [term for term in high_risk_terms if term in joined]

    specialist_terms = {
        "骨髓",
        "免疫",
        "新生儿",
        "早产",
        "化疗",
        "细胞毒",
        "呼吸抑制",
    }
    needs_specialist = (
        preg_risk in {"contraindicated_or_high_risk", "known_risk_or_specialist_use"}
        or lact_action == "avoid"
        or any(term in specialist_terms for term in matched_terms)
    )
    needs_pharmacist = (
        lact_action in {"caution", "insufficient", "context_dependent"}
        or preg_risk == "caution"
        or bool(matched_terms)
    )

    if needs_specialist:
        review_level = "specialist_review"
    elif needs_pharmacist:
        review_level = "pharmacist_review"
    else:
        review_level = "standard_clinician_confirmation"

    review_reasons = []
    if preg_risk in {"contraindicated_or_high_risk", "known_risk_or_specialist_use"}:
        review_reasons.append(f"pregnancy_risk:{preg_risk}")
    if lact_action in {"avoid", "caution", "insufficient", "context_dependent"}:
        review_reasons.append(f"lactation_action:{lact_action}")
    if matched_terms:
        review_reasons.append("matched_terms:" + ",".join(matched_terms))

    return {
        "pregnancy_risk": preg_risk,
        "lactation_action": lact_action,
        "matched_high_risk_terms": matched_terms,
        "recommended_review_level": review_level,
        "review_reasons": review_reasons,
        "requires_human_review": review_level != "standard_clinician_confirmation",
    }


def build_card(
    record: dict[str, Any],
    record_index: int,
    build_time: str,
    source_dataset: Path,
) -> dict[str, Any]:
    extracted = record.get("extracted") if isinstance(record.get("extracted"), dict) else {}

    canonical_name = clean_text(extracted.get("drug")) or clean_text(record.get("drug"))
    card_id = make_card_id(record_index, canonical_name)

    pregnancy_category = clean_text(extracted.get("pregnancy_category"))
    breastfeeding_category = clean_text(extracted.get("breastfeeding_category"))
    contraindications = normalize_list(extracted.get("contraindication"))
    cautions = normalize_list(extracted.get("caution"))
    interactions = normalize_list(extracted.get("drug_interaction"))
    risk_flags = derive_risk_flags(
        pregnancy_category,
        breastfeeding_category,
        contraindications,
        cautions,
        interactions,
    )

    evidence = extracted.get("evidence") if isinstance(extracted.get("evidence"), dict) else {}

    return {
        "card_id": card_id,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "canonical_name": canonical_name,
            "source_name": clean_text(record.get("drug")),
            "aliases": [],
            "compound_components": split_compound_components(canonical_name),
        },
        "clinical_profile": {
            "drug_category": clean_text(extracted.get("drug_category")),
            "indications": normalize_list(extracted.get("indication")),
            "mechanism": clean_text(extracted.get("mechanism")),
            "dosage": normalize_dosage(extracted.get("dosage")),
        },
        "safety": {
            "pregnancy": {
                "category": pregnancy_category,
                "normalized_risk": risk_flags["pregnancy_risk"],
                "maternal_effects": normalize_list(extracted.get("effect_on_mother")),
                "fetal_effects": normalize_list(extracted.get("effect_on_fetus")),
            },
            "lactation": {
                "category": breastfeeding_category,
                "normalized_action": risk_flags["lactation_action"],
                "safety_summary": normalize_list(extracted.get("breastfeeding_safety")),
            },
            "contraindications": contraindications,
            "cautions": cautions,
            "drug_interactions": interactions,
            "risk_flags": risk_flags,
        },
        "pharmacokinetics": normalize_pk(extracted.get("pharmacokinetics")),
        "source_trace": {
            "source": clean_text(evidence.get("source")),
            "page_doc": clean_text(evidence.get("page_doc")),
            "page_pdf": clean_text(evidence.get("page_pdf")),
            "start_page_doc": record.get("start_page_doc"),
            "end_page_doc": record.get("end_page_doc"),
            "start_page_pdf": record.get("start_page_pdf"),
            "end_page_pdf": record.get("end_page_pdf"),
            "references": normalize_list(extracted.get("references")),
        },
        "verification": {
            "support_scope": "field_level_source_extraction",
            "unsupported_claim_policy": "remove_or_abstain",
            "requires_clinician_review": risk_flags["requires_human_review"],
            "recommended_review_level": risk_flags["recommended_review_level"],
            "notes": [
                "Normalized labels are rule-derived from extracted source fields and are not final clinical judgments.",
                "Answers must cite source_trace and avoid claims not supported by card fields.",
            ],
        },
        "metadata": {
            "input_record_index": record_index,
            "source_dataset": str(source_dataset),
            "builder": "medsafe_skill/scripts/build_evidence_cards.py",
            "built_at_utc": build_time,
        },
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def build_index(cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_card_id = {}
    by_drug_name = {}
    duplicate_names: dict[str, list[str]] = {}

    for card in cards:
        card_id = card["card_id"]
        name = card["identity"]["canonical_name"]
        filename = f"evidence_cards/{card_id}.json"
        by_card_id[card_id] = {
            "canonical_name": name,
            "file": filename,
            "page_doc": card["source_trace"]["page_doc"],
            "page_pdf": card["source_trace"]["page_pdf"],
            "lactation_action": card["safety"]["lactation"]["normalized_action"],
            "pregnancy_risk": card["safety"]["pregnancy"]["normalized_risk"],
        }
        by_drug_name.setdefault(name, []).append(card_id)

    for name, ids in by_drug_name.items():
        if len(ids) > 1:
            duplicate_names[name] = ids

    return {
        "schema_version": SCHEMA_VERSION,
        "card_count": len(cards),
        "by_card_id": by_card_id,
        "by_drug_name": by_drug_name,
        "duplicate_names": duplicate_names,
    }


def build_summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    lactation_actions = Counter(
        card["safety"]["lactation"]["normalized_action"] for card in cards
    )
    pregnancy_risks = Counter(
        card["safety"]["pregnancy"]["normalized_risk"] for card in cards
    )
    review_levels = Counter(
        card["verification"]["recommended_review_level"] for card in cards
    )
    human_review_count = sum(
        1 for card in cards if card["verification"]["requires_clinician_review"]
    )
    missing_evidence = [
        card["card_id"]
        for card in cards
        if not card["source_trace"]["page_doc"] or not card["source_trace"]["page_pdf"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "card_count": len(cards),
        "lactation_actions": dict(lactation_actions),
        "pregnancy_risks": dict(pregnancy_risks),
        "review_levels": dict(review_levels),
        "requires_clinician_review_count": human_review_count,
        "missing_evidence_count": len(missing_evidence),
        "missing_evidence_card_ids": missing_evidence[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted medication JSON into MedSafe Evidence Cards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to extracted_all_repaired.json.",
    )
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=DEFAULT_SKILL_DIR,
        help="Output medsafe_skill directory.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also pretty-print each card file. JSONL is always compact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    build_time = datetime.now(timezone.utc).isoformat()

    cards = [
        build_card(record, idx, build_time, args.input)
        for idx, record in enumerate(records)
    ]

    resources_dir = args.skill_dir / "resources"
    cards_dir = args.skill_dir / "evidence_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for stale_card in cards_dir.glob("medsafe-guide2-*.json"):
        stale_card.unlink()

    write_jsonl(resources_dir / "evidence_cards.jsonl", cards)
    write_json(resources_dir / "evidence_card_index.json", build_index(cards))
    write_json(resources_dir / "evidence_card_build_summary.json", build_summary(cards))

    for card in cards:
        out_path = cards_dir / f"{card['card_id']}.json"
        if args.pretty:
            write_json(out_path, card)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(card, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")

    summary = build_summary(cards)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
