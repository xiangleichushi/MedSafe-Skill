#!/usr/bin/env python
"""Lightweight claim-level support checker for MedSafe Evidence Cards.

This is a deterministic first-pass verifier for experiments. It does not prove
medical truth; it checks whether a generated claim has lexical support in one
or more Evidence Card fields and can therefore be cited or flagged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_card_store import EvidenceCardStore, lcs_len, normalize_text  # noqa: E402


STOP_TERMS = {
    "可以",
    "使用",
    "需要",
    "建议",
    "药物",
    "患者",
    "孕期",
    "哺乳期",
    "安全",
    "风险",
    "注意",
    "治疗",
}


def split_claims(text: str) -> list[str]:
    pieces = re.split(r"[。；;，,\n]+", text)
    return [piece.strip(" ，,") for piece in pieces if piece.strip(" ，,")]


def flatten_card_fields(card: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                rows.append({"path": path, "text": text})
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                walk(item, f"{path}[{idx}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else key)
            return
        text = str(value).strip()
        if text:
            rows.append({"path": path, "text": text})

    for top_key in [
        "identity",
        "clinical_profile",
        "safety",
        "pharmacokinetics",
        "source_trace",
    ]:
        walk(card.get(top_key), top_key)
    return rows


def claim_terms(claim: str) -> list[str]:
    raw = re.findall(r"[\u4e00-\u9fffA-Za-z0-9%～~.·+-]{2,}", claim)
    terms = []
    seen = set()
    for term in raw:
        term = term.strip()
        norm = normalize_text(term)
        if len(norm) < 2 or norm in STOP_TERMS or norm in seen:
            continue
        seen.add(norm)
        terms.append(term)
    return terms


def support_score(claim: str, field_text: str) -> tuple[float, list[str]]:
    terms = claim_terms(claim)
    field_norm = normalize_text(field_text)
    matched = [term for term in terms if normalize_text(term) in field_norm]

    claim_norm = normalize_text(claim)
    exact_bonus = 1.0 if claim_norm and claim_norm in field_norm else 0.0
    common_len = lcs_len(claim_norm, field_norm)
    lcs_score = common_len / max(1, min(len(claim_norm), len(field_norm)))
    if common_len >= 6 and lcs_score >= 0.45:
        matched.append(f"LCS>={common_len}")
    if not terms and common_len < 6:
        return 0.0, []

    term_score = len([m for m in matched if not m.startswith("LCS>=")]) / max(1, len(terms))
    score = max(term_score + exact_bonus, lcs_score)
    return score, matched


def verify_claim_against_cards(
    claim: str,
    cards: list[dict[str, Any]],
    min_score: float = 0.34,
    top_n: int = 3,
) -> dict[str, Any]:
    hits = []
    for card in cards:
        for field in flatten_card_fields(card):
            if field["path"].startswith("identity.") and normalize_text(claim) != normalize_text(field["text"]):
                continue
            field_norm = normalize_text(field["text"])
            claim_norm = normalize_text(claim)
            short_enum_paths = (
                ".category",
                ".normalized_action",
                ".normalized_risk",
                ".recommended_review_level",
                ".page_doc",
                ".page_pdf",
            )
            if (
                any(field["path"].endswith(suffix) for suffix in short_enum_paths)
                and field_norm
                and field_norm in claim_norm
            ):
                if len(field_norm) == 1 and field_norm.isalpha():
                    if f"分级为{field['text']}" not in claim and f"category={field['text']}" not in claim:
                        continue
                    if "pregnancy" in field["path"] and not any(term in claim for term in ["妊娠", "孕"]):
                        continue
                    if "lactation" in field["path"] and not any(term in claim for term in ["哺乳", "乳"]):
                        continue
                source = card.get("source_trace", {})
                hits.append(
                    {
                        "score": 1.0,
                        "matched_terms": [field["text"]],
                        "card_id": card.get("card_id"),
                        "drug": card.get("identity", {}).get("canonical_name"),
                        "field_path": field["path"],
                        "field_text": field["text"],
                        "citation": {
                            "page_doc": source.get("page_doc"),
                            "page_pdf": source.get("page_pdf"),
                        },
                    }
                )
                continue
            score, matched = support_score(claim, field["text"])
            if score >= min_score and matched:
                source = card.get("source_trace", {})
                hits.append(
                    {
                        "score": round(score, 4),
                        "matched_terms": matched,
                        "card_id": card.get("card_id"),
                        "drug": card.get("identity", {}).get("canonical_name"),
                        "field_path": field["path"],
                        "field_text": field["text"],
                        "citation": {
                            "page_doc": source.get("page_doc"),
                            "page_pdf": source.get("page_pdf"),
                        },
                    }
                )
    hits.sort(key=lambda item: item["score"], reverse=True)
    return {
        "claim": claim,
        "supported": bool(hits),
        "supporting_fields": hits[:top_n],
    }


def verify_claims(claims: list[str], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [verify_claim_against_cards(claim, cards) for claim in claims]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify claims against Evidence Cards.")
    parser.add_argument("--cards", type=Path, default=SCRIPT_DIR.parent / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--drug", action="append", default=[])
    parser.add_argument("--claim", action="append", default=[])
    parser.add_argument("--answer", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EvidenceCardStore.from_jsonl(args.cards)
    cards = []
    for drug in args.drug:
        card = store.lookup_name(drug)
        if card:
            cards.append(card)
    claims = list(args.claim)
    if args.answer:
        claims.extend(split_claims(args.answer))
    result = verify_claims(claims, cards)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
