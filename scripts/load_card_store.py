#!/usr/bin/env python
"""Load and query MedSafe Evidence Cards.

This module is intentionally dependency-free so it can be reused by the
retriever, rule engine, evaluation scripts, and simple demos.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_CARDS_JSONL = DEFAULT_SKILL_DIR / "resources" / "evidence_cards.jsonl"


QUERY_EXPANSIONS = {
    "退烧": ["发热", "解热", "镇痛"],
    "退热": ["发热", "解热", "镇痛"],
    "消炎": ["抗生素", "抗菌", "感染"],
    "喂奶": ["哺乳", "乳汁", "母乳"],
    "母乳": ["哺乳", "乳汁"],
    "怀孕": ["妊娠", "胎儿", "孕期"],
    "孕期": ["妊娠", "胎儿", "怀孕"],
    "止痛": ["疼痛", "镇痛"],
    "过敏": ["抗过敏", "荨麻疹", "鼻炎"],
    "高血压": ["降压", "慢性高血压"],
    "糖尿病": ["降糖", "血糖"],
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；、:：/\\()（）\[\]【】\-_\s]+", "", text)
    return text


def iter_text_values(value: Any) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, str):
        text = value.strip()
        if text:
            values.append(text)
        return values
    if isinstance(value, list):
        for item in value:
            values.extend(iter_text_values(item))
        return values
    if isinstance(value, dict):
        for item in value.values():
            values.extend(iter_text_values(item))
        return values
    text = str(value).strip()
    if text:
        values.append(text)
    return values


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


class EvidenceCardStore:
    """In-memory Evidence Card store with exact and fuzzy name lookup."""

    def __init__(self, cards: list[dict[str, Any]]):
        self.cards = cards
        self.by_card_id = {card["card_id"]: card for card in cards}
        self.by_name: dict[str, list[dict[str, Any]]] = {}
        for card in cards:
            for name in self.card_names(card):
                key = normalize_text(name)
                if key:
                    self.by_name.setdefault(key, []).append(card)

    @classmethod
    def from_jsonl(cls, path: Path = DEFAULT_CARDS_JSONL) -> "EvidenceCardStore":
        cards = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cards.append(json.loads(line))
        return cls(cards)

    @staticmethod
    def card_names(card: dict[str, Any]) -> list[str]:
        identity = card.get("identity", {})
        names = [
            identity.get("canonical_name", ""),
            identity.get("source_name", ""),
        ]
        names.extend(identity.get("aliases", []) or [])
        names.extend(identity.get("compound_components", []) or [])
        return [str(name) for name in names if str(name).strip()]

    def get(self, card_id: str) -> dict[str, Any] | None:
        return self.by_card_id.get(card_id)

    def lookup_name(self, name: str, fuzzy: bool = True) -> dict[str, Any] | None:
        key = normalize_text(name)
        if not key:
            return None
        exact = self.by_name.get(key)
        if exact:
            return exact[0]
        if not fuzzy:
            return None

        best_card = None
        best_score = 0.0
        for card in self.cards:
            for candidate in self.card_names(card):
                cand_key = normalize_text(candidate)
                if not cand_key:
                    continue
                if key in cand_key or cand_key in key:
                    containment = min(len(key), len(cand_key)) / max(len(key), len(cand_key))
                    score = 0.75 + 0.25 * containment
                else:
                    common = lcs_len(key, cand_key)
                    score = common / max(1, min(len(key), len(cand_key)))
                if score > best_score:
                    best_score = score
                    best_card = card
        return best_card if best_score >= 0.65 else None

    def search_text(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_terms = self.expand_query_terms(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        query_norm = normalize_text(query)

        for card in self.cards:
            score = 0.0
            identity = card.get("identity", {})
            profile = card.get("clinical_profile", {})
            safety = card.get("safety", {})

            names = self.card_names(card)
            if any(normalize_text(name) and normalize_text(name) in query_norm for name in names):
                score += 30.0

            field_text = " ".join(
                iter_text_values(
                    {
                        "drug_category": profile.get("drug_category"),
                        "indications": profile.get("indications"),
                        "mechanism": profile.get("mechanism"),
                        "safety": safety,
                        "components": identity.get("compound_components"),
                    }
                )
            )
            field_norm = normalize_text(field_text)
            for term in query_terms:
                term_norm = normalize_text(term)
                if term_norm and term_norm in field_norm:
                    score += 3.0 + min(len(term_norm), 8) / 4.0
            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [card for _, card in scored[:limit]]

    @staticmethod
    def expand_query_terms(query: str) -> list[str]:
        raw_terms = [query]
        raw_terms.extend(re.split(r"[，。；、,;?\s]+", query))
        for key, expansions in QUERY_EXPANSIONS.items():
            if key in query:
                raw_terms.extend(expansions)
        terms = []
        seen = set()
        for term in raw_terms:
            term = str(term).strip()
            key = normalize_text(term)
            if key and key not in seen:
                seen.add(key)
                terms.append(term)
        return terms

    @staticmethod
    def card_brief(card: dict[str, Any], include_safety: bool = True) -> dict[str, Any]:
        safety = card.get("safety", {})
        source = card.get("source_trace", {})
        brief = {
            "card_id": card.get("card_id"),
            "drug": card.get("identity", {}).get("canonical_name"),
            "drug_category": card.get("clinical_profile", {}).get("drug_category"),
            "indications": card.get("clinical_profile", {}).get("indications", []),
            "source_trace": {
                "page_doc": source.get("page_doc"),
                "page_pdf": source.get("page_pdf"),
                "references": source.get("references", [])[:3],
            },
        }
        if include_safety:
            brief["safety"] = {
                "pregnancy_category": safety.get("pregnancy", {}).get("category"),
                "pregnancy_risk": safety.get("pregnancy", {}).get("normalized_risk"),
                "breastfeeding_category": safety.get("lactation", {}).get("category"),
                "lactation_action": safety.get("lactation", {}).get("normalized_action"),
                "review_level": card.get("verification", {}).get("recommended_review_level"),
            }
        return brief


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query MedSafe Evidence Cards.")
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--name", help="Exact/fuzzy drug-name lookup.")
    parser.add_argument("--query", help="Simple lexical query over cards.")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = EvidenceCardStore.from_jsonl(args.cards)
    if args.name:
        card = store.lookup_name(args.name)
        result = store.card_brief(card) if card else None
    elif args.query:
        result = [store.card_brief(card) for card in store.search_text(args.query, args.limit)]
    else:
        result = {"card_count": len(store.cards)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
