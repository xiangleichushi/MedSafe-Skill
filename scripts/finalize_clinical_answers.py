#!/usr/bin/env python
"""Create one final answer set after optional interactive evidence collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
PROMPT_PATH = MEDSAFE_DIR / "prompts" / "final_answer_clinical_style_v4.txt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    answer_text,
    call_chat_api,
    combine_usage,
    model_matches_pool,
    parse_model_pool,
    prompt_hash,
    response_actual_model,
    response_model_route,
    usage,
)
from medsafe_skill.scripts.run_interactive_evidence_session import (  # noqa: E402
    validate_generated_answer,
    verified_base_claims,
)
from medsafe_skill.core.supplement_request import build_unified_supplement_request  # noqa: E402


MODEL_PRIOR_HEADER = "模型内部知识"
PRESENTATION_POLICY = "clinical_consultation_natural_sections_v4_binary_sources"
GENERATION_INPUT_VERSION = "4.0.0"
MAX_VALIDATION_REVISIONS = 2
SOFT_VALIDATION_CODES = {
    "analysis_leaked_into_conclusion",
    "rationale_leaked_into_conclusion",
    "noncanonical_numbered_list_style",
    "numbered_list_item_not_on_own_line",
}
REQUIRED_SECTION_HEADERS = ("结论：", "依据：", "建议：", "风险提示：")
INTERNAL_ANSWER_TERMS = (
    "Evidence Card",
    "DecisionPacket",
    "QASP",
    "model_prior",
    "pharmacist_review",
    "specialist_review",
    "standard_clinician_confirmation",
    "Generation mode",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_tokens(value: Any) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", str(value)))


def parse_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Final answer response must be a JSON object")
    return value


def render_answer_markdown(answer: str) -> str:
    """Render model list syntax predictably in Markdown without changing JSON."""

    text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Split an inline next item only after sentence punctuation; decimals such
    # as 3.1 and dose expressions therefore remain untouched.
    text = re.sub(
        r"(?<=[。；;：:！？!?])\s*(?=(?:[1-9]\d*)\.\s+|(?:[1-9]\d*)[、．]\s*)",
        "\n",
        text,
    )
    text = re.sub(
        r"(?<=[。；;：:！？!?])\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])",
        "\n",
        text,
    )
    # A blank line before every list item is accepted consistently by
    # Markdown renderers, including after 结论/依据/建议/风险提示 headings.
    text = re.sub(
        r"\n+(?=(?:[1-9]\d*)\.\s+|(?:[1-9]\d*)[、．]\s*|[①②③④⑤⑥⑦⑧⑨⑩])",
        "\n\n",
        text,
    )
    return text


def build_validation_revision_request(errors: list[str]) -> dict[str, Any]:
    """Translate validator codes into concrete, model-actionable edits."""

    fixes = []
    if "rationale_leaked_into_conclusion" in errors:
        fixes.append(
            "重写“结论”：只保留直接的用药行动、具体时间和必要适用条件。"
            "删除“按五个半衰期推演、半衰期、药代估算、消除比例、指南、证据、"
            "适应证、抗菌谱”等解释；把这些内容完整移到“依据”。例如结论可写"
            "“建议停药约75小时后恢复哺乳”，但不能在结论解释75小时如何算出。"
        )
    if "analysis_leaked_into_conclusion" in errors:
        fixes.append("删除“结论”中的分析过程，直接陈述每个小问的行动答案。")
    if "irrelevant_washout_reasoning" in errors:
        fixes.append(
            "原问题未询问停药或恢复哺乳时间，删除正文中所有五半衰期和清除时间推演。"
        )
    if "noncanonical_numbered_list_style" in errors:
        fixes.append(
            "需要分条的部分只使用逐行的“1. 2. 3.”，不得使用①、（1）或1、。"
        )
    if "numbered_list_item_not_on_own_line" in errors:
        fixes.append("将每个“1. 2. 3.”编号放在独立新行，不能在同一行连续编号。")
    if any(error.startswith("model_prior_") for error in errors):
        fixes.append(
            "逐项修复MP声明：text必须逐字出现在结论或依据，risk_notice_text必须"
            "逐字出现在风险提示，并使用相同MP编号的model_internal_knowledge引用。"
        )
    if "derived_estimate_not_classified_as_model_internal_knowledge" in errors:
        fixes.append(
            "正文中的五半衰期推演必须建立target_missing_fact="
            "five_half_life_inference的MP声明，并在风险提示披露为模型内部知识。"
        )
    if not fixes:
        fixes.append("严格按照系统提示和校验错误逐项重写相应字段。")
    return {
        "task": "修订整个JSON答案，逐项消除校验错误；只返回合法JSON。",
        "validation_errors": errors,
        "mandatory_fixes": fixes,
        "preserve_requirements": [
            "保持外部证据与模型内部知识边界。",
            "保留结论、依据、建议、风险提示四部分及其顺序。",
            "不得删除对用户任一小问的直接回答。",
        ],
    }


def split_validation_findings(findings: list[str]) -> tuple[list[str], list[str]]:
    """Separate safety/provenance failures from non-blocking style warnings."""

    errors = []
    warnings = []
    for finding in findings:
        if finding in SOFT_VALIDATION_CODES:
            warnings.append(finding)
        else:
            errors.append(finding)
    return sorted(set(errors)), sorted(set(warnings))


def prune_unrendered_model_prior_claims(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Discard model-prior claims that never reached the clinical answer."""

    normalized = dict(payload)
    answer = str(normalized.get("answer") or "")
    claims = normalized.get("model_prior_claims")
    if not isinstance(claims, list):
        return normalized, []
    kept = []
    discarded = []
    for claim in claims:
        text = str(claim.get("text") or "").strip() if isinstance(claim, dict) else ""
        rendered_text = rendered_model_prior_span(answer, text)
        if rendered_text:
            rebound = dict(claim)
            rebound["text"] = rendered_text
            kept.append(rebound)
        else:
            discarded.append({
                "claim": claim,
                "reason": "model_prior_text_not_rendered_in_answer",
            })
    kept_ids = {
        str(claim.get("claim_id") or "")
        for claim in kept if isinstance(claim, dict)
    }
    refs = normalized.get("support_refs")
    if isinstance(refs, list):
        normalized["support_refs"] = [
            ref for ref in refs
            if not (
                isinstance(ref, dict)
                and ref.get("support_class") == "model_internal_knowledge"
                and str(ref.get("support_id") or "") not in kept_ids
            )
        ]
    normalized["model_prior_claims"] = kept
    return normalized, discarded


def rendered_model_prior_span(answer: str, claim_text: str) -> str | None:
    """Bind a model-prior claim to its exact visible answer span."""

    if not claim_text:
        return None
    exact_start = answer.find(claim_text)
    core = claim_text.rstrip("。！？.!? ")
    start = exact_start if exact_start >= 0 else answer.find(core)
    if start < 0:
        return None
    minimum_end = start + (len(claim_text) if exact_start >= 0 else len(core))
    return answer[start:minimum_end]


def remaining_critical_fields(session: dict[str, Any]) -> list[str]:
    values = session.get("values") or {}
    return [
        item["field"]
        for item in (session.get("request") or {}).get("required_fields") or []
        if str(item.get("priority", "")).startswith("critical")
        and item.get("field") not in values
    ]


def remaining_requested_fields(session: dict[str, Any]) -> list[str]:
    values = session.get("values") or {}
    return [
        item["field"]
        for item in (session.get("request") or {}).get("required_fields") or []
        if item.get("field") not in values
    ]


def model_prior_allowed_fields(session: dict[str, Any]) -> list[str]:
    remaining = set(remaining_requested_fields(session))
    return [
        item["field"]
        for item in (session.get("request") or {}).get("required_fields") or []
        if item.get("field") in remaining
        and item.get("field_kind") == "external_evidence"
    ]


def enrich_derived_estimates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade persisted v1 half-life estimates without changing user input."""

    enriched = []
    for raw in items:
        item = dict(raw)
        if item.get("method") == "five_half_life_mechanistic_estimate":
            count = int(item.get("half_life_count") or 5)
            remaining = 0.5 ** count
            item.setdefault("half_life_count", count)
            item.setdefault("estimated_remaining_fraction", remaining)
            item.setdefault("estimated_remaining_percent", round(remaining * 100, 3))
            item.setdefault("estimated_eliminated_percent", round((1 - remaining) * 100, 3))
            item.setdefault(
                "pharmacokinetic_assumption",
                "first_order_elimination_mechanistic_approximation",
            )
        enriched.append(item)
    return enriched


def query_requests_washout_or_resumption(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in lowered
        for term in (
            "停药多久",
            "停药多少",
            "停药后",
            "停用",
            "停服",
            "停了",
            "多久能喂",
            "多少天能喂",
            "恢复哺乳",
            "恢复母乳",
            "可否哺乳",
            "能否哺乳",
            "可以哺乳吗",
            "resume breastfeeding",
            "washout",
        )
    )


def compact_reasoning_trace(case: dict[str, Any]) -> dict[str, Any]:
    trace = case.get("reasoning_trace") or {}
    nodes = []
    for item in trace.get("decision_tree_nodes") or []:
        nodes.append({
            key: item.get(key)
            for key in (
                "card_id", "drug", "pathway", "node", "question",
                "outcome", "basis", "safety_note",
            )
            if item.get(key) is not None
        })
    return {
        "trace_version": trace.get("trace_version"),
        "trace_type": trace.get("trace_type"),
        "observation_trace": trace.get("observation_trace"),
        "question_clarification": trace.get("question_clarification"),
        "mother_infant_drug_assessment": trace.get("mother_infant_drug_assessment"),
        "evidence_appraisal": trace.get("evidence_appraisal"),
        "state_transitions": trace.get("state_transitions") or [],
        "decision_tree_nodes": nodes,
        "claim_evidence_bindings": trace.get("claim_evidence_bindings") or [],
        "review_triggers": trace.get("review_triggers") or [],
        "final_state": trace.get("final_state"),
    }


def generation_reasoning_summary(case: dict[str, Any]) -> dict[str, Any]:
    """Project only decision-relevant trace outcomes into the LLM context."""

    trace = case.get("reasoning_trace") or {}
    clarification = trace.get("question_clarification") or {}
    appraisal = trace.get("evidence_appraisal") or {}
    terminal_nodes = []
    for item in trace.get("decision_tree_nodes") or []:
        node = str(item.get("node") or "")
        if not (
            node.endswith("-06-shared-decision")
            or node.endswith("-06-final-action")
            or node in {"LAC-06-shared-decision", "PREG-06-final-action"}
        ):
            continue
        terminal_nodes.append({
            key: item.get(key)
            for key in ("card_id", "drug", "pathway", "node", "outcome")
            if item.get(key) is not None
        })
    return {
        "consultation_intent": clarification.get("consultation_intent"),
        "individualized": clarification.get("individualized"),
        "requested_facts": appraisal.get("requested_facts") or [],
        "unsupported_facts": appraisal.get("unsupported_facts") or [],
        "evidence_consistency": appraisal.get("consistency"),
        "pathway_terminal_outcomes": terminal_nodes,
        "state_sequence": [
            {
                "stage": item.get("qasp_stage"),
                "to": item.get("to"),
            }
            for item in trace.get("state_transitions") or []
        ],
        "final_state": trace.get("final_state"),
    }


def validate_model_prior_answer(
    *,
    payload: dict[str, Any],
    case: dict[str, Any],
    session: dict[str, Any],
    generation_input: dict[str, Any],
) -> list[str]:
    answer = str(payload.get("answer") or "").strip()
    claims = payload.get("model_prior_claims")
    refs = payload.get("support_refs")
    errors = []
    if not isinstance(claims, list):
        return ["model_prior_claims_must_be_a_list"]
    if not isinstance(refs, list):
        return ["support_refs_must_be_a_list"]

    allowed_targets = set(generation_input["model_prior_allowed_fields"])
    seen_ids = set()
    prior_texts = []
    prior_risk_notices = []
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("invalid_model_prior_claim")
            continue
        claim_id = str(claim.get("claim_id") or "")
        text = str(claim.get("text") or "").strip()
        target = str(claim.get("target_missing_fact") or "")
        if not re.fullmatch(r"MP\d{3}", claim_id):
            errors.append(f"invalid_model_prior_claim_id:{claim_id}")
        if claim_id in seen_ids:
            errors.append(f"duplicate_model_prior_claim_id:{claim_id}")
        seen_ids.add(claim_id)
        if target not in allowed_targets:
            errors.append(f"model_prior_target_not_missing:{target}")
        if not text or text not in answer:
            errors.append(f"model_prior_text_not_verbatim:{claim_id}")
        if claim.get("confidence") != "low":
            errors.append(f"model_prior_confidence_not_low:{claim_id}")
        risk_notice = str(claim.get("risk_notice_text") or "").strip()
        if not risk_notice or risk_notice not in answer:
            errors.append(f"model_prior_risk_notice_not_verbatim:{claim_id}")
        if "模型内部知识" not in risk_notice:
            errors.append(f"model_prior_source_not_disclosed:{claim_id}")
        if any(term in text for term in ("一定安全", "确定安全", "无需审核", "可以自行", "建议自行")):
            errors.append(f"unsafe_model_prior_wording:{claim_id}")
        prior_texts.append(text)
        prior_risk_notices.append(risk_notice)

    external_claim_ids = {
        str(item.get("claim_id") or "")
        for item in generation_input.get("verified_base_claims") or []
        if str(item.get("card_id") or "").strip()
    }
    external_field_ids = set(generation_input.get("supplemental_values") or {})
    prior_ref_ids = set()
    external_ref_ids = set()
    for ref in refs:
        if not isinstance(ref, dict):
            errors.append("invalid_support_ref")
            continue
        support_class = str(ref.get("support_class") or "")
        support_id = str(ref.get("support_id") or "")
        if support_class == "model_internal_knowledge":
            prior_ref_ids.add(support_id)
        elif support_class == "external_evidence":
            external_ref_ids.add(support_id)
            if support_id not in external_claim_ids | external_field_ids:
                errors.append(f"unknown_external_evidence_ref:{support_id}")
        else:
            errors.append(f"invalid_support_class:{support_class}")
    if seen_ids != prior_ref_ids:
        errors.append("model_prior_support_refs_do_not_match_claims")
    if claims:
        if not generation_input.get("allow_model_prior"):
            errors.append("model_prior_not_allowed")
    for term in INTERNAL_ANSWER_TERMS:
        if term in answer:
            errors.append(f"internal_provenance_term_exposed:{term}")
    positions = [answer.find(header) for header in REQUIRED_SECTION_HEADERS]
    if any(position < 0 for position in positions):
        errors.append("required_natural_section_missing")
    elif positions != sorted(positions):
        errors.append("natural_sections_out_of_order")
    conclusion_start = answer.find("结论：")
    evidence_start = answer.find("依据：")
    conclusion = answer[conclusion_start + len("结论："):evidence_start].strip()
    if not conclusion:
        errors.append("direct_conclusion_missing")
    if any(term in conclusion for term in ("分析如下", "需要先分析", "首先分析")):
        errors.append("analysis_leaked_into_conclusion")
    if any(
        term in conclusion
        for term in (
            "因为", "由于", "依据", "基于", "根据", "指南", "证据", "资料",
            "研究", "文献", "半衰期", "药代", "估算", "消除", "排泄",
            "乳汁浓度", "蛋白结合", "分子量", "适应证", "广谱", "覆盖",
        )
    ):
        errors.append("rationale_leaked_into_conclusion")
    if (
        not query_requests_washout_or_resumption(str(case.get("query") or ""))
        and any(
            term in answer
            for term in ("5个半衰期", "五个半衰期", "停药后约", "体内药物可消除")
        )
    ):
        errors.append("irrelevant_washout_reasoning")
    if (
        re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", answer)
        or re.search(r"(?:^|\n)\s*(?:[1-9]\d*)[、．]\s*", answer)
        or re.search(r"(?:^|\n)\s*[（(](?:[1-9]\d*)[）)]\s*", answer)
    ):
        errors.append("noncanonical_numbered_list_style")
    for marker in re.finditer(r"(?<![\d.])(?:[1-9]\d*)\.\s+", answer):
        line_start = answer.rfind("\n", 0, marker.start()) + 1
        if answer[line_start:marker.start()].strip():
            errors.append("numbered_list_item_not_on_own_line")
            break

    estimate_numbers = set()
    for estimate in generation_input.get("derived_estimates") or []:
        for key in (
            "estimated_five_half_lives_hours",
            "estimated_five_half_lives_days",
            "estimated_five_half_lives_days_rounded_1dp",
            "estimated_remaining_percent",
            "estimated_eliminated_percent",
        ):
            estimate_numbers.update(numeric_tokens(estimate.get(key)))
    estimate_claim_ids = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "")
        text = str(claim.get("text") or "")
        target = str(claim.get("target_missing_fact") or "")
        if target == "five_half_life_inference" and (
            estimate_numbers.intersection(numeric_tokens(text))
            or any(term in text for term in ("5个半衰期", "五个半衰期"))
        ):
            estimate_claim_ids.add(claim_id)
    estimate_is_visible = (
        bool(estimate_numbers.intersection(numeric_tokens(answer)))
        or any(
            term in answer
            for term in ("5个半衰期", "五个半衰期", "96.875%", "96.9%")
        )
    )
    if estimate_is_visible and not estimate_claim_ids:
        errors.append("derived_estimate_not_classified_as_model_internal_knowledge")

    evidence_answer = answer
    for text in prior_texts + prior_risk_notices:
        evidence_answer = evidence_answer.replace(text, "")
    evidence_refs = []
    for support_id in external_ref_ids:
        if support_id in external_claim_ids:
            support_type = "verified_base_claim"
        elif support_id in external_field_ids:
            support_type = "supplemental_field"
        else:
            continue
        evidence_refs.append({
            "support_type": support_type,
            "support_id": support_id,
        })
    if evidence_refs:
        errors.extend(validate_generated_answer(
            payload={"answer": evidence_answer, "support_refs": evidence_refs},
            query=str(case.get("query") or ""),
            base_claims=generation_input["verified_base_claims"],
            values=generation_input["supplemental_values"],
            estimates=[],
            decision_packet=generation_input["decision_packet"],
        ))
    review_level = str(generation_input["decision_packet"].get("review_level") or "")
    elevated_review_terms = {
        "pharmacist_review": ("药师", "临床药师"),
        "specialist_review": ("专科", "临床药师"),
    }
    expected_review_terms = elevated_review_terms.get(review_level)
    if expected_review_terms and not any(term in answer for term in expected_review_terms):
        errors.append("natural_language_review_boundary_missing")
    workflow_action = str(
        generation_input["decision_packet"].get("workflow_action") or ""
    )
    if workflow_action in {"pharmacist_review", "specialist_review"}:
        if not any(term in answer for term in ("审核", "复核", "确认", "咨询")):
            errors.append("human_review_boundary_missing")
    return sorted(set(errors))


def build_generation_input(
    case: dict[str, Any],
    session: dict[str, Any],
    allow_model_prior: bool,
) -> dict[str, Any]:
    packet = dict(case.get("decision_packet") or {})
    packet_projection = {
        key: packet.get(key)
        for key in (
            "action", "clinical_action", "evidence_bound_action",
            "evidence_status", "workflow_action", "workflow_actions",
            "decision_scope", "review_level", "review_required",
            "release_status", "missing_information", "contextual_information",
            "unsupported_facts",
        )
    }
    missing = remaining_critical_fields(session)
    remaining = remaining_requested_fields(session)
    prior_fields = model_prior_allowed_fields(session)
    if allow_model_prior and "general_clinical_interpretation" not in prior_fields:
        prior_fields.append("general_clinical_interpretation")
    if allow_model_prior and "monitoring_recommendation" not in prior_fields:
        prior_fields.append("monitoring_recommendation")
    for fact in packet.get("unsupported_facts") or []:
        field = f"unsupported_fact:{fact}"
        if field not in prior_fields:
            prior_fields.append(field)
    if not verified_base_claims(case) or packet.get("evidence_status") in {
        "partial", "insufficient", "out_of_scope"
    }:
        if "provisional_clinical_answer" not in prior_fields:
            prior_fields.append("provisional_clinical_answer")
    packet_projection["remaining_critical_fields"] = missing
    estimates = enrich_derived_estimates(
        list(session.get("derived_estimates") or [])
    )
    if estimates and "five_half_life_inference" not in prior_fields:
        prior_fields.append("five_half_life_inference")
    return {
        "generation_input_version": GENERATION_INPUT_VERSION,
        "original_query": case.get("query"),
        "verified_base_claims": verified_base_claims(case),
        "supplemental_values": dict(session.get("values") or {}),
        "derived_estimates": estimates,
        "remaining_critical_fields": missing,
        "remaining_requested_fields": remaining,
        "model_prior_allowed_fields": prior_fields,
        "decision_packet": packet_projection,
        "reasoning_summary": generation_reasoning_summary(case),
        "allow_model_prior": bool(allow_model_prior and prior_fields),
        "presentation_policy": PRESENTATION_POLICY,
    }


def build_safety_audit(
    payload: dict[str, Any],
    generation_input: dict[str, Any],
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an audit-only provenance ledger separated from answer prose."""

    refs = [
        item for item in (payload.get("support_refs") or [])
        if isinstance(item, dict)
    ]
    ids_by_class: dict[str, list[str]] = {}
    for ref in refs:
        support_class = str(ref.get("support_class") or "")
        support_id = str(ref.get("support_id") or "")
        if support_class and support_id:
            ids_by_class.setdefault(support_class, [])
            if support_id not in ids_by_class[support_class]:
                ids_by_class[support_class].append(support_id)

    base_lookup = {
        str(item.get("claim_id")): item
        for item in generation_input.get("verified_base_claims") or []
    }
    evidence_claims = [
        base_lookup[claim_id]
        for claim_id in ids_by_class.get("external_evidence", [])
        if claim_id in base_lookup
    ]
    card_claims = [
        claim for claim in evidence_claims
        if str(claim.get("card_id") or "").strip()
    ]
    card_claim_ids = [
        str(claim.get("claim_id") or "")
        for claim in card_claims
        if claim.get("claim_id")
    ]
    non_card_claim_ids = [
        str(claim.get("claim_id") or "")
        for claim in evidence_claims
        if claim.get("claim_id") and not str(claim.get("card_id") or "").strip()
    ]
    card_sources = []
    seen_cards = set()
    for claim in card_claims:
        card_id = str(claim.get("card_id") or "")
        marker = (
            card_id,
            str(claim.get("page_doc") or ""),
            str(claim.get("page_pdf") or ""),
        )
        if card_id and marker not in seen_cards:
            seen_cards.add(marker)
            card_sources.append({
                "card_id": card_id,
                "page_doc": claim.get("page_doc"),
                "page_pdf": claim.get("page_pdf"),
            })

    supplemental = generation_input.get("supplemental_values") or {}
    supplemental_used = {
        field: supplemental.get(field)
        for field in ids_by_class.get("external_evidence", [])
        if field in supplemental
    }
    estimates = {
        str(item.get("method")): item
        for item in generation_input.get("derived_estimates") or []
        if item.get("method")
    }
    model_prior_claims = [
        item for item in (payload.get("model_prior_claims") or [])
        if isinstance(item, dict)
    ]
    model_prior_text = " ".join(
        str(item.get("text") or "") for item in model_prior_claims
    )
    estimates_used = {
        method: value
        for method, value in estimates.items()
        if numeric_tokens(value).intersection(numeric_tokens(model_prior_text))
    }
    supplement_session = session or {}
    return {
        "audit_version": "3.0.0",
        "presentation_policy": PRESENTATION_POLICY,
        "knowledge_source_classes": {
            "external_evidence": {
                "used": bool(card_claims or supplemental_used),
                "evidence_card_claim_ids": card_claim_ids,
                "interactive_supplement_fields": sorted(supplemental_used),
            },
            "model_internal_knowledge": {
                "used": bool(model_prior_claims),
                "claim_ids": [
                    str(item.get("claim_id") or "")
                    for item in model_prior_claims
                ],
            },
        },
        "answer_exposes_internal_provenance_labels": any(
            term in str(payload.get("answer") or "")
            for term in INTERNAL_ANSWER_TERMS
        ),
        "structured_base_claim_ids": [
            claim_id
            for claim_id in ids_by_class.get("external_evidence", [])
            if claim_id in base_lookup
        ],
        "evidence_card_used": bool(card_claims),
        "evidence_card_claim_ids": card_claim_ids,
        "non_card_base_claim_ids": non_card_claim_ids,
        "evidence_card_sources": card_sources,
        "supplemental_values_used": supplemental_used,
        "interactive_external_evidence_log": {
            "session_version": supplement_session.get("session_version"),
            "session_status": supplement_session.get("session_status"),
            "collection_closed_by_user": supplement_session.get(
                "collection_closed_by_user"
            ),
            "turns": supplement_session.get("turns") or [],
            "accepted_values": supplement_session.get("values") or {},
        },
        "derived_estimates_used": estimates_used,
        "model_prior_used": bool(model_prior_claims),
        "model_prior_fields": [
            str(item.get("target_missing_fact") or "")
            for item in model_prior_claims
        ],
        "model_prior_claims": model_prior_claims,
        "remaining_critical_fields": generation_input.get("remaining_critical_fields") or [],
        "remaining_requested_fields": generation_input.get("remaining_requested_fields") or [],
    }


def generate_final_answer(
    *,
    case: dict[str, Any],
    session: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    generation_input = build_generation_input(case, session, not args.disable_model_prior)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(generation_input, ensure_ascii=False)},
    ]
    response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        temperature=0.0,
        max_retries=args.api_max_retries,
    )
    raw = answer_text(response)
    payload, discarded_model_prior_claims = prune_unrendered_model_prior_claims(
        parse_payload(raw)
    )
    findings = validate_model_prior_answer(
        payload=payload,
        case=case,
        session=session,
        generation_input=generation_input,
    )
    errors, warnings = split_validation_findings(findings)
    total_usage = usage(response)
    responses = [response]
    response_ids = [response.get("id")]
    revision_count = 0
    latest_raw = raw
    while errors and revision_count < MAX_VALIDATION_REVISIONS:
        revision_count += 1
        messages.extend([
            {"role": "assistant", "content": latest_raw},
            {
                "role": "user",
                "content": json.dumps(
                    build_validation_revision_request(errors),
                    ensure_ascii=False,
                ),
            },
        ])
        revision = call_chat_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=0.0,
            max_retries=args.api_max_retries,
        )
        responses.append(revision)
        latest_raw = answer_text(revision)
        payload, revision_discarded = prune_unrendered_model_prior_claims(
            parse_payload(latest_raw)
        )
        discarded_model_prior_claims.extend(revision_discarded)
        findings = validate_model_prior_answer(
            payload=payload,
            case=case,
            session=session,
            generation_input=generation_input,
        )
        errors, warnings = split_validation_findings(findings)
        total_usage = combine_usage(total_usage, usage(revision))
        response_ids.append(revision.get("id"))
    final_response = responses[-1]
    return {
        "answer": str(payload.get("answer") or "").strip(),
        "generation_input_version": generation_input["generation_input_version"],
        "model": response_actual_model(final_response, args.model),
        "configured_model_pool": parse_model_pool(args.model, args.models),
        "model_routes": [
            route for route in map(response_model_route, responses) if route
        ],
        "answer_status": payload.get("answer_status") or "final_draft_requires_clinical_review",
        "support_refs": payload.get("support_refs") or [],
        "model_prior_claims": payload.get("model_prior_claims") or [],
        "discarded_model_prior_claims": discarded_model_prior_claims,
        "model_prior_used": bool(payload.get("model_prior_claims")),
        "presentation_policy": PRESENTATION_POLICY,
        "safety_audit": build_safety_audit(payload, generation_input, session),
        "decision_packet_snapshot": generation_input["decision_packet"],
        "reasoning_trace_snapshot": compact_reasoning_trace(case),
        "remaining_critical_fields": generation_input["remaining_critical_fields"],
        "remaining_requested_fields": generation_input["remaining_requested_fields"],
        "model_prior_allowed_fields": generation_input["model_prior_allowed_fields"],
        "validation": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
        "prompt_file": PROMPT_PATH.name,
        "prompt_sha256": prompt_hash(prompt),
        "response_ids": response_ids,
        "revision_count": revision_count,
        "usage": total_usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument(
        "--models",
        default=os.getenv("CHAT_MODELS") or "",
        help="Comma-separated fallback models; --model/CHAT_MODEL is primary.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("CHAT_API_URL")
        or os.getenv("DASHSCOPE_CHAT_API_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disable-model-prior", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-each", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def compatible(existing: dict[str, Any], base_sha: str, session_sha: str, args: argparse.Namespace) -> bool:
    return (
        not existing.get("error")
        and existing.get("generation_input_version") == GENERATION_INPUT_VERSION
        and existing.get("base_record_sha256") == base_sha
        and existing.get("session_sha256") == session_sha
        and existing.get("prompt_sha256") == prompt_hash(PROMPT_PATH.read_text(encoding="utf-8").strip())
        and model_matches_pool(
            existing.get("model"), parse_model_pool(args.model, args.models)
        )
        and existing.get("model_prior_enabled") == (not args.disable_model_prior)
    )


def write_artifacts(records: list[dict[str, Any]], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reviewable = [
        item for item in records
        if not item.get("error")
        and (item.get("validation") or {}).get("passed") is True
    ]
    lines = ["# MedSafe final clinical answers", ""]
    for item in reviewable:
        lines.extend([
            f"## Case {item.get('id')}",
            "",
            f"**Query:** {item.get('query') or ''}",
            "",
            render_answer_markdown(
                item.get("answer") or f"ERROR: {item.get('error')}"
            ),
            "",
        ])
    if not reviewable:
        lines.append("No validated answers are available for review.")
    (artifact_dir / "final_answers_readable.md").write_text("\n".join(lines), encoding="utf-8")
    debug_lines = ["# MedSafe final answer debug output", ""]
    for item in records:
        debug_lines.extend([
            f"## Case {item.get('id')}",
            "",
            f"Validation: `{json.dumps(item.get('validation'), ensure_ascii=False)}`",
            "",
            render_answer_markdown(
                item.get("answer") or f"ERROR: {item.get('error')}"
            ),
            "",
        ])
    (artifact_dir / "final_answers_debug.md").write_text(
        "\n".join(debug_lines), encoding="utf-8"
    )
    with (artifact_dir / "clinical_answers_for_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["序号", "输入问题", "建议回答"])
        writer.writeheader()
        for item in reviewable:
            writer.writerow({
                "序号": item.get("id"),
                "输入问题": item.get("query") or "",
                "建议回答": item.get("answer") or "",
            })
    with (artifact_dir / "final_answer_audit.jsonl").open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps({
                "id": item.get("id"),
                "model": item.get("model"),
                "generation_mode": item.get("generation_mode"),
                "validation": item.get("validation"),
                "safety_audit": item.get("safety_audit"),
                "discarded_model_prior_claims": (
                    item.get("discarded_model_prior_claims") or []
                ),
                "decision_packet_snapshot": item.get("decision_packet_snapshot"),
                "reasoning_trace_snapshot": item.get("reasoning_trace_snapshot"),
                "support_refs": item.get("support_refs") or [],
                "response_ids": item.get("response_ids") or [],
                "error": item.get("error"),
            }, ensure_ascii=False) + "\n")
    save_json(artifact_dir / "final_run_summary.json", {
        "record_count": len(records),
        "error_count": sum(bool(item.get("error")) for item in records),
        "reviewable_record_count": len(reviewable),
        "excluded_from_review_ids": [
            item.get("id") for item in records if item not in reviewable
        ],
        "model_prior_case_count": sum(bool(item.get("model_prior_used")) for item in records),
        "reasoning_trace_record_count": sum(
            bool(item.get("reasoning_trace_snapshot")) for item in records
        ),
        "validation_failure_count": sum(
            not (item.get("validation") or {}).get("passed", False)
            for item in records
            if not item.get("error")
        ),
        "validation_warning_case_count": sum(
            bool((item.get("validation") or {}).get("warnings"))
            for item in records
            if not item.get("error")
        ),
        "validation_warning_distribution": {
            warning: sum(
                warning in ((item.get("validation") or {}).get("warnings") or [])
                for item in records
            )
            for warning in sorted({
                warning
                for item in records
                for warning in (
                    (item.get("validation") or {}).get("warnings") or []
                )
            })
        },
        "generation_modes": {
            mode: sum(item.get("generation_mode") == mode for item in records)
            for mode in sorted({str(item.get("generation_mode")) for item in records})
        },
    })


def main() -> None:
    args = parse_args()
    args.models = ",".join(parse_model_pool(args.model, args.models))
    os.environ["CHAT_MODELS"] = args.models
    print(f"[ChatAPI] Configured model pool: {args.models}", flush=True)
    if not args.answers.exists():
        raise FileNotFoundError(args.answers)
    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")
    output = args.output or args.answers.with_name("clinical_case_final_answers.json")
    artifact_dir = args.artifact_dir or args.answers.parent / "final_artifacts"
    base_records = load_json(args.answers)
    if not isinstance(base_records, list):
        raise ValueError("--answers must contain a JSON list")
    failed_ids = [item.get("id") for item in base_records if item.get("error")]
    invalid_ids = [
        item.get("id")
        for item in base_records
        if not item.get("error")
        and (
            not item.get("answer")
            or not item.get("decision_packet")
            or not item.get("reasoning_trace")
            or not (item.get("claim_verification") or {}).get("claims")
        )
    ]
    if failed_ids or invalid_ids:
        raise RuntimeError(
            "Stage 1 contains no usable complete answer set. "
            f"error_records={len(failed_ids)} invalid_records={len(invalid_ids)} "
            f"examples={(failed_ids + invalid_ids)[:10]}. Rerun Stage 1 after fixing the API."
        )
    if args.limit is not None:
        base_records = base_records[: args.limit]
    existing_records = load_json(output) if args.resume and output.exists() else []
    existing_by_id = {str(item.get("id")): item for item in existing_records}
    results = []

    for index, case in enumerate(base_records, start=1):
        case_id = case.get("id")
        print(f"[Finalizer] Case {index}/{len(base_records)} id={case_id}")
        request = build_unified_supplement_request(case)
        session_path = args.answers.with_name(f"interactive_evidence_case_{case_id}.json")
        session = load_json(session_path) if session_path.exists() else {
            "case_id": case_id,
            "request": request or {"required_fields": []},
            "values": {},
            "derived_estimates": [],
            "session_status": "not_required" if not request else "not_collected",
        }
        if request:
            session["request"] = request
        base_sha = sha256_value(case)
        session_sha = sha256_value(session)
        old = existing_by_id.get(str(case_id))
        if args.resume and old and compatible(old, base_sha, session_sha, args):
            refreshed = dict(old)
            generation_input = build_generation_input(
                case,
                session,
                not args.disable_model_prior,
            )
            findings = validate_model_prior_answer(
                payload=refreshed,
                case=case,
                session=session,
                generation_input=generation_input,
            )
            hard_errors, soft_warnings = split_validation_findings(findings)
            if hard_errors:
                print(
                    f"[Finalizer] Existing id={case_id} has hard validation "
                    f"errors; regenerate: {','.join(hard_errors)}"
                )
            else:
                print(
                    f"[Finalizer] Revalidated existing id={case_id}; "
                    f"warnings={','.join(soft_warnings) or 'none'}"
                )
                refreshed["validation"] = {
                    "passed": True,
                    "errors": [],
                    "warnings": soft_warnings,
                }
                refreshed["safety_audit"] = build_safety_audit(
                    refreshed,
                    generation_input,
                    session,
                )
                results.append(refreshed)
                continue

        common = {
            "id": case_id,
            "group": "medsafe_final",
            "query": case.get("query"),
            "model": args.model,
            "configured_model_pool": parse_model_pool(args.model, args.models),
            "base_record_sha256": base_sha,
            "session_sha256": session_sha,
            "session_path": str(session_path) if session_path.exists() else None,
            "model_prior_enabled": not args.disable_model_prior,
        }
        try:
            generated = generate_final_answer(case=case, session=session, args=args)
            record = {
                **common,
                "generation_mode": (
                    "reasoning_trace_with_supplement_and_disclosed_model_prior"
                    if generated.get("model_prior_used")
                    else (
                        "reasoning_trace_with_user_supplement"
                        if session.get("values")
                        else "reasoning_trace_evidence_bound"
                    )
                ),
                **generated,
            }
            results.append(record)
        except Exception as exc:
            error = {
                **common,
                "generation_mode": "generation_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            results.append(error)
            print(f"[Finalizer][ERROR] id={case_id}: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                save_json(output, results)
                write_artifacts(results, artifact_dir)
                raise
        if args.save_each:
            save_json(output, results)
            write_artifacts(results, artifact_dir)

    save_json(output, results)
    write_artifacts(results, artifact_dir)
    print(json.dumps({
        "record_count": len(results),
        "output": str(output),
        "artifact_dir": str(artifact_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
