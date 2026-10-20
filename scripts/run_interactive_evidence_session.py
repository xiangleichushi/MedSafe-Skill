#!/usr/bin/env python
"""Collect missing evidence in a terminal and map free text to variables with an LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
PROMPT_PATH = MEDSAFE_DIR / "prompts" / "supplemental_evidence_extractor_v2.txt"
ANSWER_PROMPT_PATH = MEDSAFE_DIR / "prompts" / "supplemental_answer_generator_v1.txt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    answer_text,
    call_chat_api,
    combine_usage,
    parse_model_pool,
    prompt_hash,
    usage,
)
from medsafe_skill.core.supplement_request import build_unified_supplement_request  # noqa: E402


FIELD_LABELS = {
    "authoritative_breastfeeding_resumption_recommendation": "停乳或恢复哺乳建议",
    "active_ingredients_and_formulation": "复方药物的活性成分和剂型",
    "lactation_compatibility_assessment": "哺乳期相容性结论及适用条件",
    "terminal_elimination_half_life": "终末消除半衰期",
    "active_metabolites_and_half_lives": "活性代谢物及其半衰期（没有也请明确说明）",
    "molecular_weight": "分子量",
    "protein_binding_fraction": "蛋白结合率",
    "fraction_metabolized_and_elimination_route": "代谢比例、代谢途径和主要排泄途径",
    "milk_transfer_or_relative_infant_dose": "乳汁浓度、乳血比或相对婴儿剂量",
    "infant_oral_bioavailability": "婴儿口服生物利用度",
    "monitoring_recommendation": "母亲和婴儿监测建议",
    "time_since_last_dose": "实际或计划末次给药时间",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument(
        "--models",
        default=os.getenv("CHAT_MODELS") or "",
        help="Comma-separated fallback models; --model/CHAT_MODEL is primary.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--answer-max-tokens", type=int, default=2200)
    parser.add_argument("--no-generate-answer", action="store_true")
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def find_case(records: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for item in records:
        if str(item.get("id")) == str(case_id) and item.get("group") == "medsafe_full":
            return item
    raise ValueError(f"Cannot find medsafe_full case id={case_id}")


def parse_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        raise ValueError("Supplemental evidence response must contain an entries list")
    return value


def validate_entries(
    *,
    terminal_input: str,
    payload: dict[str, Any],
    allowed_fields: set[str],
    field_kinds: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    accepted = []
    rejected = []
    for raw in payload.get("entries", []):
        field = str(raw.get("field") or "").strip()
        span = str(raw.get("source_span") or "").strip()
        errors = []
        if field not in allowed_fields:
            errors.append("field_not_requested")
        if not span or span not in terminal_input:
            errors.append("source_span_not_literal")
        if raw.get("value") in (None, ""):
            errors.append("empty_value")
        if errors:
            rejected.append({"field": field, "errors": ",".join(errors)})
            continue
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        field_kind = (field_kinds or {}).get(field, "external_evidence")
        accepted.append({
            "field": field,
            "value": raw.get("value"),
            "unit": raw.get("unit"),
            "population": raw.get("population"),
            "formulation_and_route": raw.get("formulation_and_route"),
            "source_span": span,
            "source": source,
            "field_kind": field_kind,
        })
    return accepted, rejected


def half_life_estimate(values: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    item = values.get("terminal_elimination_half_life")
    if not item:
        return None
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(item.get("value") or ""))]
    if not numbers:
        return None
    number = max(numbers)
    unit = str(item.get("unit") or "").lower()
    if unit in {"天", "day", "days", "d"}:
        hours = number * 24
    elif unit in {"小时", "时", "hour", "hours", "h", "hr"}:
        hours = number
    else:
        return None
    half_life_count = 5
    remaining_fraction = 0.5 ** half_life_count
    return {
        "method": "five_half_life_mechanistic_estimate",
        "half_life_count": half_life_count,
        "numeric_selection_rule": "upper_bound_when_a_range_is_reported",
        "input_half_life_hours": hours,
        "estimated_five_half_lives_hours": round(hours * half_life_count, 3),
        "estimated_five_half_lives_days": round(hours * half_life_count / 24, 3),
        "estimated_five_half_lives_days_rounded_1dp": round(hours * half_life_count / 24, 1),
        "estimated_remaining_fraction": remaining_fraction,
        "estimated_remaining_percent": round(remaining_fraction * 100, 3),
        "estimated_eliminated_percent": round((1 - remaining_fraction) * 100, 3),
        "pharmacokinetic_assumption": "first_order_elimination_mechanistic_approximation",
        "status": "mechanistic_estimate_not_clinical_recommendation",
        "warning": (
            "This estimate is not an automatically validated breastfeeding resumption recommendation "
            "and does not automatically account for active metabolites, repeated dosing, nonlinear "
            "kinetics, organ function, milk transfer, infant absorption, or patient-specific clearance."
        ),
    }


def verified_base_claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts = {
        str(item.get("claim_id")): bool(item.get("supported"))
        for item in (case.get("claim_verification") or {}).get("claims") or []
    }
    claims = []
    for item in (case.get("claim_payload") or {}).get("claims") or []:
        claim_id = str(item.get("claim_id") or "")
        if not claim_id or not verdicts.get(claim_id):
            continue
        claims.append({
            key: item.get(key)
            for key in (
                "claim_id", "question_id", "question_text", "section", "claim_type",
                "subject_drug", "text", "card_id", "field_path", "evidence_span",
                "page_doc", "page_pdf", "unsupported_fact",
            )
        })
    return claims


def validate_generated_answer(
    *,
    payload: dict[str, Any],
    query: str,
    base_claims: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    estimates: list[dict[str, Any]],
    decision_packet: dict[str, Any],
) -> list[str]:
    errors = []
    generated = str(payload.get("answer") or "").strip()
    refs = payload.get("support_refs")
    if not generated:
        errors.append("missing_answer")
    if not isinstance(refs, list):
        return errors + ["support_refs_must_be_a_list"]
    if generated and not refs:
        errors.append("missing_support_refs")

    allowed = {
        "verified_base_claim": {str(item.get("claim_id")) for item in base_claims},
        "supplemental_field": set(values),
        "derived_estimate": {str(item.get("method")) for item in estimates},
        "decision_packet": set(decision_packet),
    }
    used = set()
    for ref in refs:
        if not isinstance(ref, dict):
            errors.append("invalid_support_ref")
            continue
        support_type = str(ref.get("support_type") or "")
        support_id = str(ref.get("support_id") or "")
        if support_type not in allowed or support_id not in allowed[support_type]:
            errors.append(f"unknown_support_ref:{support_type}:{support_id}")
        used.add((support_type, support_id))

    def numeric_tokens(value: Any) -> set[str]:
        text = re.sub(r"(?m)^\s*\d+[.、]\s*", "", str(value or ""))
        return {
            format(float(token), ".12g")
            for token in re.findall(r"\d+(?:\.\d+)?", text)
        }

    estimate_numbers = {
        format(float(number), ".12g")
        for item in estimates
        for number in (
            item.get("half_life_count"),
            item.get("input_half_life_hours"),
            item.get("estimated_five_half_lives_hours"),
            item.get("estimated_five_half_lives_days"),
            item.get("estimated_five_half_lives_days_rounded_1dp"),
            item.get("estimated_remaining_percent"),
            item.get("estimated_eliminated_percent"),
        )
        if number is not None
    }
    if estimate_numbers.intersection(numeric_tokens(generated)):
        if not any(kind == "derived_estimate" for kind, _ in used):
            errors.append("derived_number_without_estimate_reference")
        has_boundary = (
            any(phrase in generated for phrase in (
                "不能单独作为", "不等同于", "不代表", "不是",
                "仅为药代估算", "仅为机制性估算",
            ))
            and any(word in generated for word in ("哺乳", "安全", "临床", "依据"))
        )
        if not has_boundary:
            errors.append("derived_estimate_missing_nonclinical_boundary")

    used_supplemental_fields = {
        support_id
        for support_type, support_id in used
        if support_type == "supplemental_field"
    }
    allowed_numbers = numeric_tokens(query) | {"5"}
    for claim in base_claims:
        allowed_numbers.update(numeric_tokens(claim.get("text")))
        allowed_numbers.update(numeric_tokens(claim.get("page_doc")))
        allowed_numbers.update(numeric_tokens(claim.get("page_pdf")))
    for item in values.values():
        allowed_numbers.update(numeric_tokens(item.get("value")))
        allowed_numbers.update(numeric_tokens(item.get("source_span")))
    allowed_numbers.update(estimate_numbers)
    introduced = sorted(numeric_tokens(generated) - allowed_numbers)
    if introduced:
        errors.append("unbound_numbers:" + ",".join(introduced))
    return errors


def generate_supplemental_answer(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    session: dict[str, Any],
) -> dict[str, Any]:
    base_claims = verified_base_claims(case)
    values = dict(session.get("values") or {})
    estimates = list(session.get("derived_estimates") or [])
    decision_packet = dict(case.get("decision_packet") or {})
    remaining_critical_fields = [
        item["field"]
        for item in (session.get("request") or {}).get("required_fields") or []
        if str(item.get("priority", "")).startswith("critical") and item["field"] not in values
    ]
    packet_projection = {
        key: decision_packet.get(key)
        for key in (
            "action", "decision_scope", "review_level", "review_required",
            "release_status", "missing_information", "unsupported_facts",
        )
    }
    packet_projection["remaining_critical_fields"] = remaining_critical_fields
    generation_input = {
        "original_query": case.get("query"),
        "verified_base_claims": base_claims,
        "supplemental_values": values,
        "derived_estimates": estimates,
        "remaining_critical_fields": remaining_critical_fields,
        "decision_packet": packet_projection,
    }
    prompt = ANSWER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(generation_input, ensure_ascii=False)},
    ]
    response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        max_tokens=args.answer_max_tokens,
        timeout=args.timeout,
        temperature=0.0,
        max_retries=args.api_max_retries,
    )
    raw = answer_text(response).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Supplemental answer response must be a JSON object")
    errors = validate_generated_answer(
        payload=payload,
        query=str(case.get("query") or ""),
        base_claims=base_claims,
        values=values,
        estimates=estimates,
        decision_packet=generation_input["decision_packet"],
    )
    total_usage = usage(response)
    response_ids = [response.get("id")]
    revision_count = 0
    if errors:
        revision_count = 1
        messages.extend([
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": json.dumps({
                    "task": "Revise the JSON answer so every listed validation error is fixed. Return JSON only.",
                    "validation_errors": errors,
                }, ensure_ascii=False),
            },
        ])
        revision = call_chat_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.answer_max_tokens,
            timeout=args.timeout,
            temperature=0.0,
            max_retries=args.api_max_retries,
        )
        revised_raw = answer_text(revision).strip()
        if revised_raw.startswith("```"):
            revised_raw = re.sub(r"^```(?:json)?\s*", "", revised_raw, flags=re.I)
            revised_raw = re.sub(r"\s*```$", "", revised_raw)
        revised_payload = json.loads(revised_raw)
        if not isinstance(revised_payload, dict):
            raise ValueError("Revised supplemental answer response must be a JSON object")
        revised_errors = validate_generated_answer(
            payload=revised_payload,
            query=str(case.get("query") or ""),
            base_claims=base_claims,
            values=values,
            estimates=estimates,
            decision_packet=generation_input["decision_packet"],
        )
        payload = revised_payload
        errors = revised_errors
        total_usage = combine_usage(total_usage, usage(revision))
        response_ids.append(revision.get("id"))
    return {
        "answer": str(payload.get("answer") or "").strip(),
        "answer_status": payload.get("answer_status") or "conditional_pending_source_and_clinical_review",
        "support_refs": payload.get("support_refs") or [],
        "validation": {"passed": not errors, "errors": errors},
        "prompt_file": ANSWER_PROMPT_PATH.name,
        "prompt_sha256": prompt_hash(prompt),
        "response_ids": response_ids,
        "revision_count": revision_count,
        "usage": total_usage,
        "raw_payload": payload,
    }


def print_status(request: dict[str, Any], values: dict[str, dict[str, Any]]) -> None:
    required = request.get("required_fields") or []
    specs = {item.get("field"): item for item in required}
    print("\n已识别变量：")
    if not values:
        print("  （暂无）")
    for field, item in values.items():
        label = (specs.get(field) or {}).get("label") or FIELD_LABELS.get(field, field)
        unit = f" {item.get('unit')}" if item.get("unit") else ""
        print(f"  - {label}: {item.get('value')}{unit}")
    missing = [item["field"] for item in required if item["field"] not in values]
    print(
        "尚未补充的字段（可留空）："
        + (
            "、".join((specs.get(field) or {}).get("label") or FIELD_LABELS.get(field, field) for field in missing)
            if missing else "无"
        )
    )
    estimate = half_life_estimate(values)
    if estimate:
        print(
            f"机制性估算：根据已确认的半衰期输入，5个半衰期约 "
            f"{estimate['estimated_five_half_lives_hours']} 小时。"
            "该计算值是药代动力学估算，不等同于专业人员直接提供的恢复哺乳建议。"
        )


def main() -> None:
    args = parse_args()
    args.models = ",".join(parse_model_pool(args.model, args.models))
    os.environ["CHAT_MODELS"] = args.models
    print(f"[ChatAPI] Configured model pool: {args.models}", flush=True)
    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")
    records = load_json(args.answers)
    case = find_case(records, args.case_id)
    request = build_unified_supplement_request(case)
    if not request:
        raise ValueError("This case contains no missing case fact or external evidence gap.")
    output = args.output or args.answers.with_name(f"interactive_evidence_case_{args.case_id}.json")
    session = load_json(output) if args.resume and output.exists() else {
        "session_version": "3.1.0",
        "case_id": case.get("id"),
        "query": case.get("query"),
        "request": request,
        "model": args.model,
        "configured_model_pool": parse_model_pool(args.model, args.models),
        "prompt_file": PROMPT_PATH.name,
        "prompt_sha256": prompt_hash(PROMPT_PATH.read_text(encoding="utf-8").strip()),
        "turns": [],
        "values": {},
        "session_status": "awaiting_user_evidence",
    }
    session["session_version"] = "3.1.0"
    session["request"] = request
    session["prompt_file"] = PROMPT_PATH.name
    session["prompt_sha256"] = prompt_hash(PROMPT_PATH.read_text(encoding="utf-8").strip())
    values: dict[str, dict[str, Any]] = dict(session.get("values") or {})
    allowed_fields = {item["field"] for item in request.get("required_fields") or []}
    field_kinds = {
        item["field"]: item.get("field_kind") or "external_evidence"
        for item in request.get("required_fields") or []
    }
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    print("\n=== MedSafe 补充信息交互会话 ===")
    print("病例问题：" + str(case.get("query") or ""))
    print(str(request.get("question") or ""))
    print("本阶段输入默认视为专业人员确认的权威信息，无需填写置信度；来源信息为可选项。")
    print("可一次输入多个数据。命令：/show 查看；/done 保存结束；/quit 退出。")
    print_status(request, values)

    exit_command = "/done"
    for _ in range(args.max_turns):
        try:
            terminal_input = input("\n请提供补充信息：").strip()
        except EOFError:
            terminal_input = "/done"
        if not terminal_input:
            continue
        if terminal_input == "/show":
            print_status(request, values)
            continue
        if terminal_input in {"/done", "/quit"}:
            exit_command = terminal_input
            break

        user_payload = {
            "allowed_fields": sorted(allowed_fields),
            "field_specs": request.get("required_fields") or [],
            "existing_values": values,
            "terminal_input": terminal_input,
        }
        try:
            response = call_chat_api(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                max_tokens=1400,
                timeout=args.timeout,
                temperature=0.0,
                max_retries=args.api_max_retries,
            )
            payload = parse_payload(answer_text(response))
        except Exception as exc:
            session["turns"].append({
                "turn": len(session["turns"]) + 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "terminal_input": terminal_input,
                "accepted_entries": [],
                "rejected_entries": [{"field": "", "errors": "model_or_json_error"}],
                "error": f"{type(exc).__name__}: {exc}",
            })
            session["values"] = values
            save_json(output, session)
            print(f"本轮模型解析失败，输入已保存，可重新表述后继续：{type(exc).__name__}: {exc}")
            continue
        accepted, rejected = validate_entries(
            terminal_input=terminal_input,
            payload=payload,
            allowed_fields=allowed_fields,
            field_kinds=field_kinds,
        )
        for item in accepted:
            values[item["field"]] = item
        session["turns"].append({
            "turn": len(session["turns"]) + 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_input": terminal_input,
            "accepted_entries": accepted,
            "rejected_entries": rejected,
            "unmapped_text": payload.get("unmapped_text") or "",
            "response_id": response.get("id"),
            "usage": response.get("usage") or {},
        })
        session["values"] = values
        session["derived_estimates"] = [value for value in [half_life_estimate(values)] if value]
        save_json(output, session)
        if rejected:
            print("以下抽取项未通过结构校验：" + json.dumps(rejected, ensure_ascii=False))
        print_status(request, values)

    requested_fields = [item["field"] for item in request.get("required_fields") or []]
    critical = [
        item["field"]
        for item in request.get("required_fields") or []
        if str(item.get("priority", "")).startswith("critical")
    ]
    session["values"] = values
    session["derived_estimates"] = [value for value in [half_life_estimate(values)] if value]
    all_requested_complete = all(field in values for field in requested_fields)
    if exit_command == "/done":
        session["session_status"] = (
            "supplementation_complete"
            if all_requested_complete
            else "supplementation_complete_with_missing_fields"
        )
        session["collection_closed_by_user"] = True
    else:
        session["session_status"] = (
            "critical_fields_complete_additional_information_missing"
            if critical and all(field in values for field in critical)
            else "awaiting_additional_user_information"
        )
        session["collection_closed_by_user"] = False
    session["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output, session)
    print("\n会话已保存：" + str(output))
    print("状态：" + session["session_status"])

    if exit_command == "/done" and not args.no_generate_answer:
        print("\n正在根据指南卡片和专业人员补充变量生成回答...")
        try:
            generated = generate_supplemental_answer(args=args, case=case, session=session)
            session["answer_generation"] = generated
            session["answer_generation_status"] = (
                "generated_from_accepted_inputs"
                if generated["validation"]["passed"]
                else "generated_draft_failed_support_validation"
            )
            save_json(output, session)
            answer_output = args.answer_output or output.with_name(
                f"interactive_answer_case_{args.case_id}.md"
            )
            answer_output.parent.mkdir(parents=True, exist_ok=True)
            answer_output.write_text(
                "\n".join([
                    f"# Case {case.get('id')} supplemental-evidence answer",
                    "",
                    f"**Query:** {case.get('query') or ''}",
                    "",
                    f"**Status:** `{session['answer_generation_status']}`",
                    "",
                    generated["answer"],
                    "",
                ]),
                encoding="utf-8",
            )
            print("\n=== 补充证据后的条件式回答 ===")
            print(generated["answer"])
            print("\n回答文件：" + str(answer_output))
            if not generated["validation"]["passed"]:
                print(
                    "警告：回答未通过支持关系校验，不应进入人工评价："
                    + json.dumps(generated["validation"]["errors"], ensure_ascii=False)
                )
        except Exception as exc:
            session["answer_generation_status"] = "generation_error"
            session["answer_generation_error"] = f"{type(exc).__name__}: {exc}"
            save_json(output, session)
            print(f"回答生成失败，交互变量仍已保存：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
