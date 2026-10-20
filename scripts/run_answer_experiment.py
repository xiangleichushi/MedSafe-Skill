#!/usr/bin/env python
"""Run leak-free controlled answer baselines and MedSafe-Skill ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MEDSAFE_DIR.parent
PROMPTS_DIR = MEDSAFE_DIR / "prompts"
IMPLEMENTATION_FILES = (
    SCRIPT_DIR / "run_answer_experiment.py",
    MEDSAFE_DIR / "core" / "action_policy.py",
    MEDSAFE_DIR / "core" / "claim_ledger.py",
    MEDSAFE_DIR / "core" / "evidence_request.py",
    MEDSAFE_DIR / "core" / "qasp_state_machine.py",
    MEDSAFE_DIR / "core" / "router.py",
    PROMPTS_DIR / "claim_json_repair_v1.txt",
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medsafe_skill.core.claim_ledger import (  # noqa: E402
    build_revision_prompt,
    parse_claim_payload,
    render_verified_claims,
    verify_claims,
)
from medsafe_skill.core.action_policy import human_review_notice  # noqa: E402
from medsafe_skill.core.evidence_request import render_evidence_requests  # noqa: E402
from medsafe_skill.core.observation_extractor import (  # noqa: E402
    apply_semantic_verification,
    parse_observation_payload,
    parse_semantic_verification_payload,
    semantic_verification_request,
    validate_observation_payload,
)
from medsafe_skill.core.reasoning_trace import build_structured_reasoning_trace  # noqa: E402
from medsafe_skill.core.orchestrator import execute_skill_plan  # noqa: E402
from medsafe_skill.core.qasp_state_machine import finalize_after_claim_verification  # noqa: E402
from medsafe_skill.core.router import RouteDecision  # noqa: E402
from medsafe_skill.scripts.load_card_store import EvidenceCardStore  # noqa: E402


DEFAULT_QUERY_PATH = MEDSAFE_DIR / "experiments" / "hallucination_public.json"
DEFAULT_RETRIEVAL = MEDSAFE_DIR / "output" / "skill_benchmark_retrieval_top_10.json"
DEFAULT_OUTPUT = MEDSAFE_DIR / "output" / "controlled_answer_experiment.json"
DEFAULT_CARDS = MEDSAFE_DIR / "resources" / "evidence_cards.jsonl"
DEFAULT_OBSERVATION_CACHE = MEDSAFE_DIR / "output" / "clinical_observations_cache.json"

GROUPS: dict[str, dict[str, Any]] = {
    "llm_only": {"mode": "direct", "prompt": "llm_only_v1.txt", "use_evidence": False},
    "standard_rag": {"mode": "direct", "prompt": "standard_rag_v1.txt", "use_evidence": True},
    "expert_prompt_rag": {"mode": "direct", "prompt": "expert_prompt_v1.txt", "use_evidence": True},
    "hrc_cards_only": {"mode": "direct", "prompt": "standard_rag_v1.txt", "use_evidence": True},
    "medsafe_full": {"mode": "skill", "disabled_skills": set(), "verify": True},
    "medsafe_no_gate": {"mode": "skill", "disabled_skills": {"evidence_sufficiency"}, "verify": True},
    "medsafe_no_missing": {"mode": "skill", "disabled_skills": {"missing_information"}, "verify": True},
    "medsafe_no_csrp": {"mode": "skill", "disabled_skills": {"safety_escalation"}, "verify": True},
    "medsafe_no_verifier": {"mode": "skill", "disabled_skills": {"claim_citation"}, "verify": False},
    "oracle_skill": {"mode": "skill", "disabled_skills": set(), "verify": True, "oracle": True},
}

_MODEL_POOL_POSITIONS: dict[tuple[str, tuple[str, ...]], int] = {}
TRANSPORT_COMPATIBLE_LEGACY_IMPLEMENTATION_HASHES = {
    # Generated before model-pool transport was added. The prompts, rules, and
    # evidence contracts are unchanged, so completed records remain reusable.
    "41a397296ab2f2a0b4413cb0fe44b9fc09c44b2139292a9a9d69e64459efc82d",
}


class ModelPoolExhaustedError(RuntimeError):
    """Raised after every configured chat model is unavailable."""


def parse_model_pool(
    primary_model: str,
    models: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return a stable, de-duplicated model fallback order."""

    raw: str | list[str] | tuple[str, ...] | None = models
    if raw is None:
        raw = os.getenv("CHAT_MODELS") or ""
    if isinstance(raw, str):
        candidates = re.split(r"[,;\s]+", raw.strip()) if raw.strip() else []
    else:
        candidates = [str(value).strip() for value in raw]
    ordered = [str(primary_model).strip(), *candidates]
    result = []
    for candidate in ordered:
        if candidate and candidate not in result:
            result.append(candidate)
    if not result:
        raise ValueError("At least one chat model must be configured")
    return result


def model_matches_pool(model: Any, pool: list[str]) -> bool:
    """Match provider-resolved model versions to their configured aliases."""

    value = str(model or "").strip()
    return any(value == candidate or value.startswith(candidate + "-") for candidate in pool)


def response_model_route(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    route = response.get("_medsafe_model_route")
    return dict(route) if isinstance(route, dict) else {}


def response_actual_model(response: Any, fallback: str) -> str:
    if isinstance(response, dict):
        route = response_model_route(response)
        return str(route.get("actual_model") or response.get("model") or fallback)
    return fallback


def unique_model_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for route in routes:
        if not route:
            continue
        marker = json.dumps(route, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            result.append(route)
    return result


def _response_error_summary(response: requests.Response | None) -> dict[str, Any]:
    if response is None:
        return {}
    message = ""
    code = None
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
                message = str(error.get("message") or "")
            else:
                code = payload.get("code")
                message = str(payload.get("message") or "")
    except ValueError:
        message = response.text or ""
    return {
        "http_status": response.status_code,
        "provider_code": code,
        "message": message[:500],
    }


def _quota_or_access_failure(status: int | None, detail: dict[str, Any]) -> bool:
    if status in {403, 429}:
        return True
    if status != 400:
        return False
    text = f"{detail.get('provider_code') or ''} {detail.get('message') or ''}".lower()
    markers = (
        "quota", "free tier", "allocation", "insufficient", "model not",
        "access denied", "permission", "arrearage",
    )
    return any(marker in text for marker in markers)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prompt_text(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def normalize_api_key(api_key: str, api_url: str) -> str:
    if api_key and "dashscope.aliyuncs.com" in api_url and not api_key.lower().startswith("bearer "):
        return f"Bearer {api_key}"
    return api_key


def call_chat_api(
    *,
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    temperature: float,
    max_retries: int = 2,
    models: str | list[str] | tuple[str, ...] | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint with auditable model fallback.

    The next model is used only for quota/access failures (403, exhausted 429,
    or a provider quota/access 400). Network and server errors are retried on
    the same model first. The selected position persists within the process so
    an exhausted model is not called again for every subsequent case.
    """

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if max_completion_tokens is not None and max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive")
    if thinking_budget is not None and (enable_thinking is not True or thinking_budget < 1):
        raise ValueError("thinking_budget requires enabled thinking and a positive budget")
    pool = parse_model_pool(model, models)
    state_key = (api_url, tuple(pool))
    start_index = min(_MODEL_POOL_POSITIONS.get(state_key, 0), len(pool) - 1)
    route_attempts: list[dict[str, Any]] = []

    for model_index in range(start_index, len(pool)):
        candidate = pool[model_index]
        switch_to_next = False
        for attempt in range(max_retries + 1):
            try:
                request_body = {"model": candidate, "messages": messages, "temperature": temperature}
                if max_completion_tokens is None:
                    request_body["max_tokens"] = max_tokens
                else:
                    request_body["max_completion_tokens"] = max_completion_tokens
                if enable_thinking is not None:
                    request_body["enable_thinking"] = enable_thinking
                if thinking_budget is not None:
                    request_body["thinking_budget"] = thinking_budget
                response = requests.post(
                    api_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": normalize_api_key(api_key, api_url),
                        "Content-Type": "application/json",
                        "User-Agent": "medsafe-controlled-experiment/1.0",
                    },
                    json=request_body,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                actual_model = str(data.get("model") or candidate)
                route_attempts.append({
                    "configured_model": candidate,
                    "attempt": attempt + 1,
                    "outcome": "success",
                    "actual_model": actual_model,
                    "response_id": data.get("id"),
                })
                _MODEL_POOL_POSITIONS[state_key] = model_index
                data["_medsafe_model_route"] = {
                    "pool": pool,
                    "selected_index": model_index,
                    "configured_model": candidate,
                    "actual_model": actual_model,
                    "fallback_used": model_index > 0,
                    "attempts": route_attempts,
                }
                return data
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = (
                    exc.response.status_code
                    if isinstance(exc, requests.HTTPError) and exc.response is not None
                    else None
                )
                detail = _response_error_summary(
                    exc.response if isinstance(exc, requests.HTTPError) else None
                )
                route_attempts.append({
                    "configured_model": candidate,
                    "attempt": attempt + 1,
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    **detail,
                })
                retryable_same_model = status is None or status == 429 or (
                    status is not None and status >= 500
                )
                quota_or_access = _quota_or_access_failure(status, detail)
                if retryable_same_model and attempt < max_retries:
                    delay = min(2 ** attempt, 8)
                    print(
                        f"[ChatAPI][WARN] model={candidate} {type(exc).__name__}; "
                        f"retry {attempt + 1}/{max_retries} after {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                if quota_or_access and model_index + 1 < len(pool):
                    next_model = pool[model_index + 1]
                    _MODEL_POOL_POSITIONS[state_key] = model_index + 1
                    print(
                        f"[ChatAPI][MODEL_SWITCH] model={candidate} status={status}; "
                        f"switching to {next_model}",
                        flush=True,
                    )
                    switch_to_next = True
                    break
                if quota_or_access:
                    failures = "; ".join(
                        f"{item.get('configured_model')}:{item.get('http_status') or item.get('error_type')}"
                        for item in route_attempts
                        if item.get("outcome") == "error"
                    )
                    raise ModelPoolExhaustedError(
                        f"All configured chat models are unavailable ({failures})"
                    ) from exc
                raise
        if not switch_to_next:
            break
    raise ModelPoolExhaustedError(
        "All configured chat models are unavailable: " + ", ".join(pool)
    )


def answer_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected chat response: {response}") from exc


def usage(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("usage")
    return value if isinstance(value, dict) else {}


def combine_usage(*values: dict[str, Any]) -> dict[str, int]:
    fields = {field for value in values for field in value if isinstance(value.get(field), (int, float))}
    return {field: int(sum(float(value.get(field, 0)) for value in values)) for field in fields}


def parse_claim_payload_with_repair(
    *,
    raw: str,
    response: dict[str, Any],
    task_context: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """Parse claim JSON and request a full schema-bound regeneration if invalid."""

    current_raw = raw
    current_response = response
    repairs: list[dict[str, Any]] = []
    repair_usage: dict[str, int] = {}
    max_repairs = max(0, int(getattr(args, "json_repair_attempts", 2)))
    repair_system = prompt_text("claim_json_repair_v1.txt")

    for attempt in range(max_repairs + 1):
        try:
            payload = parse_claim_payload(current_raw)
            return payload, current_raw, current_response, repairs, repair_usage
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt >= max_repairs:
                raise ValueError(
                    f"Claim JSON remained invalid after {max_repairs} repair attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(
                f"[Experiment][JSON] Invalid claim payload; repair "
                f"{attempt + 1}/{max_repairs}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            repair_input = {
                "original_task_context": task_context,
                "invalid_output": current_raw,
                "parser_error": f"{type(exc).__name__}: {exc}",
                "repair_attempt": attempt + 1,
            }
            repair_response = call_chat_api(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": json.dumps(repair_input, ensure_ascii=False)},
                ],
                max_tokens=max(args.max_tokens, 3200),
                timeout=args.timeout,
                temperature=0.0,
                max_retries=getattr(args, "api_max_retries", 2),
            )
            repairs.append({
                "attempt": attempt + 1,
                "parser_error": repair_input["parser_error"],
                "invalid_raw": current_raw,
                "invalid_response_id": (
                    current_response.get("id") if isinstance(current_response, dict) else None
                ),
                "repair_response_id": (
                    repair_response.get("id") if isinstance(repair_response, dict) else None
                ),
                "repair_prompt_file": "claim_json_repair_v1.txt",
                "repair_prompt_sha256": prompt_hash(repair_system),
                "model_route": response_model_route(repair_response),
                "usage": usage(repair_response),
            })
            repair_usage = combine_usage(repair_usage, usage(repair_response))
            current_raw = answer_text(repair_response)
            current_response = repair_response

    raise AssertionError("unreachable")


def extract_clinical_observations(query: str, args: argparse.Namespace) -> dict[str, Any]:
    extractor_system = prompt_text("observation_extractor_v1.txt")
    extractor_response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": extractor_system},
            {"role": "user", "content": query},
        ],
        max_tokens=args.observation_max_tokens,
        timeout=args.timeout,
        temperature=0.0,
        max_retries=getattr(args, "api_max_retries", 2),
    )
    raw = answer_text(extractor_response)
    payload = parse_observation_payload(raw)
    structurally_validated = validate_observation_payload(query=query, payload=payload)

    verifier_system = prompt_text("observation_semantic_verifier_v1.txt")
    verifier_request = semantic_verification_request(
        query=query,
        structurally_validated=structurally_validated,
    )
    verifier_response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": verifier_system},
            {"role": "user", "content": json.dumps(verifier_request, ensure_ascii=False)},
        ],
        max_tokens=getattr(args, "observation_semantic_max_tokens", args.observation_max_tokens),
        timeout=args.timeout,
        temperature=0.0,
        max_retries=getattr(args, "api_max_retries", 2),
    )
    verifier_payload = parse_semantic_verification_payload(answer_text(verifier_response))
    validated = apply_semantic_verification(
        structurally_validated=structurally_validated,
        verification_payload=verifier_payload,
    )
    return {
        "query_sha256": query_hash(query),
        "model": response_actual_model(extractor_response, args.model),
        "configured_model_pool": parse_model_pool(
            args.model, getattr(args, "models", None)
        ),
        "temperature": 0.0,
        "prompt_file": "observation_extractor_v1.txt",
        "prompt_sha256": prompt_hash(extractor_system),
        "raw_payload": payload,
        "structurally_validated_observations": structurally_validated,
        "semantic_verifier": {
            "model": response_actual_model(verifier_response, args.model),
            "model_route": response_model_route(verifier_response),
            "temperature": 0.0,
            "prompt_file": "observation_semantic_verifier_v1.txt",
            "prompt_sha256": prompt_hash(verifier_system),
            "request": verifier_request,
            "raw_payload": verifier_payload,
            "usage": usage(verifier_response),
            "raw_response": verifier_response,
        },
        "validated_observations": validated,
        "usage": combine_usage(usage(extractor_response), usage(verifier_response)),
        "model_route": response_model_route(extractor_response),
        "raw_response": extractor_response,
    }


def observation_record_is_compatible(record: Any) -> bool:
    """Reject stale span-only caches when the semantic verifier is required."""

    if not isinstance(record, dict):
        return False
    observations = record.get("validated_observations") or {}
    verifier = record.get("semantic_verifier") or {}
    current_extractor_hash = prompt_hash(prompt_text("observation_extractor_v1.txt"))
    current_verifier_hash = prompt_hash(prompt_text("observation_semantic_verifier_v1.txt"))
    return (
        observations.get("extraction_version") == "2.0.0"
        and observations.get("observation_source") == "llm_extracted_semantically_validated"
        and observations.get("validation", {}).get("semantic_verification_completed") is True
        and record.get("prompt_sha256") == current_extractor_hash
        and verifier.get("prompt_file") == "observation_semantic_verifier_v1.txt"
        and verifier.get("prompt_sha256") == current_verifier_hash
    )


def retrieval_hits(item: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not item:
        return []
    return list((item.get("medsafe_skill") or {}).get("retrieved_cards") or [])


def full_cards(hits: list[dict[str, Any]], store: EvidenceCardStore, max_cards: int) -> list[dict[str, Any]]:
    cards = []
    seen = set()
    for hit in hits:
        card_id = (hit.get("card") or {}).get("card_id") if hit.get("matched") else None
        if not card_id or card_id in seen:
            continue
        card = store.get(card_id)
        if card:
            cards.append(card)
            seen.add(card_id)
        if len(cards) >= max_cards:
            break
    return cards


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card.get("card_id"),
        "identity": card.get("identity"),
        "clinical_profile": card.get("clinical_profile"),
        "safety": card.get("safety"),
        "pharmacokinetics": card.get("pharmacokinetics"),
        "source_trace": card.get("source_trace"),
        "verification": card.get("verification"),
    }


def direct_user_prompt(query: str, cards: list[dict[str, Any]] | None) -> str:
    if cards is None:
        return query
    context = json.dumps({"retrieved_evidence_cards": [compact_card(card) for card in cards]}, ensure_ascii=False)
    return f"用户问题：\n{query}\n\n检索证据：\n{context}\n\n请回答用户问题。"


def generation_plan_projection(plan: dict[str, Any]) -> dict[str, Any]:
    """Keep the decision contract for generation without duplicating audit traces."""

    workflow = plan.get("clinical_workflow") or {}
    route = plan.get("route") or {}
    evidence = plan.get("evidence_sufficiency") or {}
    card_decisions = []
    for item in workflow.get("card_decisions") or []:
        card_decisions.append({
            "card_id": item.get("card_id"),
            "drug": item.get("drug"),
            "action_label": item.get("action_label"),
            "review_level": item.get("review_level"),
            "rationale": item.get("rationale") or [],
            "citation": item.get("citation") or {},
        })
    return {
        "generation_projection_version": "1.0.0",
        "orchestrator_version": plan.get("orchestrator_version"),
        "route": {
            "context": route.get("context"),
            "requested_facts": route.get("requested_facts") or [],
            "unsafe_premise_detected": bool(route.get("unsafe_premise_detected")),
        },
        "selected_skills": plan.get("selected_skills") or [],
        "evidence_sufficiency": {
            "card_count": evidence.get("card_count"),
            "requested_facts": evidence.get("requested_facts") or [],
            "unsupported_facts": evidence.get("unsupported_facts") or [],
            "all_requested_facts_supported": evidence.get("all_requested_facts_supported"),
        },
        "missing_information": plan.get("missing_information") or [],
        "clinical_workflow": {
            "current_state": workflow.get("current_state"),
            "decision_packet": workflow.get("decision_packet") or {},
            "card_decision_summary": card_decisions,
            "response_mode": workflow.get("response_mode"),
        },
        "response_mode": plan.get("response_mode"),
        "requires_revision_verification": plan.get("requires_revision_verification"),
        "retrieval_quality": plan.get("retrieval_quality") or {},
    }


def skill_user_prompt(query: str, plan: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    payload = {
        "query": query,
        "execution_plan": generation_plan_projection(plan),
        "evidence_cards": [compact_card(card) for card in cards],
    }
    return "请根据以下执行计划和证据生成声明 JSON：\n" + json.dumps(payload, ensure_ascii=False)


def oracle_route(item: dict[str, Any]) -> RouteDecision:
    skills = item.get("required_skills")
    facts = item.get("requested_facts") or item.get("required_facts")
    if not isinstance(skills, list) or not isinstance(facts, list):
        raise ValueError("Oracle key requires required_skills and requested_facts arrays")
    return RouteDecision(
        router_version="oracle-key-1.0.0",
        context=str(item.get("context") or "unspecified"),
        selected_skills=[str(value) for value in skills],
        requested_facts=[str(value) for value in facts],
        missing_information=list(item.get("missing_information") or []),
        unsafe_premise_detected=bool(item.get("unsafe_premise_detected")),
        trigger_trace={"source": "hidden_oracle_key"},
    )


def append_sources(answer: str, verification: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    supported_ids = {
        item.get("card_id")
        for item in verification.get("claims", [])
        if item.get("supported") and item.get("card_id")
    }
    lines = []
    for card in cards:
        if card.get("card_id") not in supported_ids:
            continue
        source = card.get("source_trace", {})
        drug = card.get("identity", {}).get("canonical_name")
        lines.append(
            f"- {drug}：《妊娠哺乳期用药指南》第{source.get('page_doc')}页"
            f"（PDF第{source.get('page_pdf')}页）"
        )
    if not lines:
        return answer
    return answer.rstrip() + "\n来源：\n" + "\n".join(lines)


def run_direct(
    *,
    query: str,
    cards: list[dict[str, Any]],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = prompt_text(config["prompt"])
    response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": direct_user_prompt(query, cards if config["use_evidence"] else None)},
        ],
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        temperature=args.temperature,
        max_retries=getattr(args, "api_max_retries", 2),
    )
    return {
        "answer": answer_text(response),
        "model": response_actual_model(response, args.model),
        "model_routes": [response_model_route(response)],
        "prompt_file": config["prompt"],
        "prompt_sha256": prompt_hash(system),
        "usage": usage(response),
        "raw_response": response,
    }


def apply_retrieval_quality_guardrail(
    plan: dict[str, Any],
    retrieval_quality: dict[str, Any],
) -> None:
    """Synchronize all decision axes when the Critic rejects final retrieval."""

    if retrieval_quality.get("status") != "failed_after_retries":
        return
    unsupported = plan["evidence_sufficiency"].setdefault("unsupported_facts", [])
    if "retrieval_subquestion_coverage" not in unsupported:
        unsupported.append("retrieval_subquestion_coverage")
    workflow = plan["clinical_workflow"]
    packet = workflow.setdefault("decision_packet", {})
    packet_unsupported = packet.setdefault("unsupported_facts", [])
    if "retrieval_subquestion_coverage" not in packet_unsupported:
        packet_unsupported.append("retrieval_subquestion_coverage")
    plan["evidence_sufficiency"]["all_requested_facts_supported"] = False
    packet["action"] = "undetermined"
    packet["clinical_action"] = "undetermined"
    packet["evidence_bound_action"] = "undetermined"
    packet["evidence_status"] = "partial"
    packet["decision_scope"] = "retrieval_incomplete"
    packet["review_level"] = (
        "specialist_review"
        if packet.get("review_level") == "specialist_review"
        else "pharmacist_review"
    )
    packet["review_required"] = True
    packet["release_status"] = "advice_draft_human_review_required"
    actions = [
        action
        for action in (packet.get("workflow_actions") or [])
        if action not in {"ready", "external_search"}
    ]
    for action in ("ask_clarification", packet["review_level"]):
        if action not in actions:
            actions.append(action)
    packet["workflow_actions"] = actions
    packet["workflow_action"] = "ask_clarification"
    action_plan = (workflow.get("qasp") or {}).get("P_action_plan") or {}
    action_plan.update({
        "final_action": "undetermined",
        "clinical_action": "undetermined",
        "evidence_bound_action": "undetermined",
        "evidence_status": "partial",
        "decision_scope": "retrieval_incomplete",
        "review_level": packet["review_level"],
        "workflow_action": packet["workflow_action"],
        "workflow_actions": actions,
        "unsupported_facts": packet_unsupported,
        "response_mode": "external_evidence_supplementation_or_human_review",
    })
    workflow.setdefault("audit", {})["retrieval_quality_guardrail"] = {
        "applied": True,
        "status": retrieval_quality.get("status"),
        "reason": retrieval_quality.get("reason"),
        "action_effect": "evidence_bound_action_downgraded_to_undetermined",
        "evidence_interpretation": "relevant_evidence_cards_missing",
        "next_step": "ask_clarification_external_evidence_or_human_review",
    }
    plan["response_mode"] = "external_evidence_supplementation_or_human_review"
    workflow["response_mode"] = "external_evidence_supplementation_or_human_review"


def run_skill(
    *,
    query: str,
    hits: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    config: dict[str, Any],
    args: argparse.Namespace,
    oracle_item: dict[str, Any] | None,
    observation_record: dict[str, Any] | None = None,
    retrieval_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_override = oracle_route(oracle_item or {}) if config.get("oracle") else None
    plan = execute_skill_plan(
        query=query,
        retrieved_hits=hits,
        store=args.store,
        max_cards=args.max_cards,
        route_override=route_override,
        disabled_skills=set(config.get("disabled_skills") or set()),
        validated_observations=(observation_record or {}).get("validated_observations"),
    )
    retrieval_quality = (retrieval_item or {}).get("retrieval_quality") or {}
    plan["retrieval_quality"] = retrieval_quality
    apply_retrieval_quality_guardrail(plan, retrieval_quality)
    system = prompt_text("claim_generator_v1.txt")
    initial_user = skill_user_prompt(query, plan, cards)
    response = call_chat_api(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": initial_user},
        ],
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        temperature=args.temperature,
        max_retries=getattr(args, "api_max_retries", 2),
    )
    raw = answer_text(response)
    payload, raw, final_raw_response, format_repairs, format_repair_usage = (
        parse_claim_payload_with_repair(
            raw=raw,
            response=response,
            task_context=initial_user,
            args=args,
        )
    )
    unsupported_facts = plan["evidence_sufficiency"].get("unsupported_facts", [])
    verification = verify_claims(payload["claims"], cards, unsupported_facts)
    revisions = []
    total_usage = combine_usage(
        (observation_record or {}).get("usage", {}),
        usage(response),
        format_repair_usage,
    )

    if config.get("verify") and not verification["all_supported"]:
        revision_response = call_chat_api(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": initial_user},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": build_revision_prompt(payload, verification)},
            ],
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            max_retries=getattr(args, "api_max_retries", 2),
        )
        revised_raw = answer_text(revision_response)
        revision_prompt = build_revision_prompt(payload, verification)
        revision_task_context = initial_user + "\n\nCLAIM REVISION REQUIREMENTS:\n" + revision_prompt
        revised_payload, revised_raw, final_revision_response, revision_format_repairs, revision_repair_usage = (
            parse_claim_payload_with_repair(
                raw=revised_raw,
                response=revision_response,
                task_context=revision_task_context,
                args=args,
            )
        )
        revised_verification = verify_claims(revised_payload["claims"], cards, unsupported_facts)
        format_repairs.extend(revision_format_repairs)
        revisions.append({
            "payload": revised_payload,
            "verification": revised_verification,
            "usage": combine_usage(usage(revision_response), revision_repair_usage),
            "raw_response": final_revision_response,
        })
        total_usage = combine_usage(total_usage, usage(revision_response), revision_repair_usage)
        payload = revised_payload
        verification = revised_verification
        final_raw_response = final_revision_response

    if config.get("verify"):
        rendered = render_verified_claims(payload, verification)
    else:
        allow_all = {
            "claims": [
                {"claim_id": str(item.get("claim_id") or f"C{index:03d}"), "supported": True}
                for index, item in enumerate(payload["claims"], start=1)
            ]
        }
        rendered = render_verified_claims(payload, allow_all)
    rendered = append_sources(rendered, verification, cards)
    plan["clinical_workflow"] = finalize_after_claim_verification(
        plan["clinical_workflow"],
        all_supported=bool(verification.get("all_supported")),
    )
    decision_packet = plan["clinical_workflow"]["decision_packet"]
    evidence_request_text = render_evidence_requests(decision_packet.get("evidence_requests") or [])
    if evidence_request_text:
        rendered = rendered.rstrip() + "\n" + evidence_request_text
    review_notice = human_review_notice(decision_packet)
    rendered = rendered.rstrip() + "\n" + review_notice
    reasoning_trace = build_structured_reasoning_trace(
        plan=plan,
        observation_record=observation_record,
        claim_verification=verification,
    )
    return {
        "answer": rendered,
        "model": response_actual_model(final_raw_response, args.model),
        "configured_model_pool": parse_model_pool(
            args.model, getattr(args, "models", None)
        ),
        "model_routes": unique_model_routes([
            route
            for route in [
                response_model_route(response),
                *[
                    repair.get("model_route") or {}
                    for repair in format_repairs
                ],
                *[
                    response_model_route(revision.get("raw_response"))
                    for revision in revisions
                ],
                response_model_route(final_raw_response),
            ]
            if route
        ]),
        "plan": plan,
        "claim_payload": payload,
        "claim_verification": verification,
        "clinical_observation_extraction": observation_record,
        "decision_packet": decision_packet,
        "human_review_notice": review_notice,
        "reasoning_trace": reasoning_trace,
        "revision_count": len(revisions),
        "revisions": revisions,
        "json_format_repair_count": len(format_repairs),
        "json_format_repairs": format_repairs,
        "prompt_file": "claim_generator_v1.txt",
        "prompt_sha256": prompt_hash(system),
        "usage": total_usage,
        "raw_response": final_raw_response,
        "initial_raw_response": response,
    }


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _response_id(response: Any) -> str | None:
    return str(response.get("id")) if isinstance(response, dict) and response.get("id") else None


def write_run_artifacts(
    *,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    query_count: int,
    retrieval_sha: str,
) -> Path:
    """Write answer, trace, API audit, and reproducibility artifacts."""

    artifact_dir = args.artifact_dir or args.output.with_name(args.output.stem + "_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    successful = [item for item in results if not item.get("error")]
    answer_rows = [
        {
            "id": item.get("id"),
            "group": item.get("group"),
            "query": item.get("query"),
            "case_source": item.get("case_source"),
            "answer": item.get("answer"),
            "retrieved_card_ids": item.get("retrieved_card_ids") or [],
            "final_action": (item.get("decision_packet") or {}).get("action"),
            "review_level": (item.get("decision_packet") or {}).get("review_level"),
            "human_review_notice": item.get("human_review_notice"),
        }
        for item in successful
    ]
    save_jsonl(artifact_dir / "answers_full.jsonl", answer_rows)

    readable = [
        "# MedSafe-Skill clinical case answers",
        "",
        "This file is generated for rapid human inspection. The structured audit trail is in `reasoning_traces.jsonl`.",
        "",
    ]
    for item in successful:
        packet = item.get("decision_packet") or {}
        state = ((item.get("plan") or {}).get("clinical_workflow") or {}).get("current_state")
        readable.extend([
            f"## Case {item.get('id')} | {item.get('group')}",
            "",
            f"**Query:** {item.get('query') or ''}",
            "",
            f"**Final action:** `{packet.get('action') or 'not_applicable'}`  ",
            f"**Review level:** `{packet.get('review_level') or 'not_applicable'}`  ",
            f"**Terminal state:** `{state or 'not_applicable'}`  ",
            f"**Evidence Cards:** {', '.join(str(value) for value in (item.get('retrieved_card_ids') or []) if value) or 'None'}",
            "",
            "### Final answer",
            "",
            str(item.get("answer") or ""),
            "",
        ])
    (artifact_dir / "answers_readable.md").write_text("\n".join(readable), encoding="utf-8")

    trace_rows = [
        {
            "id": item.get("id"),
            "group": item.get("group"),
            "query": item.get("query"),
            "case_source": item.get("case_source"),
            "reasoning_trace": item.get("reasoning_trace"),
        }
        for item in successful
        if item.get("reasoning_trace")
    ]
    save_jsonl(artifact_dir / "reasoning_traces.jsonl", trace_rows)

    audit_rows = []
    for item in results:
        observation = item.get("clinical_observation_extraction") or {}
        verifier = observation.get("semantic_verifier") or {}
        audit_rows.append({
            "id": item.get("id"),
            "group": item.get("group"),
            "model": item.get("model"),
            "configured_model_pool": item.get("configured_model_pool") or [],
            "model_routes": item.get("model_routes") or [],
            "fallback_used": any(
                route.get("fallback_used")
                for route in (item.get("model_routes") or [])
            ),
            "temperature": item.get("temperature"),
            "prompt_file": item.get("prompt_file"),
            "prompt_sha256": item.get("prompt_sha256"),
            "response_id": _response_id(item.get("raw_response")),
            "usage": item.get("usage") or {},
            "elapsed_seconds": item.get("elapsed_seconds"),
            "observation_cache_hit": observation.get("cache_hit"),
            "observation_prompt_sha256": observation.get("prompt_sha256"),
            "semantic_verifier_prompt_sha256": verifier.get("prompt_sha256"),
            "semantic_verifier_response_id": _response_id(verifier.get("raw_response")),
            "revision_count": item.get("revision_count"),
            "json_format_repair_count": item.get("json_format_repair_count"),
            "json_format_repair_response_ids": [
                repair.get("repair_response_id")
                for repair in (item.get("json_format_repairs") or [])
                if repair.get("repair_response_id")
            ],
            "error": item.get("error"),
        })
    save_jsonl(artifact_dir / "api_audit.jsonl", audit_rows)

    action_counts = Counter(
        str((item.get("decision_packet") or {}).get("action") or "not_applicable")
        for item in successful
    )
    review_counts = Counter(
        str((item.get("decision_packet") or {}).get("review_level") or "not_applicable")
        for item in successful
    )
    state_counts = Counter(
        str(((item.get("plan") or {}).get("clinical_workflow") or {}).get("current_state") or "not_applicable")
        for item in successful
    )
    save_json(artifact_dir / "run_summary.json", {
        "query_count": query_count,
        "record_count": len(results),
        "successful_record_count": len(successful),
        "error_record_count": len(results) - len(successful),
        "groups": args.groups,
        "action_counts": dict(sorted(action_counts.items())),
        "review_level_counts": dict(sorted(review_counts.items())),
        "terminal_state_counts": dict(sorted(state_counts.items())),
        "actual_model_counts": dict(sorted(Counter(
            str(item.get("model") or "unknown") for item in successful
        ).items())),
        "fallback_record_count": sum(
            any(route.get("fallback_used") for route in (item.get("model_routes") or []))
            for item in successful
        ),
    })

    review_fields = [
        "id", "group", "query", "answer", "final_action", "review_level",
        "actual_model", "fallback_used",
        "final_state", "reasoning_trace_available",
    ]
    with (artifact_dir / "human_evaluation_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=review_fields)
        writer.writeheader()
        for item in successful:
            writer.writerow({
                "id": item.get("id"),
                "group": item.get("group"),
                "query": item.get("query"),
                "answer": item.get("answer"),
                "final_action": (item.get("decision_packet") or {}).get("action"),
                "review_level": (item.get("decision_packet") or {}).get("review_level"),
                "actual_model": item.get("model"),
                "fallback_used": any(
                    route.get("fallback_used")
                    for route in (item.get("model_routes") or [])
                ),
                "final_state": ((item.get("plan") or {}).get("clinical_workflow") or {}).get("current_state"),
                "reasoning_trace_available": bool(item.get("reasoning_trace")),
            })

    clinical_review_fields = [
        "id", "query", "answer",
        "guideline_consistency_1_5", "clinical_correctness_1_5",
        "completeness_1_5", "no_unsupported_claims_1_5", "safety_1_5",
        "potential_clinical_risk", "reviewer_comments",
    ]
    with (artifact_dir / "clinical_answer_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=clinical_review_fields)
        writer.writeheader()
        for item in successful:
            writer.writerow({
                "id": item.get("id"),
                "query": item.get("query"),
                "answer": item.get("answer"),
            })

    manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": query_count,
        "record_count": len(results),
        "successful_record_count": len(successful),
        "groups": args.groups,
        "model": args.model,
        "configured_model_pool": parse_model_pool(
            args.model, getattr(args, "models", None)
        ),
        "actual_model_counts": dict(sorted(Counter(
            str(item.get("model") or "unknown") for item in successful
        ).items())),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "observation_max_tokens": args.observation_max_tokens,
        "observation_semantic_max_tokens": args.observation_semantic_max_tokens,
        "query_path": str(args.query_path.resolve()),
        "query_sha256": file_hash(args.query_path),
        "retrieval_output": str(args.retrieval_output.resolve()),
        "retrieval_sha256": retrieval_sha,
        "retrieval_sha256_values_in_records": sorted({
            str(item.get("retrieval_sha256"))
            for item in results
            if item.get("retrieval_sha256")
        }),
        "cards_path": str(args.cards.resolve()),
        "cards_sha256": file_hash(args.cards),
        "observation_cache": str(args.observation_cache.resolve()),
        "observation_cache_sha256": file_hash(args.observation_cache) if args.observation_cache.exists() else None,
        "primary_output": str(args.output.resolve()),
        "primary_output_sha256": file_hash(args.output),
        "prompt_hashes": {
            path.name: file_hash(path)
            for path in sorted(PROMPTS_DIR.glob("*.txt"))
        },
        "implementation_sha256": implementation_hash(),
        "notes": [
            "reasoning_traces.jsonl stores structured auditable traces, not hidden chain-of-thought.",
            "human_evaluation_candidates.csv is unblinded; use the dedicated review builder for blinded scoring.",
            "clinical_answer_review.csv supports post-output clinical rating and is not a gold-label file.",
        ],
    }
    save_json(artifact_dir / "run_manifest.json", manifest)
    return artifact_dir


def validate_end_to_end_smoke(
    *,
    results: list[dict[str, Any]],
    expected_ids: set[Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Fail a smoke run when it bypasses any primary MedSafe stage."""

    items = [
        item for item in results
        if item.get("id") in expected_ids and item.get("group") == "medsafe_full"
    ]
    errors = []
    if len(items) != len(expected_ids):
        errors.append(f"expected {len(expected_ids)} medsafe_full records, found {len(items)}")
    terminal_states = {
        "human_review_required", "ready_for_clinician_confirmation", "abstained",
        "awaiting_external_evidence",
    }
    for item in items:
        qid = item.get("id")
        if item.get("error"):
            errors.append(f"id={qid}: result contains error")
            continue
        observation = (item.get("clinical_observation_extraction") or {}).get("validated_observations") or {}
        if observation.get("observation_source") != "llm_extracted_semantically_validated":
            errors.append(f"id={qid}: semantic observation validation not completed")
        for key in ("answer", "claim_payload", "claim_verification", "decision_packet", "reasoning_trace"):
            if not item.get(key):
                errors.append(f"id={qid}: missing {key}")
        notice = str(item.get("human_review_notice") or "")
        if not notice or not str(item.get("answer") or "").rstrip().endswith(notice):
            errors.append(f"id={qid}: review notice is not the final answer line")
        state = ((item.get("plan") or {}).get("clinical_workflow") or {}).get("current_state")
        if state not in terminal_states:
            errors.append(f"id={qid}: non-terminal workflow state {state}")
        payload_claims = (item.get("claim_payload") or {}).get("claims") or []
        for claim in payload_claims:
            if claim.get("claim_type") == "medical":
                if not claim.get("question_id"):
                    errors.append(f"id={qid}: medical claim missing question_id")
                if not claim.get("subject_drug"):
                    errors.append(f"id={qid}: medical claim missing subject_drug")
        requested = (((item.get("plan") or {}).get("route") or {}).get("requested_facts") or [])
        fields = observation.get("fields") or {}
        exposure_status = (fields.get("drug_exposure_status") or {}).get("value")
        query = str(item.get("query") or "")
        answer = str(item.get("answer") or "")
        asks_post_stop_breastfeeding = (
            exposure_status == "stopped"
            and any(term in query for term in ("哺乳", "喂奶", "母乳"))
            and any(term in query for term in ("可否", "能否", "可以", "能不能"))
        )
        if asks_post_stop_breastfeeding and "breastfeeding_interval" not in requested:
            errors.append(f"id={qid}: stopped-drug breastfeeding question was not routed as breastfeeding_interval")
        stopped_drug_use_phrases = (
            "支持在哺乳期继续使用该药",
            "建议继续使用该药",
            "可以继续使用该药",
            "可继续使用该药",
        )
        if exposure_status == "stopped" and any(phrase in answer for phrase in stopped_drug_use_phrases):
            errors.append(f"id={qid}: stopped-drug answer incorrectly recommends continuing the medication")
        if len(requested) > 1 and payload_claims:
            question_ids = {claim.get("question_id") for claim in payload_claims if claim.get("question_id")}
            if len(question_ids) < 2:
                errors.append(f"id={qid}: multi-fact query was not separated into multiple answer scopes")
        quality = (item.get("plan") or {}).get("retrieval_quality") or {}
        if quality.get("status") == "critic_unavailable":
            errors.append(f"id={qid}: Critic API was unavailable; retrieval quality was not validated")
        if quality.get("status") == "failed_after_retries":
            has_retrieval_abstention = any(
                claim.get("claim_type") == "evidence_absence"
                and claim.get("unsupported_fact") == "retrieval_subquestion_coverage"
                for claim in payload_claims
            )
            if not has_retrieval_abstention:
                errors.append(f"id={qid}: failed retrieval was not disclosed as evidence absence")
            packet_unsupported = (item.get("decision_packet") or {}).get("unsupported_facts") or []
            if "retrieval_subquestion_coverage" not in packet_unsupported:
                errors.append(f"id={qid}: failed retrieval gap is missing from DecisionPacket")

    report = {
        "smoke_validation_version": "1.1.0",
        "expected_ids": sorted(str(value) for value in expected_ids),
        "checked_medsafe_full_records": len(items),
        "passed": not errors,
        "errors": errors,
    }
    save_json(artifact_dir / "end_to_end_smoke_report.json", report)
    if errors:
        raise RuntimeError("End-to-end smoke validation failed: " + "; ".join(errors))
    return report


def result_is_compatible(
    item: dict[str, Any],
    *,
    args: argparse.Namespace,
    retrieval_sha: str,
    query_lookup: dict[Any, str],
) -> bool:
    """Invalidate resumed answers when evidence, prompts, model, or query changed."""

    if item.get("error"):
        return False
    group = str(item.get("group") or "")
    config = GROUPS.get(group)
    if not config:
        return False
    prompt_name = "claim_generator_v1.txt" if config["mode"] == "skill" else config["prompt"]
    expected_prompt_hash = prompt_hash(prompt_text(prompt_name))
    qid = item.get("id")
    configured_pool = parse_model_pool(args.model, getattr(args, "models", None))
    model_policy = getattr(args, "resume_model_policy", "pool")
    if model_policy == "preserve":
        model_compatible = bool(item.get("model"))
    elif model_policy == "exact":
        model_compatible = model_matches_pool(item.get("model"), [args.model])
    else:
        model_compatible = model_matches_pool(item.get("model"), configured_pool)
    implementation_compatible = (
        item.get("implementation_sha256") == implementation_hash()
        or item.get("implementation_sha256")
        in TRANSPORT_COMPATIBLE_LEGACY_IMPLEMENTATION_HASHES
    )
    return (
        item.get("query") == query_lookup.get(qid)
        and item.get("retrieval_sha256") == retrieval_sha
        and item.get("prompt_sha256") == expected_prompt_hash
        and implementation_compatible
        and model_compatible
        and float(item.get("temperature", -1)) == float(args.temperature)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-path", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--retrieval-output", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--groups", nargs="+", default=["llm_only", "standard_rag", "hrc_cards_only", "medsafe_full"])
    parser.add_argument("--oracle-key", type=Path)
    parser.add_argument("--observation-cache", type=Path, default=DEFAULT_OBSERVATION_CACHE)
    parser.add_argument("--observation-max-tokens", type=int, default=1200)
    parser.add_argument("--observation-semantic-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--observation-failure-policy",
        choices=["error", "deterministic_fallback"],
        default="error",
        help="Use deterministic_fallback only for debugging; primary experiments should use error.",
    )
    parser.add_argument("--max-cards", type=int, default=5)
    parser.add_argument("--api-url", default=os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--api-key", default=os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or "qwen3.7-max")
    parser.add_argument(
        "--models",
        default=os.getenv("CHAT_MODELS") or "",
        help="Comma-separated fallback models. CHAT_MODEL remains the primary model.",
    )
    parser.add_argument(
        "--resume-model-policy",
        choices=["exact", "pool", "preserve"],
        default="pool",
        help=(
            "Model compatibility for successful --resume records: exact=primary "
            "alias only, pool=any configured alias, preserve=keep any recorded "
            "model while all query/evidence/prompt/implementation checks still pass."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--json-repair-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-each", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--strict-smoke",
        action="store_true",
        help="Require every selected query to complete the full medsafe_full path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models = ",".join(parse_model_pool(args.model, args.models))
    os.environ["CHAT_MODELS"] = args.models
    print(f"[ChatAPI] Configured model pool: {args.models}", flush=True)
    unknown = [group for group in args.groups if group not in GROUPS]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}; available={sorted(GROUPS)}")
    if "oracle_skill" in args.groups and not args.oracle_key:
        raise ValueError("--oracle-key is required for oracle_skill")
    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")

    all_queries = load_json(args.query_path)
    query_lookup = {
        item.get("id"): str(item.get("query") or item.get("query_text") or "")
        for item in all_queries
    }
    queries = all_queries[args.start :]
    if args.limit is not None:
        queries = queries[: args.limit]
    retrieval_items = {item.get("id"): item for item in load_json(args.retrieval_output)}
    oracle_items = {item.get("id"): item for item in load_json(args.oracle_key)} if args.oracle_key else {}
    args.store = EvidenceCardStore.from_jsonl(args.cards)
    observation_cache = (
        load_json(args.observation_cache)
        if args.observation_cache.exists()
        else {}
    )
    if not isinstance(observation_cache, dict):
        raise ValueError("--observation-cache must contain a JSON object keyed by query SHA-256")

    retrieval_sha = file_hash(args.retrieval_output)
    existing_results = load_json(args.output) if args.resume and args.output.exists() else []
    results = [
        item for item in existing_results
        if result_is_compatible(
            item,
            args=args,
            retrieval_sha=retrieval_sha,
            query_lookup=query_lookup,
        )
    ]
    if existing_results and len(results) != len(existing_results):
        print(f"[Experiment] Invalidated {len(existing_results) - len(results)} stale/error records")
    done = {(item.get("id"), item.get("group")) for item in results}
    total = len(queries) * len(args.groups)
    progress = 0

    for query_item in queries:
        qid = query_item.get("id")
        query = str(query_item.get("query") or query_item.get("query_text") or "")
        retrieval_item = retrieval_items.get(qid)
        hits = retrieval_hits(retrieval_item)
        cards = full_cards(hits, args.store, args.max_cards)
        pending_skill_groups = [
            group
            for group in args.groups
            if (qid, group) not in done and GROUPS[group]["mode"] == "skill"
        ]
        observation_record: dict[str, Any] | None = None
        if pending_skill_groups:
            cache_key = query_hash(query)
            cached = observation_cache.get(cache_key)
            if observation_record_is_compatible(cached):
                observation_record = dict(cached)
                observation_record["cache_hit"] = True
            else:
                if cached:
                    print(f"[Experiment] Ignore stale observation cache id={qid}")
                print(f"[Experiment] Extract observations id={qid}")
                try:
                    observation_record = extract_clinical_observations(query, args)
                    observation_record["cache_hit"] = False
                    observation_cache[cache_key] = observation_record
                    save_json(args.observation_cache, observation_cache)
                except Exception as exc:
                    observation_record = {
                        "query_sha256": cache_key,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "validated_observations": None,
                        "usage": {},
                        "cache_hit": False,
                        "fallback_used": args.observation_failure_policy == "deterministic_fallback",
                    }
        for group in args.groups:
            progress += 1
            if (qid, group) in done:
                print(f"[Experiment] Skip {progress}/{total} id={qid} group={group}")
                continue
            print(f"[Experiment] Run {progress}/{total} id={qid} group={group}")
            started = time.time()
            result: dict[str, Any] = {
                "id": qid,
                "group": group,
                "query": query,
                "case_source": query_item.get("source"),
                "model": args.model,
                "configured_model_pool": parse_model_pool(args.model, args.models),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "retrieval_sha256": retrieval_sha,
                "implementation_sha256": implementation_hash(),
                "retrieved_card_ids": [card.get("card_id") for card in cards],
            }
            fatal_error: Exception | None = None
            try:
                config = GROUPS[group]
                if config["mode"] == "direct":
                    generated = run_direct(query=query, cards=cards, config=config, args=args)
                else:
                    if observation_record and observation_record.get("error") and args.observation_failure_policy == "error":
                        error = observation_record["error"]
                        raise RuntimeError(
                            f"Clinical observation extraction failed: {error.get('type')}: {error.get('message')}"
                        )
                    generated = run_skill(
                        query=query,
                        hits=hits,
                        cards=cards,
                        config=config,
                        args=args,
                        oracle_item=oracle_items.get(qid),
                        observation_record=observation_record,
                        retrieval_item=retrieval_item,
                    )
                result.update(generated)
            except Exception as exc:
                result["error"] = {"type": type(exc).__name__, "message": str(exc)}
                print(f"[Experiment][ERROR] id={qid} group={group}: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    fatal_error = exc
            result["elapsed_seconds"] = round(time.time() - started, 3)
            results.append(result)
            if args.save_each or fatal_error is not None:
                save_json(args.output, results)
            if fatal_error is not None:
                raise fatal_error

    save_json(args.output, results)
    artifact_dir = write_run_artifacts(
        args=args,
        results=results,
        query_count=len(queries),
        retrieval_sha=retrieval_sha,
    )
    if args.strict_smoke:
        if "medsafe_full" not in args.groups:
            raise ValueError("--strict-smoke requires the medsafe_full group")
        validate_end_to_end_smoke(
            results=results,
            expected_ids={item.get("id") for item in queries},
            artifact_dir=artifact_dir,
        )
    print(json.dumps({
        "query_count": len(queries),
        "groups": args.groups,
        "saved": len(results),
        "output": str(args.output),
        "artifact_dir": str(artifact_dir),
        "strict_smoke_passed": bool(args.strict_smoke),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
