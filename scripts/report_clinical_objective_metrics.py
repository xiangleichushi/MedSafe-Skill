#!/usr/bin/env python3
"""Compute reproducible, non-human metrics for the 51-case clinical run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATES = {
    "ready_for_clinician_confirmation",
    "human_review_required",
    "awaiting_external_evidence",
    "abstained",
}
ACTION_SEVERITY = {
    "continue": 0,
    "caution": 1,
    "undetermined": 2,
    "avoid": 3,
}
REVIEW_SEVERITY = {
    "standard_clinician_confirmation": 0,
    "pharmacist_review": 1,
    "specialist_review": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("medsafe_skill/output/clinical_cases_full"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def percent(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 1)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def numeric_summary(values: Iterable[int | float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {
            "n": 0,
            "total": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    total = sum(data)
    return {
        "n": len(data),
        "total": round(total, 3),
        "mean": round(statistics.mean(data), 3),
        "median": round(statistics.median(data), 3),
        "q1": round(percentile(data, 0.25) or 0, 3),
        "q3": round(percentile(data, 0.75) or 0, 3),
        "min": round(min(data), 3),
        "max": round(max(data), 3),
    }


def field_value(observation: dict[str, Any], field: str) -> Any:
    return (
        ((observation.get("validated_observations") or {}).get("fields") or {})
        .get(field, {})
        .get("value")
    )


def numbered_section_count(answer: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+[.、]\s*", answer))


def clinical_text_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text.lower())


def rouge_l(reference: str, candidate: str) -> tuple[float, float, float]:
    reference_tokens = clinical_text_tokens(reference)
    candidate_tokens = clinical_text_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0
    previous = [0] * (len(candidate_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, candidate_token in enumerate(candidate_tokens, start=1):
            if reference_token == candidate_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    recall = lcs / len(reference_tokens)
    precision = lcs / len(candidate_tokens)
    f1 = 0.0 if not precision + recall else 2 * precision * recall / (
        precision + recall
    )
    return precision, recall, f1


def token_totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prompt = []
    completion = []
    total = []
    for record in records:
        usage = record.get("usage") or {}
        prompt.append(int(usage.get("prompt_tokens") or 0))
        completion.append(int(usage.get("completion_tokens") or 0))
        total.append(int(usage.get("total_tokens") or 0))
    return {
        "prompt_tokens": numeric_summary(prompt),
        "completion_tokens": numeric_summary(completion),
        "total_tokens": numeric_summary(total),
    }


def trace_is_contiguous(transitions: list[dict[str, Any]]) -> bool:
    if not transitions:
        return False
    if [item.get("sequence") for item in transitions] != list(
        range(1, len(transitions) + 1)
    ):
        return False
    return all(
        transitions[index - 1].get("to") == transitions[index].get("from")
        for index in range(1, len(transitions))
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calculate(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = load_json(run_dir / "clinical_cases_public.json", [])
    import_summary = load_json(run_dir / "clinical_cases_import_summary.json", {})
    observation_cache = load_json(run_dir / "clinical_observations_cache.json", {})
    retrieval = load_json(run_dir / "hrc_retrieval_with_cards.json", [])
    stage1 = load_json(run_dir / "clinical_case_answers.json", [])
    final = load_json(run_dir / "clinical_case_final_answers_v6.json", None)
    if final is None:
        final = load_json(run_dir / "clinical_case_final_answers_v5.json", None)
    if final is None:
        final = load_json(run_dir / "clinical_case_final_answers_v4.json", [])
    references = load_json(
        run_dir / "clinical_cases_reference_held_out.json",
        [],
    )
    sessions = {}
    for path in run_dir.glob("interactive_evidence_case_*.json"):
        session = load_json(path, {})
        sessions[str(session.get("case_id"))] = session

    observations = list(observation_cache.values())
    retrieval_by_id = {str(item.get("id")): item for item in retrieval}
    stage1_by_id = {str(item.get("id")): item for item in stage1}
    final_by_id = {str(item.get("id")): item for item in final}
    observation_by_id = {
        str(item.get("case_id")): item for item in observations
    }

    contexts = [field_value(item, "reproductive_context") for item in observations]
    intents = [field_value(item, "consultation_intent") for item in observations]
    query_lengths = [len(str(item.get("query") or "")) for item in cases]
    context_agreement = []
    context_mismatch_ids = []
    for case in cases:
        case_id = str(case.get("id"))
        observed_context = field_value(
            observation_by_id.get(case_id, {}),
            "reproductive_context",
        ) or "unspecified"
        retrieved_context = (
            (
                retrieval_by_id.get(case_id, {}).get("medsafe_skill") or {}
            ).get("query_context")
            or "unspecified"
        )
        agrees = observed_context == retrieved_context
        context_agreement.append(agrees)
        if not agrees:
            context_mismatch_ids.append(case.get("id"))

    structural_candidates = []
    accepted_fields = []
    observation_verdicts = []
    risk_verdicts = []
    for observation in observations:
        structural_fields = (
            (observation.get("structurally_validated_observations") or {})
            .get("fields") or {}
        )
        validated_fields = (
            (observation.get("validated_observations") or {}).get("fields") or {}
        )
        structural_candidates.append(sum(
            item.get("value") is not None for item in structural_fields.values()
        ))
        accepted_fields.append(sum(
            item.get("value") is not None for item in validated_fields.values()
        ))
        raw_verifier = (
            (observation.get("semantic_verifier") or {}).get("raw_payload") or {}
        )
        observation_verdicts.extend(
            item.get("verdict") for item in raw_verifier.get("field_verdicts") or []
        )
        risk_verdicts.extend(
            item.get("verdict")
            for item in raw_verifier.get("risk_factor_verdicts") or []
        )

    quality_statuses = []
    critic_rounds = []
    final_candidate_counts = []
    attached_card_counts = []
    unmatched_counts = []
    unique_retrieved_cards = set()
    unique_retrieved_drugs = set()
    retrieval_contexts = []
    retrieval_card_actions = []
    source_action_pairs = []
    source_classification_labels = []
    source_classification_profiles = []
    source_classification_signals = []
    pathway_scopes = []
    systemic_absorption_statuses = []
    appraisal_items = []
    requested_facts_supported = []
    unsupported_facts = []
    for item in retrieval:
        quality = item.get("retrieval_quality") or {}
        quality_statuses.append(quality.get("status"))
        critic_rounds.append(int(quality.get("critic_rounds") or 0))
        final_candidate_counts.append(len(item.get("final_reranked_top_20") or []))
        skill = item.get("medsafe_skill") or {}
        cards = skill.get("retrieved_cards") or []
        attached_card_counts.append(len(cards))
        unmatched_counts.append(len(skill.get("unmatched_drugs") or []))
        retrieval_contexts.append(skill.get("query_context"))
        for card in cards:
            card_payload = card.get("card") or card
            if card_payload.get("card_id"):
                unique_retrieved_cards.add(str(card_payload.get("card_id")))
            if card.get("drug"):
                unique_retrieved_drugs.add(str(card.get("drug")))
        workflow = skill.get("clinical_workflow") or {}
        for decision in workflow.get("card_decisions") or []:
            retrieval_card_actions.append(decision.get("action_label"))
            for pathway in decision.get("pathway_results") or []:
                source_action_pairs.append((
                    pathway.get("source_action"),
                    pathway.get("action"),
                ))
                pathway_scopes.append(pathway.get("pathway"))
                classification = pathway.get("source_classification") or {}
                source_classification_labels.append(classification.get("raw_label"))
                source_classification_profiles.append(
                    classification.get("profile_id")
                )
                source_classification_signals.append(
                    classification.get("source_signal")
                )
                systemic_absorption_statuses.append(
                    (pathway.get("systemic_absorption") or {}).get("status")
                )

    stage1_actions = []
    review_levels = []
    release_statuses = []
    final_states = []
    missing_slots = []
    missing_severities = []
    unsupported_decision_facts = []
    evidence_request_counts = []
    state_transition_counts = []
    decision_node_counts = []
    trace_contiguous = []
    legal_terminal = []
    claim_counts = []
    supported_claim_counts = []
    unsupported_claim_counts = []
    card_bound_claims = []
    page_traceable_claims = []
    field_traceable_claims = []
    claim_types = []
    card_action_ceiling_violations = 0
    review_under_escalations = 0
    deterministic_audit_cases = 0
    l1_l5_not_inferred_cases = 0
    decision_tree_node_ids = []
    for record in stage1:
        packet = record.get("decision_packet") or {}
        trace = record.get("reasoning_trace") or {}
        stage1_actions.append(packet.get("action"))
        review_levels.append(packet.get("review_level"))
        release_statuses.append(packet.get("release_status"))
        final_states.append(trace.get("final_state"))
        for item in packet.get("missing_information") or []:
            missing_slots.append(item.get("slot"))
            missing_severities.append(item.get("severity"))
        unsupported_decision_facts.extend(packet.get("unsupported_facts") or [])
        evidence_request_counts.append(len(packet.get("evidence_requests") or []))
        transitions = trace.get("state_transitions") or []
        state_transition_counts.append(len(transitions))
        decision_node_counts.append(len(trace.get("decision_tree_nodes") or []))
        decision_tree_node_ids.extend(
            item.get("node") for item in trace.get("decision_tree_nodes") or []
        )
        trace_contiguous.append(trace_is_contiguous(transitions))
        legal_terminal.append(trace.get("final_state") in TERMINAL_STATES)

        verification = record.get("claim_verification") or {}
        claims = verification.get("claims") or []
        claim_counts.append(int(verification.get("claim_count") or 0))
        supported_claim_counts.append(int(verification.get("supported_count") or 0))
        unsupported_claim_counts.append(int(verification.get("unsupported_count") or 0))
        for claim in claims:
            claim_types.append(claim.get("claim_type"))
            if claim.get("card_id"):
                card_bound_claims.append(claim)
                if claim.get("page_doc") and claim.get("page_pdf"):
                    page_traceable_claims.append(claim)
                if claim.get("field_path"):
                    field_traceable_claims.append(claim)

        plan_workflow = ((record.get("plan") or {}).get("clinical_workflow") or {})
        decisions = plan_workflow.get("card_decisions") or []
        bounded_actions = [
            decision.get("action_label") for decision in decisions
            if decision.get("action_label") in ACTION_SEVERITY
        ]
        final_action = packet.get("action")
        if "avoid" in bounded_actions and final_action in {"continue", "caution"}:
            card_action_ceiling_violations += 1
        card_reviews = [
            REVIEW_SEVERITY.get(decision.get("review_level"), -1)
            for decision in decisions
        ]
        if card_reviews and REVIEW_SEVERITY.get(packet.get("review_level"), -1) < max(
            card_reviews
        ):
            review_under_escalations += 1
        audit = plan_workflow.get("audit") or {}
        if (
            audit.get("deterministic_control_flow") is True
            and audit.get("llm_used_for_state_transitions") is False
            and audit.get("unknown_values_preserved") is True
        ):
            deterministic_audit_cases += 1
        if audit.get("l1_l5_inferred") is False:
            l1_l5_not_inferred_cases += 1

        evidence = (record.get("plan") or {}).get("evidence_sufficiency") or {}
        requested_facts_supported.append(
            evidence.get("all_requested_facts_supported") is True
        )
        unsupported_facts.extend(evidence.get("unsupported_facts") or [])
        appraisal_items.extend(
            (trace.get("evidence_appraisal") or {}).get("per_card") or []
        )

    session_value_fields = []
    derived_estimate_counts = []
    requested_session_fields = []
    for session in sessions.values():
        values = session.get("values") or {}
        session_value_fields.extend(values.keys())
        derived_estimate_counts.append(len(session.get("derived_estimates") or []))
        requested_session_fields.extend(
            item.get("field")
            for item in (session.get("request") or {}).get("required_fields") or []
        )

    answer_lengths = [len(str(record.get("answer") or "")) for record in final]
    numbered_sections = [
        numbered_section_count(str(record.get("answer") or "")) for record in final
    ]
    final_revisions = [int(record.get("revision_count") or 0) for record in final]
    support_classes = []
    prior_fields = []
    prior_claim_counts = []
    final_models = []
    route_attempts = []
    fallback_routes = 0
    for record in final:
        support_classes.extend(
            ref.get("support_class") or ref.get("support_type")
            for ref in record.get("support_refs") or []
        )
        priors = record.get("model_prior_claims") or []
        prior_claim_counts.append(len(priors))
        prior_fields.extend(item.get("target_missing_fact") for item in priors)
        final_models.append(record.get("model"))
        for route in record.get("model_routes") or []:
            fallback_routes += int(bool(route.get("fallback_used")))
            route_attempts.extend(route.get("attempts") or [])

    unsafe_downgrades = sum(
        1
        for source, bounded in source_action_pairs
        if source in ACTION_SEVERITY
        and bounded in ACTION_SEVERITY
        and ACTION_SEVERITY[bounded] < ACTION_SEVERITY[source]
    )
    traceable_appraisals = sum(
        item.get("traceable_to_source_page") is True for item in appraisal_items
    )
    appraisal_coverages = [
        float(item.get("requested_fact_coverage"))
        for item in appraisal_items
        if isinstance(item.get("requested_fact_coverage"), (int, float))
    ]

    final_validation_passes = sum(
        (record.get("validation") or {}).get("passed") is True for record in final
    )
    internal_leakage_cases = sum(
        bool((record.get("safety_audit") or {}).get(
            "answer_exposes_internal_provenance_labels"
        ))
        for record in final
    )
    final_card_cases = sum(
        bool((record.get("safety_audit") or {}).get("evidence_card_used"))
        for record in final
    )
    final_prior_cases = sum(
        bool((record.get("safety_audit") or {}).get("model_prior_used"))
        for record in final
    )
    final_supplement_cases = sum(
        bool((record.get("safety_audit") or {}).get("supplemental_values_used"))
        for record in final
    )
    final_estimate_cases = sum(
        bool((record.get("safety_audit") or {}).get("derived_estimates_used"))
        for record in final
    )
    discarded_prior_count = sum(
        len(record.get("discarded_model_prior_claims") or []) for record in final
    )
    low_confidence_prior_claims = sum(
        item.get("confidence") == "low"
        for record in final
        for item in record.get("model_prior_claims") or []
    )
    fixed_disclosure_prior_claims = sum(
        item.get("risk_disclosure")
        == "未经外部证据核验，不作为用药决策依据"
        for record in final
        for item in record.get("model_prior_claims") or []
    )
    verbatim_prior_claims = sum(
        str(item.get("text") or "") in str(record.get("answer") or "")
        for record in final
        for item in record.get("model_prior_claims") or []
    )

    unsupported_claim_total = 0
    unsupported_claim_used = 0
    supported_claim_used = 0
    for record in stage1:
        final_record = final_by_id.get(str(record.get("id")), {})
        used_ids = set(
            (final_record.get("safety_audit") or {}).get(
                "structured_base_claim_ids"
            ) or []
        )
        for claim in (record.get("claim_verification") or {}).get("claims") or []:
            if claim.get("supported") is True:
                supported_claim_used += int(claim.get("claim_id") in used_ids)
            else:
                unsupported_claim_total += 1
                unsupported_claim_used += int(claim.get("claim_id") in used_ids)

    action_pair_distribution = distribution(
        f"{source}->{bounded}" for source, bounded in source_action_pairs
    )
    conservative_escalations = sum(
        source in ACTION_SEVERITY
        and bounded in ACTION_SEVERITY
        and ACTION_SEVERITY[bounded] > ACTION_SEVERITY[source]
        for source, bounded in source_action_pairs
    )
    unchanged_bounded_actions = sum(
        source == bounded for source, bounded in source_action_pairs
    )

    reference_by_id = {str(item.get("id")): item for item in references}
    rouge_precision = []
    rouge_recall = []
    rouge_f1 = []
    exact_reference_matches = 0
    reference_statuses = []
    for record in final:
        reference_record = reference_by_id.get(str(record.get("id")))
        if not reference_record:
            continue
        reference = str(reference_record.get("reference_answer") or "")
        answer = str(record.get("answer") or "")
        if not reference:
            continue
        precision, recall, f1 = rouge_l(reference, answer)
        rouge_precision.append(precision)
        rouge_recall.append(recall)
        rouge_f1.append(f1)
        exact_reference_matches += int(reference.strip() == answer.strip())
        reference_statuses.append(reference_record.get("reference_status"))

    metrics = {
        "metric_scope": {
            "description": (
                "Objective internal-process, traceability, coverage, safety-policy, "
                "and resource metrics. These are not clinical accuracy estimates."
            ),
            "run_dir": str(run_dir),
            "timing_included": False,
            "human_scores_included": False,
        },
        "cohort": {
            "case_count": len(cases),
            "unique_case_id_count": len({str(item.get("id")) for item in cases}),
            "reference_answer_count": import_summary.get("reference_answer_count"),
            "reference_answers_excluded_from_model_input": import_summary.get(
                "reference_answers_excluded_from_model_input"
            ),
            "query_length_characters": numeric_summary(query_lengths),
            "reproductive_context_distribution": distribution(contexts),
            "consultation_intent_distribution": distribution(intents),
            "cases_with_accepted_risk_factors": sum(
                bool(
                    (item.get("validated_observations") or {}).get("risk_factors")
                )
                for item in observations
            ),
            "accepted_risk_factor_distribution": distribution(
                factor.get("factor")
                for item in observations
                for factor in (
                    (item.get("validated_observations") or {}).get("risk_factors")
                    or []
                )
            ),
        },
        "observation_extraction": {
            "cache_record_count": len(observations),
            "case_id_coverage_count": len({
                str(item.get("case_id")) for item in observations
            }),
            "structural_candidate_fields": numeric_summary(structural_candidates),
            "semantically_accepted_fields": numeric_summary(accepted_fields),
            "semantic_retention_rate_percent": percent(
                sum(accepted_fields), sum(structural_candidates)
            ),
            "field_verdict_distribution": distribution(observation_verdicts),
            "risk_factor_verdict_distribution": distribution(risk_verdicts),
            "zero_structural_error_cases": sum(
                (
                    (
                        (item.get("validated_observations") or {})
                        .get("validation") or {}
                    ).get("structural_error_count")
                    or 0
                ) == 0
                for item in observations
            ),
            "zero_semantic_error_cases": sum(
                (
                    (
                        (item.get("validated_observations") or {})
                        .get("validation") or {}
                    ).get("semantic_error_count")
                    or 0
                ) == 0
                for item in observations
            ),
            "all_non_null_spans_validated_cases": sum(
                (
                    (
                        (item.get("validated_observations") or {})
                        .get("validation") or {}
                    ).get("all_non_null_values_span_validated")
                    is True
                )
                for item in observations
            ),
            "all_accepted_values_semantically_verified_cases": sum(
                (
                    (
                        (item.get("validated_observations") or {})
                        .get("validation") or {}
                    ).get("all_accepted_values_semantically_verified")
                    is True
                )
                for item in observations
            ),
            "stage1_observation_cache_hit_cases": sum(
                bool((item.get("clinical_observation_extraction") or {}).get(
                    "cache_hit"
                ))
                for item in stage1
            ),
            "recorded_token_usage": token_totals(observations),
        },
        "retrieval_and_evidence": {
            "retrieval_record_count": len(retrieval),
            "retrieval_quality_status_distribution": distribution(quality_statuses),
            "critic_round_distribution": distribution(critic_rounds),
            "cases_with_retrieval_retry": sum(value > 1 for value in critic_rounds),
            "final_reranked_candidate_count": numeric_summary(final_candidate_counts),
            "attached_evidence_card_count": numeric_summary(attached_card_counts),
            "unmatched_drug_count": numeric_summary(unmatched_counts),
            "cases_with_unmatched_drugs": sum(value > 0 for value in unmatched_counts),
            "unique_retrieved_card_count": len(unique_retrieved_cards),
            "unique_retrieved_drug_count": len(unique_retrieved_drugs),
            "retrieval_context_distribution": distribution(retrieval_contexts),
            "observation_retrieval_context_exact_agreement_count": sum(
                context_agreement
            ),
            "observation_retrieval_context_exact_agreement_percent": percent(
                sum(context_agreement), len(context_agreement)
            ),
            "observation_retrieval_context_mismatch_ids": context_mismatch_ids,
            "card_decision_action_distribution": distribution(
                retrieval_card_actions
            ),
            "pathway_scope_distribution": distribution(pathway_scopes),
            "source_classification_raw_label_distribution": distribution(
                source_classification_labels
            ),
            "source_classification_profile_distribution": distribution(
                source_classification_profiles
            ),
            "source_classification_signal_distribution": distribution(
                source_classification_signals
            ),
            "systemic_absorption_status_distribution": distribution(
                systemic_absorption_statuses
            ),
            "cases_all_requested_facts_supported": sum(requested_facts_supported),
            "cases_with_unsupported_requested_facts": sum(
                not value for value in requested_facts_supported
            ),
            "unsupported_fact_distribution": distribution(unsupported_facts),
            "appraised_card_count": len(appraisal_items),
            "source_page_traceable_appraisal_count": traceable_appraisals,
            "source_page_traceability_percent": percent(
                traceable_appraisals, len(appraisal_items)
            ),
            "requested_fact_coverage_per_card": numeric_summary(
                appraisal_coverages
            ),
        },
        "decision_and_governance": {
            "action_distribution": distribution(stage1_actions),
            "review_level_distribution": distribution(review_levels),
            "release_status_distribution": distribution(release_statuses),
            "final_state_distribution": distribution(final_states),
            "legal_terminal_state_cases": sum(legal_terminal),
            "state_trace_contiguous_cases": sum(trace_contiguous),
            "state_transition_count": numeric_summary(state_transition_counts),
            "decision_tree_node_count": numeric_summary(decision_node_counts),
            "decision_tree_node_execution_distribution": distribution(
                decision_tree_node_ids
            ),
            "missing_information_item_count": len(missing_slots),
            "missing_information_slot_distribution": distribution(missing_slots),
            "missing_information_severity_distribution": distribution(
                missing_severities
            ),
            "unsupported_decision_fact_distribution": distribution(
                unsupported_decision_facts
            ),
            "cases_with_external_evidence_request": sum(
                value > 0 for value in evidence_request_counts
            ),
            "external_evidence_request_count": sum(evidence_request_counts),
            "deterministic_control_audit_pass_cases": deterministic_audit_cases,
            "cases_confirming_l1_l5_not_inferred": l1_l5_not_inferred_cases,
            "pathway_source_to_bounded_action_comparison_count": len(
                source_action_pairs
            ),
            "source_to_bounded_action_transition_distribution": (
                action_pair_distribution
            ),
            "conservative_action_escalation_count": conservative_escalations,
            "unchanged_bounded_action_count": unchanged_bounded_actions,
            "unsafe_action_downgrade_count": unsafe_downgrades,
            "case_level_avoid_to_continue_or_caution_violation_count": (
                card_action_ceiling_violations
            ),
            "review_under_escalation_count": review_under_escalations,
        },
        "claim_verification": {
            "stage1_case_count": len(stage1),
            "all_claims_supported_case_count": sum(
                (item.get("claim_verification") or {}).get("all_supported") is True
                for item in stage1
            ),
            "claim_count": numeric_summary(claim_counts),
            "supported_claim_count": sum(supported_claim_counts),
            "unsupported_claim_count": sum(unsupported_claim_counts),
            "claim_support_rate_percent": percent(
                sum(supported_claim_counts), sum(claim_counts)
            ),
            "claim_type_distribution": distribution(claim_types),
            "card_bound_claim_count": len(card_bound_claims),
            "card_bound_claim_rate_percent": percent(
                len(card_bound_claims), sum(claim_counts)
            ),
            "card_bound_claims_with_doc_and_pdf_pages": len(
                page_traceable_claims
            ),
            "card_bound_page_traceability_percent": percent(
                len(page_traceable_claims), len(card_bound_claims)
            ),
            "card_bound_claims_with_field_path": len(field_traceable_claims),
            "card_bound_field_path_traceability_percent": percent(
                len(field_traceable_claims), len(card_bound_claims)
            ),
            "unsupported_claims_available_for_isolation_check": (
                unsupported_claim_total
            ),
            "unsupported_claims_used_in_final_answer": unsupported_claim_used,
            "unsupported_claim_isolation_rate_percent": percent(
                unsupported_claim_total - unsupported_claim_used,
                unsupported_claim_total,
            ),
            "supported_stage1_claims_reused_in_final_answer": (
                supported_claim_used
            ),
            "stage1_revision_distribution": distribution(
                int(item.get("revision_count") or 0) for item in stage1
            ),
            "stage1_json_repair_distribution": distribution(
                int(item.get("json_format_repair_count") or 0) for item in stage1
            ),
            "recorded_token_usage": token_totals(stage1),
        },
        "interactive_supplementation": {
            "session_file_count": len(sessions),
            "case_without_session_file_count": len(cases) - len(sessions),
            "session_status_distribution": distribution(
                item.get("session_status") for item in sessions.values()
            ),
            "requested_field_count": len(requested_session_fields),
            "requested_field_distribution": distribution(requested_session_fields),
            "cases_with_user_supplied_values": sum(
                bool(item.get("values")) for item in sessions.values()
            ),
            "user_supplied_value_count": len(session_value_fields),
            "user_supplied_field_distribution": distribution(session_value_fields),
            "operator_input_policy": "accepted_as_confirmed_workflow_input",
            "cases_with_derived_estimates": sum(
                value > 0 for value in derived_estimate_counts
            ),
            "derived_estimate_count": sum(derived_estimate_counts),
        },
        "final_answer_generation": {
            "record_count": len(final),
            "unique_case_id_count": len({str(item.get("id")) for item in final}),
            "error_count": sum(bool(item.get("error")) for item in final),
            "validation_pass_count": final_validation_passes,
            "validation_pass_rate_percent": percent(
                final_validation_passes, len(final)
            ),
            "generation_mode_distribution": distribution(
                item.get("generation_mode") for item in final
            ),
            "final_model_distribution": distribution(final_models),
            "answer_length_characters": numeric_summary(answer_lengths),
            "numbered_section_count": numeric_summary(numbered_sections),
            "numbered_format_case_count": sum(value > 0 for value in numbered_sections),
            "internal_provenance_leakage_case_count": internal_leakage_cases,
            "direct_evidence_card_case_count": final_card_cases,
            "direct_evidence_card_case_rate_percent": percent(
                final_card_cases, len(final)
            ),
            "model_prior_case_count": final_prior_cases,
            "model_prior_case_rate_percent": percent(final_prior_cases, len(final)),
            "model_prior_claim_count": sum(prior_claim_counts),
            "low_confidence_model_prior_claim_count": (
                low_confidence_prior_claims
            ),
            "fixed_risk_disclosure_model_prior_claim_count": (
                fixed_disclosure_prior_claims
            ),
            "verbatim_bound_model_prior_claim_count": verbatim_prior_claims,
            "model_prior_target_field_distribution": distribution(prior_fields),
            "discarded_model_prior_claim_count": discarded_prior_count,
            "supplemental_value_used_case_count": final_supplement_cases,
            "derived_estimate_used_case_count": final_estimate_cases,
            "support_reference_class_distribution": distribution(support_classes),
            "final_revision_distribution": distribution(final_revisions),
            "cases_requiring_final_revision": sum(value > 0 for value in final_revisions),
            "api_response_count": sum(
                len(item.get("response_ids") or []) for item in final
            ),
            "route_attempt_outcome_distribution": distribution(
                item.get("outcome") for item in route_attempts
            ),
            "fallback_route_count": fallback_routes,
            "recorded_token_usage": token_totals(final),
            "audit_version_distribution": distribution(
                (item.get("safety_audit") or {}).get("audit_version")
                for item in final
            ),
            "trace_version_distribution": distribution(
                (item.get("reasoning_trace_snapshot") or {}).get("trace_version")
                for item in final
            ),
            "prompt_sha256_complete_case_count": sum(
                bool(item.get("prompt_sha256")) for item in final
            ),
            "base_record_sha256_complete_case_count": sum(
                bool(item.get("base_record_sha256")) for item in final
            ),
            "session_sha256_complete_case_count": sum(
                bool(item.get("session_sha256")) for item in final
            ),
        },
        "historical_reference_similarity": {
            "reference_count": len(rouge_f1),
            "reference_status_distribution": distribution(reference_statuses),
            "exact_match_count": exact_reference_matches,
            "character_token_rouge_l_precision": numeric_summary(
                rouge_precision
            ),
            "character_token_rouge_l_recall": numeric_summary(rouge_recall),
            "character_token_rouge_l_f1": numeric_summary(rouge_f1),
            "interpretation": (
                "Descriptive similarity to historical workbook answers only. "
                "The references are explicitly not adjudicated gold answers."
            ),
        },
        "interpretation_limits": {
            "not_clinical_accuracy": (
                "A 100% deterministic validation rate means schema, provenance, "
                "numeric, and policy checks passed; it does not establish medical "
                "correctness."
            ),
            "retrieval_ranking_metrics_unavailable_without_gold": [
                "Recall@k",
                "Precision@k",
                "MRR",
                "nDCG@k",
            ],
            "clinical_metrics_unavailable_without_expert_gold": [
                "gold action accuracy",
                "macro-F1",
                "major safety error rate",
                "contraindication miss rate",
                "review routing accuracy",
                "decision-path exact match",
            ],
            "reference_answer_warning": (
                "The held-out workbook answers are marked source_workbook_text_not_gold. "
                "BLEU, ROUGE, or BERTScore against them may be secondary descriptive "
                "similarity only, not clinical correctness."
            ),
        },
    }

    case_rows = []
    for case in cases:
        case_id = str(case.get("id"))
        observation = observation_by_id.get(case_id, {})
        retrieved = retrieval_by_id.get(case_id, {})
        first = stage1_by_id.get(case_id, {})
        last = final_by_id.get(case_id, {})
        session = sessions.get(case_id, {})
        packet = first.get("decision_packet") or {}
        trace = first.get("reasoning_trace") or {}
        quality = retrieved.get("retrieval_quality") or {}
        skill = retrieved.get("medsafe_skill") or {}
        verification = first.get("claim_verification") or {}
        audit = last.get("safety_audit") or {}
        case_rows.append({
            "id": case.get("id"),
            "reproductive_context": field_value(observation, "reproductive_context"),
            "consultation_intent": field_value(observation, "consultation_intent"),
            "accepted_observation_field_count": (
                ((observation.get("validated_observations") or {}).get(
                    "validation"
                ) or {}).get("valid_field_count")
            ),
            "accepted_risk_factor_count": len(
                (observation.get("validated_observations") or {}).get(
                    "risk_factors"
                ) or []
            ),
            "retrieval_status": quality.get("status"),
            "critic_rounds": quality.get("critic_rounds"),
            "retrieved_card_count": skill.get("retrieved_card_count"),
            "unmatched_drug_count": len(skill.get("unmatched_drugs") or []),
            "action": packet.get("action"),
            "review_level": packet.get("review_level"),
            "final_state": trace.get("final_state"),
            "missing_information_count": len(packet.get("missing_information") or []),
            "external_evidence_request_count": len(
                packet.get("evidence_requests") or []
            ),
            "stage1_claim_count": verification.get("claim_count"),
            "stage1_supported_claim_count": verification.get("supported_count"),
            "stage1_revision_count": first.get("revision_count"),
            "session_value_count": len(session.get("values") or {}),
            "derived_estimate_count": len(session.get("derived_estimates") or []),
            "final_validation_passed": (
                (last.get("validation") or {}).get("passed")
            ),
            "final_evidence_card_used": audit.get("evidence_card_used"),
            "final_model_prior_used": audit.get("model_prior_used"),
            "final_model_prior_claim_count": len(
                last.get("model_prior_claims") or []
            ),
            "final_revision_count": last.get("revision_count"),
            "answer_character_count": len(str(last.get("answer") or "")),
            "answer_numbered_section_count": numbered_section_count(
                str(last.get("answer") or "")
            ),
            "final_total_tokens": (last.get("usage") or {}).get("total_tokens"),
            "final_model": last.get("model"),
        })
    return metrics, case_rows


def markdown_report(metrics: dict[str, Any]) -> str:
    cohort = metrics["cohort"]
    observation = metrics["observation_extraction"]
    retrieval = metrics["retrieval_and_evidence"]
    decision = metrics["decision_and_governance"]
    claims = metrics["claim_verification"]
    supplement = metrics["interactive_supplementation"]
    final = metrics["final_answer_generation"]
    similarity = metrics["historical_reference_similarity"]

    def row(metric: str, value: Any, denominator: Any = None) -> str:
        def display(item: Any) -> Any:
            if isinstance(item, float) and item.is_integer():
                return int(item)
            return item

        value = display(value)
        denominator = display(denominator)
        if denominator is None:
            return f"| {metric} | {value} |"
        return (
            f"| {metric} | {value}/{denominator} "
            f"({percent(value, denominator)}%) |"
        )

    lines = [
        "# MedSafe-Skill objective metrics",
        "",
        "> These are objective process, traceability, coverage, safety-policy, "
        "and resource metrics. They must not be described as clinical accuracy.",
        "",
        "## Cohort and pipeline completion",
        "",
        "| Metric | Result |",
        "|---|---:|",
        row("Real-world consultation cases", cohort["case_count"]),
        row(
            "Semantically verified observation records",
            observation["all_accepted_values_semantically_verified_cases"],
            cohort["case_count"],
        ),
        row(
            "Automated HRC Critic gate passed",
            retrieval["retrieval_quality_status_distribution"].get("passed", 0),
            cohort["case_count"],
        ),
        row(
            "Final answers passing deterministic validation",
            final["validation_pass_count"],
            final["record_count"],
        ),
        row(
            "Complete structured reasoning traces",
            decision["legal_terminal_state_cases"],
            cohort["case_count"],
        ),
        "",
        "## Evidence and claim traceability",
        "",
        "| Metric | Result |",
        "|---|---:|",
        row(
            "Cases with a directly cited Evidence Card in the final answer",
            final["direct_evidence_card_case_count"],
            final["record_count"],
        ),
        row(
            "Appraised cards traceable to source pages",
            retrieval["source_page_traceable_appraisal_count"],
            retrieval["appraised_card_count"],
        ),
        row(
            "Stage-1 claims passing deterministic verification",
            claims["supported_claim_count"],
            claims["claim_count"]["total"],
        ),
        row(
            "Card-bound claims with document and PDF pages",
            claims["card_bound_claims_with_doc_and_pdf_pages"],
            claims["card_bound_claim_count"],
        ),
        row(
            "Card-bound claims with a field path",
            claims["card_bound_claims_with_field_path"],
            claims["card_bound_claim_count"],
        ),
        row(
            "Rejected Stage-1 claims isolated from final answers",
            (
                claims["unsupported_claims_available_for_isolation_check"]
                - claims["unsupported_claims_used_in_final_answer"]
            ),
            claims["unsupported_claims_available_for_isolation_check"],
        ),
        row(
            "Cases with all requested facts supported by retrieved cards",
            retrieval["cases_all_requested_facts_supported"],
            cohort["case_count"],
        ),
        "",
        "## Safety and governance",
        "",
        "| Metric | Result |",
        "|---|---:|",
        row(
            "Cases with valid contiguous state-machine traces",
            decision["state_trace_contiguous_cases"],
            cohort["case_count"],
        ),
        row(
            "Cases passing deterministic-control audit",
            decision["deterministic_control_audit_pass_cases"],
            cohort["case_count"],
        ),
        row(
            "Unsafe source-to-bounded action downgrades",
            decision["unsafe_action_downgrade_count"],
            decision["pathway_source_to_bounded_action_comparison_count"],
        ),
        row(
            "Conservative source-to-bounded action escalations",
            decision["conservative_action_escalation_count"],
            decision["pathway_source_to_bounded_action_comparison_count"],
        ),
        row(
            "Case-level avoid-to-continue/caution violations",
            decision["case_level_avoid_to_continue_or_caution_violation_count"],
            cohort["case_count"],
        ),
        row(
            "Review-level under-escalations",
            decision["review_under_escalation_count"],
            cohort["case_count"],
        ),
        row(
            "Internal provenance labels leaked into clinical prose",
            final["internal_provenance_leakage_case_count"],
            final["record_count"],
        ),
        "",
        "## Evidence gaps and controlled fallback",
        "",
        "| Metric | Result |",
        "|---|---:|",
        row(
            "Cases requiring external evidence requests",
            decision["cases_with_external_evidence_request"],
            cohort["case_count"],
        ),
        row(
            "Cases with user-supplied supplemental values",
            supplement["cases_with_user_supplied_values"],
            cohort["case_count"],
        ),
        row(
            "Cases using programmatic derived estimates",
            final["derived_estimate_used_case_count"],
            cohort["case_count"],
        ),
        row(
            "Cases using disclosed low-confidence model priors",
            final["model_prior_case_count"],
            final["record_count"],
        ),
        row(
            "Model-prior claims carrying fixed risk disclosures",
            final["fixed_risk_disclosure_model_prior_claim_count"],
            final["model_prior_claim_count"],
        ),
        "",
        "## Output and resource use",
        "",
        "| Metric | Result |",
        "|---|---:|",
        row(
            "Cases requiring one final automatic revision",
            final["cases_requiring_final_revision"],
            final["record_count"],
        ),
        row("Final-answer API responses", final["api_response_count"]),
        row(
            "Final-answer recorded total tokens",
            int(final["recorded_token_usage"]["total_tokens"]["total"]),
        ),
        row(
            "Mean final-answer tokens per case",
            final["recorded_token_usage"]["total_tokens"]["mean"],
        ),
        row(
            "Median final-answer length, Chinese characters",
            final["answer_length_characters"]["median"],
        ),
        row(
            "Historical-answer character-token ROUGE-L F1, mean",
            similarity["character_token_rouge_l_f1"]["mean"],
        ),
        "",
        "## Distributions",
        "",
        f"- Reproductive context: `{json.dumps(cohort['reproductive_context_distribution'], ensure_ascii=False)}`",
        f"- Consultation intent: `{json.dumps(cohort['consultation_intent_distribution'], ensure_ascii=False)}`",
        f"- Final action: `{json.dumps(decision['action_distribution'], ensure_ascii=False)}`",
        f"- Review level: `{json.dumps(decision['review_level_distribution'], ensure_ascii=False)}`",
        f"- Final state: `{json.dumps(decision['final_state_distribution'], ensure_ascii=False)}`",
        f"- Source-native classification labels: `{json.dumps(retrieval['source_classification_raw_label_distribution'], ensure_ascii=False)}`",
        f"- Source-to-bounded action transitions: `{json.dumps(decision['source_to_bounded_action_transition_distribution'], ensure_ascii=False)}`",
        f"- Missing-information slots: `{json.dumps(decision['missing_information_slot_distribution'], ensure_ascii=False)}`",
        f"- Model-prior target fields: `{json.dumps(final['model_prior_target_field_distribution'], ensure_ascii=False)}`",
        "",
        "## Interpretation boundary",
        "",
        metrics["interpretation_limits"]["not_clinical_accuracy"],
        "",
        metrics["interpretation_limits"]["reference_answer_warning"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "objective_metrics"
    metrics, rows = calculate(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "clinical_objective_metrics.json", metrics)
    (output_dir / "clinical_objective_metrics.md").write_text(
        markdown_report(metrics),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "case_level_objective_metrics.csv").open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "case_count": metrics["cohort"]["case_count"],
        "final_validation_pass_count": (
            metrics["final_answer_generation"]["validation_pass_count"]
        ),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
