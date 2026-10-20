#!/usr/bin/env python
"""V7: scoped appraisal -> optional terminal supplementation -> consultation draft."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "medsafe_skill"
sys.path.insert(0, str(ROOT))

from medsafe_skill.core.consultation_v7 import (  # noqa: E402
    VERSION, allowed_answer_refs, build_packet, digest, make_sources, normalize_appraisal_structure, render_answer, require,
    validate_answer, validate_appraisal,
)
from medsafe_skill.scripts.load_card_store import EvidenceCardStore, normalize_text  # noqa: E402
from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    _response_error_summary, answer_text, call_chat_api, parse_model_pool,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def version_hash() -> str:
    files = [Path(__file__), SKILL / "core/consultation_v7.py",
             SKILL / "scripts/run_answer_experiment.py",
             SKILL / "prompts/consultation_appraisal_v7.txt", SKILL / "prompts/consultation_answer_v7.txt"]
    return digest({p.name: p.read_text(encoding="utf-8") for p in files})


def parse_json(raw: str) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        require(len(lines) >= 3 and lines[-1].strip() == "```", "incomplete JSON fence")
        value = "\n".join(lines[1:-1])
    result = json.loads(value)
    require(isinstance(result, dict), "JSON object required")
    return result


def repair_guidance(error: Exception, payload: dict, parsed: dict | None) -> str:
    message = str(error)
    if (not message.startswith("answer contract violations:") and
            ("quote is not a literal span" in message or "support must contain source spans" in message)):
        return ("support.quote只能复制sources[ref].text中的连续逐字原文，不能省略中间文字、插入...、"
                "拼接不同句子或改写标点。需要两处原文时用两个support对象，ref可以相同。"
                "Q的完整原文为：" + str(payload.get("query", "")) +
                "。未知且原文未提及的患者信息不得填入patient_facts；有决策意义时放入missing_patient，"
                "其余未知项可以不输出，不要为unknown编造support。逐项核对所有引文，而非只修第一项。")
    if "context must have a source-linked reproductive_context fact" in message:
        return ("context和patient_facts中name=reproductive_context的value必须使用同一枚举："
                "pregnancy/lactation/preconception/mixed/unknown。可能妊娠的咨询可采用pregnancy路径，"
                "但须另记pregnancy_confirmation=unconfirmed并引用原文，不能声称已确诊妊娠；"
                "reason、conditions和最终结论都必须保留这一不确定性。若同时询问妊娠和哺乳，两字段均用mixed，"
                "分别引用原文依据，不能只保留哺乳或只保留妊娠。其他情境同样不得改变病例事实。")
    if "conflict must cite" in message or "within-source conflict" in message:
        return ("不同来源的冲突列出两个真实ref；若同一卡片同一字段已描述研究结论不一致，refs只列该ref，"
                "并增加support数组，分别逐字引用该字段内两段不同观点，至少两条不同quote。"
                "不能为凑足两个ref编造来源、重复同一个ref或引用无关字段。若只是资料不足而非事实冲突，"
                "移至缺失信息，不把未知伪装成冲突。")
    if message.startswith("answer contract violations:"):
        decisions = (payload.get("decision_packet") or {}).get("decisions", [])
        contracts = [{"subquestion_id": d["id"], "answer_mode": d["answer_mode"],
                      "required_conditions": d["conditions"], "clinical_action": d["clinical_action"]}
                     for d in decisions]
        return ("上述错误是同一回复的独立问题，必须一次全部修复，不能只修第一条。子问题合约：" +
                json.dumps(contracts, ensure_ascii=False) +
                "。packet为general且conditions为空时，可用general解释缺口和下一步，不要编造条件；"
                "conditional须列出实际适用假设，已有packet条件须逐字保留。"
                "引用只从该子问题allowed_refs_by_subquestion选择，不能跨子问题借用M编号或无关卡片；"
                "无法用允许引用支撑的内容应删除或改为明确的缺口，不得仅换引用来掩盖。"
                "external的support.quote必须是sources[ref].text连续逐字片段，不能插入省略号、拼接、翻译或改写；"
                "需要非连续内容时使用多个support项。患者Q原文与药物知识分开，不得用Q证明药物疗效。"
                "不能为修复这些错误改变packet的clinical_action或新增未经记录的M条目。")
    if "irrelevant/inapplicable card cannot support this subquestion" in message:
        conflicts = []
        catalog = payload.get("card_catalog", {})
        sources = payload.get("sources", {})
        for sub in (parsed or {}).get("subquestions", []):
            assessed = {a.get("card_id"): a for a in sub.get("card_assessments", [])}
            for ref in sub.get("basis_refs", []):
                if sources.get(ref, {}).get("kind") != "evidence_card":
                    continue
                cid = ref.partition(":")[0]
                assessment = assessed.get(cid, {})
                if (assessment.get("role") in {None, "irrelevant"}
                        or assessment.get("applicability") == "not_applicable"):
                    conflicts.append({"subquestion_id": sub.get("id"), "ref": ref,
                                      "drug": catalog.get(cid, ""), "role": assessment.get("role"),
                                      "applicability": assessment.get("applicability"),
                                      "reason": assessment.get("reason", "missing card assessment")})
        return ("本次矛盾引用：" + json.dumps(conflicts, ensure_ascii=False) +
                "。basis_refs是该子问题的有效决策依据，不是所有候选卡片或所有检查过的资料。"
                "逐项依据原问题与来源重评：若卡片确实无关或不适用，从该子问题basis_refs移除，"
                "并同步检查contraindication_evidence和whole_product_support，不能保留相同的不适用引用；"
                "若引用是必要的但缺少card_assessment，补充真实的身份与适用性判断。"
                "不能仅为通过校验把irrelevant/not_applicable改成适用，改变判断必须给出实际匹配理由。"
                "依据发生变化后重新核对proposed_action、evidence_status、conditions、缺失药物知识和review_level，"
                "不能维持只靠已删除依据支持的肯定结论。允许basis_refs=[]并保留undetermined及证据缺口，"
                "不可因此省略子问题；allow_model_knowledge=false时仍不得增加模型记忆。")
    if "unknown assessed card" in message:
        return "card_id只填卡片ID，不含冒号和字段路径。可用卡片：" + json.dumps(payload.get("card_catalog", {}), ensure_ascii=False)
    if "empty refs" in message:
        return "每条推理必须引用已有M编号或实际外部输入ref；用户补充的直接事实用external并引用U编号。没有记录的药物参数不可凭空新增。"
    if "structured half-life input" in message:
        return ("仅当确实记得该确切药物或明确成分的半衰期时，在同一M条目填写half_life_hours的min和max"
                "（小时），并在text中写明同一数值、适用药物/成分。若参数未知，省略half_life_hours，"
                "删除依赖该参数的数值推演，保留不确定性及可回答的一般信息。不能按短效药物、植物提取物"
                "或中成药的一般印象编造区间；复方不能凭空指定整体半衰期。不得为了通过校验填写数字。")
    if "conclusion must disclose model knowledge" in message:
        return "决策依据中使用了M编号；答案basis需要单列model_knowledge依据，refs引用该M编号，并在conclusion_basis中列出此依据序号。"
    if "conclusion_basis" in message:
        return "conclusion_basis必须是非空整数数组，元素为该子问题basis的现有序号，从1开始。"
    if "half-life reasoning" in message:
        return "当前没有适用的已记录半衰期参数。删除半衰期及清除时间推演，依据现有安全资料回答；不得推定未知药代参数。"
    if "absolute safety assurance" in message:
        return "删除零风险保证，改为资料支持的有边界结论；例如可用不等于保证不会发生任何不良反应。"
    if message.startswith("invalid decision evidence reference: "):
        bad = message.partition(": ")[2]
        refs = payload.get("sources", {})
        stem = bad.rsplit("[", 1)[0] if "[" in bad else bad
        candidates = [ref for ref in refs if ref.startswith(stem + "[")]
        if not candidates:
            candidates = difflib.get_close_matches(bad, refs, n=5, cutoff=0.65)
        if candidates:
            return "该引用不存在。请根据对应原文核对；可能的真实字段ref：" + ", ".join(candidates[:8])
        return "该引用不存在；只能使用输入sources中实际提供的ref，或删去无法验证的依据。"
    if "invalid card applicability" in message:
        actual = []
        for sub in (parsed or {}).get("subquestions", []):
            for item in sub.get("card_assessments", []):
                if item.get("applicability") not in {"applicable", "conditional", "not_applicable"}:
                    actual.append(str(item.get("applicability")))
        return ("card_assessments[].applicability只能是 applicable、conditional、not_applicable；"
                + "本次其他取值：" + ", ".join(actual[:5]))
    if "contraindication must be scoped" in message:
        return ("逐个核对contraindication_evidence[].support：其ref必须逐字对应sources，"
                "且必须列入同一subquestion的basis_refs，相关卡片在该子问题须适用；"
                "不满足时删除这条禁忌，不能把其他药物的禁忌移给目标药。")
    return "请按固定字段和取值修复，不要增添未经核对的患者事实或来源。"


class Client:
    def __init__(self, args):
        self.args = args
        self.api_key = os.getenv("CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        self.url = (os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        # CHAT_MODELS alone must not silently prepend an unrelated default model.
        configured = os.getenv("CHAT_MODELS", "")
        first = next(iter(configured.replace(";", ",").replace(" ", ",").split(",")), "")
        preferred = os.getenv("CHAT_MODEL") or first or "qwen3.7-max-preview"
        requested_pool = parse_model_pool(preferred, configured)
        mode = getattr(args, "thinking_mode", "off")
        thinking_only = {"qwen3.7-max-preview", "qwen3.7-max-2026-05-17"}
        skipped = [name for name in requested_pool if mode == "off" and name.lower() in thinking_only]
        self.pool = [name for name in requested_pool if name not in skipped]
        if skipped:
            print(f"[V7] Skipping thinking-only models with --thinking-mode off: {','.join(skipped)}", flush=True)
        require(bool(self.pool), "No model supports --thinking-mode off in CHAT_MODELS. "
                "Use --thinking-mode bounded or configure qwen3.7-max-2026-06-08 / qwen3.7-plus-2026-05-26.")
        self.model = self.pool[0]
        self.settings = {"thinking_mode": getattr(args, "thinking_mode", "off"),
                         "thinking_budget": getattr(args, "thinking_budget", 1024),
                         "max_completion_tokens": getattr(args, "max_tokens", 8192),
                         "model_pool": self.pool}

    def system_prompt(self, prompt_name):
        return (SKILL / "prompts" / prompt_name).read_text(encoding="utf-8")

    def check_response(self, event):
        """Optional transport audit hook; the standalone V7 behavior is unchanged."""

    def json_call(self, prompt_name, payload, validator, audit, persist):
        require(bool(self.api_key), "Set DASHSCOPE_API_KEY or CHAT_API_KEY before an API call")
        prompt = self.system_prompt(prompt_name)
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(self.args.json_retries + 1):
            parsed = None
            thinking = self.settings["thinking_mode"] == "bounded"
            request_settings = {"enable_thinking": thinking,
                                "max_completion_tokens": self.args.max_tokens}
            if thinking:
                request_settings["thinking_budget"] = self.settings["thinking_budget"]
            event = {"started_at": now(), "prompt": prompt_name, "prompt_sha256": digest(prompt),
                     "request_sha256": digest(payload), "request": payload,
                     "attempt": attempt + 1, "model_pool": self.pool,
                     "request_settings": request_settings}
            audit.append(event)
            persist()
            try:
                response = call_chat_api(
                    api_url=self.url, api_key=self.api_key, model=self.model, models=self.pool,
                    messages=messages, max_tokens=self.args.max_tokens, timeout=self.args.api_timeout,
                    temperature=0, max_retries=self.args.api_max_retries,
                    **request_settings,
                )
                raw = answer_text(response)
                event.update({"raw_response": raw, "actual_model": response.get("model"),
                              "model_route": response.get("_medsafe_model_route"),
                              "usage": response.get("usage"), "finished_at": now(),
                              "finish_reason": response.get("choices", [{}])[0].get("finish_reason")})
                self.check_response(event)
                measured = response.get("usage") or {}
                completion = measured.get("completion_tokens") or 0
                reasoning = (measured.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                if (completion > self.args.max_tokens or (not thinking and reasoning > 0)
                        or (thinking and reasoning > self.settings["thinking_budget"])):
                    event["budget_violation"] = {"completion_tokens": completion, "reasoning_tokens": reasoning}
                    raise RuntimeError("Provider did not honor requested token/thinking limits; stopped without another model retry")
            except Exception as exc:
                event.update({"error_type": type(exc).__name__, "finished_at": now()})
                if isinstance(exc, requests.HTTPError):
                    detail = _response_error_summary(exc.response)
                    event["provider_error"] = detail
                    persist()
                    raise RuntimeError(
                        f"Chat API rejected model pool {','.join(self.pool)}: "
                        f"HTTP {detail.get('http_status')} {detail.get('provider_code') or ''}: "
                        f"{detail.get('message') or 'No provider detail returned'}"
                    ) from exc
                persist()
                raise
            try:
                require(event["finish_reason"] != "length", "response truncated; raise --max-tokens")
                parsed = parse_json(raw)
                data = validator(parsed)
                event["validation"] = "passed"
                event["validation_adjustments"] = data.get("_validation_adjustments", [])
                persist()
                return data
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                event["validation"] = str(exc)
                persist()
                if attempt == self.args.json_retries:
                    raise ValueError(f"{prompt_name}: JSON/schema validation failed: {exc}") from exc
                print(f"[V7][repair] {exc}", flush=True)
                guidance = repair_guidance(exc, payload, parsed)
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                     f"上次JSON未通过结构/引用校验：{exc}。{guidance}"
                     "只纠正错误字段，保持其余有效内容和子问题ID；重新输出完整JSON。"},
                ]


def pick_cards(retrieval: dict, store: EvidenceCardStore, top_k: int) -> list[dict]:
    """Keep final candidates first, then fill from attached retrieval cards."""
    if top_k <= 0:
        return []
    names = retrieval.get("final_reranked_top_20") or []
    attached = retrieval.get("medsafe_skill", {}).get("retrieved_cards", [])
    selected = []

    def add_card(cid):
        if cid in store.by_card_id and cid not in selected and len(selected) < top_k:
            selected.append(cid)

    if names:
        for name in names:
            if len(selected) >= top_k:
                break
            for entry in attached:
                if entry.get("matched") is not False and normalize_text(entry.get("drug")) == normalize_text(name):
                    cid = (entry.get("card") or {}).get("card_id")
                    add_card(cid)
            if not any(normalize_text(store.by_card_id[c]["identity"]["canonical_name"]) == normalize_text(name)
                       for c in selected):
                for card in store.by_name.get(normalize_text(name), []):
                    identity = card.get("identity", {})
                    actual_names = [identity.get("canonical_name"), identity.get("source_name"),
                                    *(identity.get("aliases") or [])]
                    if any(normalize_text(n) == normalize_text(name) for n in actual_names if n):
                        add_card(card["card_id"])
    # Attached RRF fillers remain candidates; appraise decides their applicability.
    for entry in attached:
        if len(selected) >= top_k:
            break
        if entry.get("matched") is False:
            continue
        add_card((entry.get("card") or {}).get("card_id"))
    return [store.by_card_id[c] for c in selected]


def sources_for(record):
    return make_sources(record["query"], record["cards"], record["supplements"])


def compact_sources(sources):
    # Full field paths are already in reference IDs; avoid repeating page/card metadata per field in prompts.
    return {ref: {"kind": source["kind"], "text": source["text"]} for ref, source in sources.items()}


def card_catalog(record):
    return {c["card_id"]: c["identity"]["canonical_name"] for c in record["cards"]}


def select_records(records, args):
    selected = records
    ids = getattr(args, "case_ids", None)
    if ids:
        missing = set(ids) - {str(r["id"]) for r in records}
        require(not missing, f"Requested case IDs not found: {sorted(missing)}")
        selected = [r for r in records if str(r["id"]) in ids]
    return selected[:args.limit] if args.limit else selected


def appraise(record, client, final, persist):
    sources = sources_for(record)
    previous = record.get("packets", [])
    had_previous = bool(previous)
    payload = {"query": record["query"], "sources": compact_sources(sources), "allow_model_knowledge": final,
               "card_catalog": card_catalog(record),
               "retrieval_status": record["retrieval_status"],
               "previous_subquestions": [{"id": d["id"], "question": d["question"],
                                          "card_assessments": d["card_assessments"]}
                    for d in previous[0]["decisions"]] if previous else []}

    def validator(data):
        normalize_appraisal_structure(data, sources, previous[0]["decisions"] if previous else None)
        validate_appraisal(data, sources, final)
        if previous:
            require([s["id"] for s in data["subquestions"]] ==
                    [s["id"] for s in previous[0]["decisions"]], "subquestion IDs/order changed after supplementation")
        return data

    data = client.json_call("consultation_appraisal_v7.txt", payload, validator, record["api_audit"], persist)
    packet = build_packet(data, sources, version=len(previous) + 1, final=final,
                          parent=previous[-1]["packet_id"] if previous else None)
    record.setdefault("appraisals", []).append(data)
    record.setdefault("packets", []).append(packet)
    record["state_history"].append({"at": now(), "state": "reappraised" if had_previous else "appraised",
                                     "packet_id": packet["packet_id"]})
    return packet


def export(run_dir, records):
    results, markdown = [], ["# MedSafe v7 consultation drafts", "这些是研究草稿；结构校验通过不等于临床正确性验证。"]
    for record in records:
        final = record.get("final") or {}
        result = {"id": record["id"], "query": record["query"], "status": record["status"],
                  "answer": final.get("answer"), "validation": final.get("validation"),
                  "error": record.get("error"), "decision_packet": final.get("decision_packet"),
                  "claim_ledger": final.get("claim_ledger", []), "packet_history": record.get("packets", []),
                  "state_history": record.get("state_history", []), "supplements": record["supplements"],
                  "sources": sources_for(record), "api_audit": record["api_audit"]}
        results.append(result)
        markdown.extend([f"\n## Case {record['id']}", record["query"], f"状态：{record['status']}",
                         final.get("answer") or "尚未生成最终草稿。" + (f"错误：{record['error']}" if record.get("error") else "")])
    save(run_dir / "final_answers_v7.json", results)
    path = run_dir / "final_answers_readable.md"
    path.write_text("\n\n".join(markdown) + "\n", encoding="utf-8")


def print_requests(record):
    print(f"\n=== Case {record['id']} ===\n{record['query']}", flush=True)
    packet = record["packets"][-1]
    count = 0
    for decision in packet["decisions"]:
        print(f"\n子问题：{decision['question']}")
        for missing in decision["missing_patient"]:
            print(f"  患者信息（{'影响当前决策' if missing['critical'] else '可用条件说明处理'}）：{missing['question']}")
            count += 1
        for missing in decision["missing_drug_knowledge"]:
            print(f"  药物资料（{missing['drug']}）：{missing['question']}")
            count += 1
    return count


def prepare(args, client, records, persist):
    require(args.source_run_dir is not None, "prepare requires --source-run-dir")
    source = args.source_run_dir.resolve()
    require(source != args.run_dir.resolve(), "Use a new --run-dir; never overwrite the frozen source run")
    queries = load(source / "clinical_cases_public.json")
    retrievals = load(source / "hrc_retrieval_with_cards.json")
    require(len({str(q['id']) for q in queries}) == len(queries), "duplicate public case IDs")
    require(len({str(r['id']) for r in retrievals}) == len(retrievals), "duplicate retrieval IDs")
    by_id = {str(r["id"]): r for r in retrievals}
    store = EvidenceCardStore.from_jsonl(args.cards)
    selected = select_records(queries, args)
    supplement_source = getattr(args, "reuse_supplements_from", None)
    imported = {str(r["id"]): r for r in load(supplement_source)} if supplement_source else {}
    appraisal_source = getattr(args, "reuse_appraisals_from", None)
    reused = {str(r["id"]): r for r in load(appraisal_source)} if appraisal_source else {}
    old = {str(r["id"]): r for r in records}
    for query in selected:
        cid = query["id"]
        raw = by_id.get(str(cid))
        require(raw is not None and raw.get("query") == query["query"], f"case {cid}: missing/mismatched retrieval")
        require(not raw.get("error"), f"case {cid}: source retrieval contains an error")
        status = (raw.get("retrieval_quality") or {}).get("status", "not_recorded")
        require(status != "critic_unavailable", f"case {cid}: Critic API unavailable; rerun retrieval first")
        cards = pick_cards(raw, store, args.top_k)
        prior = reused.get(str(cid))
        original = imported.get(str(cid)) or prior
        if original:
            require(original["query"] == query["query"], f"Case {cid}: supplementary input belongs to a different query")
        if prior:
            require(prior["query"] == query["query"] and prior["cards"] == cards and
                    prior["retrieval_status"] == status, f"Case {cid}: cached appraisal inputs changed")
            require(bool(prior.get("appraisals")), f"Case {cid}: no completed initial appraisal to reuse")
            require(not imported.get(str(cid)) or imported[str(cid)].get("supplements", []) ==
                    prior.get("supplements", []), f"Case {cid}: reused supplements differ from appraisal record")
        fingerprint = digest({"id": cid, "query": query["query"], "cards": cards,
                              "retrieval_status": status, "implementation": version_hash(),
                              "imported_supplements": {"text_log": original.get("supplements", []),
                                                       "closed": original.get("supplementation_closed", False)} if original else None,
                              "generation_settings": getattr(client, "settings", {}),
                              "initial_appraisal_import": digest(prior["appraisals"][0]) if prior else None})
        record = old.get(str(cid))
        if record and record["input_fingerprint"] != fingerprint:
            raise ValueError(f"Case {cid} input/code changed. Use a NEW run directory to preserve audit history.")
        if record and record.get("packets"):
            print(f"[V7] Reuse prepared case {cid}", flush=True)
            continue
        if record is None:
            record = {"id": cid, "query": query["query"], "cards": cards, "retrieval_status": status,
                      "retrieval_reused_from": str(source), "input_fingerprint": fingerprint,
                      "implementation_sha256": version_hash(), "supplements": [], "api_audit": [],
                      "generation_settings": getattr(client, "settings", {}),
                      "packets": [], "state_history": [], "status": "preparing"}
            if original:
                record["supplements"] = original.get("supplements", [])
                record["supplementation_closed"] = bool(original.get("supplementation_closed"))
                record["supplement_import"] = {"from": str(supplement_source or appraisal_source), "at": now(),
                                               "original_text_reused_only": True}
            records.append(record)
        print(f"[V7] Appraise case {cid}; cards={len(cards)}", flush=True)
        try:
            if prior:
                sources = sources_for(record)
                initial = deepcopy(prior["appraisals"][0])
                normalize_appraisal_structure(initial, sources)
                validate_appraisal(initial, sources, False)
                packet = build_packet(initial, sources, version=1, final=False, parent=None)
                record["appraisals"] = [initial]
                record["packets"] = [packet]
                record["appraisal_import"] = {"from": str(appraisal_source), "at": now(),
                                              "source_implementation_sha256": prior.get("implementation_sha256"),
                                              "source_appraisal_sha256": digest(prior["appraisals"][0]),
                                              "rebuilt_under_current_rules": True}
                record["api_audit"].append({"at": now(), "action": "reuse_initial_appraisal_no_api",
                                            **record["appraisal_import"]})
                record["state_history"].append({"at": now(), "state": "appraised_from_validated_cache",
                                                "packet_id": packet["packet_id"]})
                print(f"[V7] Reused validated initial appraisal for case {cid}; no API call", flush=True)
            else:
                appraise(record, client, False, persist)
            record["status"] = "supplementation_complete" if record.get("supplementation_closed") else "awaiting_supplementation"
            record.pop("error", None)
        except Exception as exc:
            record["status"] = "prepare_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if not args.continue_on_error:
                raise
        finally:
            persist()


def supplement(args, client, records, persist):
    for record in select_records(records, args):
        require(bool(record.get("packets")), f"case {record['id']}: prepare incomplete")
        require(record["implementation_sha256"] == version_hash(), "Code changed; use a new run directory")
        require(record.get("generation_settings", {}) == getattr(client, "settings", {}),
                "Generation settings changed; use the settings from prepare or a new run directory")
        if record.get("supplementation_closed") and not args.reopen:
            print(f"[V7] Reuse completed supplementation for case {record['id']}")
            continue
        count = print_requests(record)
        if not count and not args.reopen:
            record["supplementation_closed"] = True
            record["status"] = "supplementation_not_needed"
            persist()
            continue
        print("输入作为本工作流已确认信息保存，不需要置信度；建议保留来源。\n"
              "患者信息不能靠模型补全；药物知识可跳过，第三步会标明模型知识。\n"
              "命令：/show 查看原始补充；/done 结束本例（允许留空）；/quit 保存退出。")
        while True:
            try:
                value = input("请补充：").strip()
            except EOFError:
                print("输入已关闭，已保存当前内容；本例尚未结束，请在交互终端继续。")
                return
            if value == "/quit":
                persist()
                return
            if value == "/show":
                print(json.dumps(record["supplements"], ensure_ascii=False, indent=2))
                continue
            if value == "/done":
                # A single semantic call after a completed turn batch, not after every keystroke.
                signature = digest(record["supplements"])
                try:
                    if record["supplements"] and record.get("supplement_appraisal_sha256") != signature:
                        appraise(record, client, False, persist)
                        record["supplement_appraisal_sha256"] = signature
                    record["supplementation_closed"] = True
                    record["status"] = "supplementation_complete"
                    record.pop("error", None)
                    print("已保存；仍未知的患者情况保留为未知，第三步将重新评估药物知识和条件性结论。")
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["status"] = "supplement_appraisal_error"
                    raise
                finally:
                    persist()
                break
            if value:
                record["supplements"].append({"text": value, "at": now()})
                record["supplementation_closed"] = False
                record.pop("final", None)
                record["status"] = "supplementation_in_progress"
                persist()
                print("原文已保存，将在 /done 后由大模型统一理解。")


def finalize(args, client, records, persist):
    for record in select_records(records, args):
        require(record["implementation_sha256"] == version_hash(), "Code changed; use a new run directory")
        require(record.get("generation_settings", {}) == getattr(client, "settings", {}),
                "Generation settings changed; use the settings from prepare or a new run directory")
        require(bool(record.get("packets")), f"case {record['id']}: prepare incomplete")
        require(record.get("supplementation_closed"),
                f"case {record['id']}: run supplement first; /done explicitly skips unavailable information")
        fingerprint = digest({"input": record["input_fingerprint"], "supplements": record["supplements"]})
        if record.get("final", {}).get("fingerprint") == fingerprint:
            print(f"[V7] Reuse final case {record['id']}")
            record["status"] = "draft_generated"
            record.pop("error", None)
            persist()
            continue
        print(f"[V7] Finalize case {record['id']}", flush=True)
        try:
            if record.get("final_plan_fingerprint") != fingerprint:
                packet = appraise(record, client, True, persist)
                record["final_plan_fingerprint"] = fingerprint
                record["final_packet_id"] = packet["packet_id"]
                persist()
            else:
                packet = next(p for p in record["packets"] if p["packet_id"] == record["final_packet_id"])
            sources = sources_for(record)
            used = set().union(*(allowed_answer_refs(d, packet, sources) for d in packet["decisions"]))
            payload = {"query": record["query"],
                       "sources": compact_sources({r: s for r, s in sources.items() if r in used}),
                       "decision_packet": packet,
                       "allowed_refs_by_subquestion": {d["id"]: sorted(allowed_answer_refs(d, packet, sources)) for d in packet["decisions"]}}
            data = client.json_call("consultation_answer_v7.txt", payload,
                                    lambda data: validate_answer(data, packet, sources), record["api_audit"], persist)
            answer, ledger = render_answer(data, packet, sources)
            record["final"] = {"fingerprint": fingerprint, "answer": answer, "structured_answer": data,
                               "claim_ledger": ledger, "decision_packet": packet,
                               "validation": {"schema_and_source_spans_passed": True,
                                              "clinical_accuracy_evaluated": False}, "at": now()}
            record["status"] = "draft_generated"
            record.pop("error", None)
            record["state_history"].append({"at": now(), "state": "draft_generated", "packet_id": packet["packet_id"]})
        except Exception as exc:
            record["status"] = "finalize_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if not args.continue_on_error:
                raise
        finally:
            persist()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "supplement", "finalize"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--reuse-supplements-from", type=Path,
                        help="Explicitly import matching original user input from an older consultation_cases_v7.json")
    parser.add_argument("--reuse-appraisals-from", type=Path,
                        help="Revalidate and rebuild the first completed appraisal from an older consultation_cases_v7.json")
    parser.add_argument("--case-ids", nargs="+", help="Select particular case IDs, e.g. --case-ids 3")
    parser.add_argument("--cards", type=Path, default=SKILL / "resources/evidence_cards.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="First N cases, 0 means all")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Maximum unique cards: final candidates first, then attached retrieval fillers")
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--json-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Total completion tokens INCLUDING reasoning")
    parser.add_argument("--thinking-mode", choices=["off", "bounded"], default="off")
    parser.add_argument("--thinking-budget", type=int, default=1024, help="Per-request reasoning budget in bounded mode, including JSON repairs")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--reopen", action="store_true", help="Reopen supplementation, preserving original input logs")
    args = parser.parse_args()
    require(args.limit >= 0 and args.top_k > 0, "limit must be >=0; top-k must be >0")
    require(args.json_retries >= 0 and args.api_max_retries >= 0 and args.max_tokens > 0 and args.api_timeout > 0,
            "invalid timeout/retry/token limits")
    require(args.thinking_budget > 0, "thinking-budget must be positive")
    require(args.thinking_mode == "off" or args.thinking_budget < args.max_tokens,
            "thinking-budget must leave room within max-tokens for the final JSON")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    path = args.run_dir / "consultation_cases_v7.json"
    records = load(path) if path.exists() else []
    if args.stage != "prepare":
        require(bool(records), "No prepared cases. Run prepare first.")

    def persist():
        save(path, records)
        export(args.run_dir, records)

    client = Client(args)
    print(f"[V7] stage={args.stage}; model pool={','.join(client.pool)}; resume=automatic", flush=True)
    try:
        {"prepare": prepare, "supplement": supplement, "finalize": finalize}[args.stage](args, client, records, persist)
    finally:
        if records:
            persist()
    selected = select_records(records, args)
    failures = [r["id"] for r in selected if r.get("error")]
    print(json.dumps({"version": VERSION, "selected_count": len(selected), "error_case_ids": failures,
                      "generated_count": sum(bool(r.get("final")) for r in selected),
                      "answers": str(args.run_dir / "final_answers_readable.md"),
                      "audit": str(path)}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
