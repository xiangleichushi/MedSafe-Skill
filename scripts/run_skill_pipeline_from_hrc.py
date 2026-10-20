#!/usr/bin/env python
"""Run one live HRC-RAG query and enrich it with MedSafe-Skill outputs.

This script is intended for the AutoDL/server environment where the original
project files are available:

- retriever_critic.py
- vector_store/
- query.json
- local embedding model
- chat-completions API access for query decomposition and critic rerank

It does not reimplement HRC-RAG. Instead, it imports retriever_critic.py from
the project root, runs HybridRetriever.retrieve_single(query), then attaches
Evidence Cards, CSRP labels, a deterministic draft answer, and field-level
claim support checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MEDSAFE_DIR = SCRIPT_DIR.parent
HRC_CRITIC_PROMPT = MEDSAFE_DIR / "prompts" / "hrc_critic_clinical_v2.txt"
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from medsafe_skill.core.qasp_state_machine import finalize_after_claim_verification  # noqa: E402
from apply_csrp_rules import apply_csrp_to_card, detect_context  # noqa: E402
from attach_cards_to_hcr_results import attach_cards  # noqa: E402
from load_card_store import EvidenceCardStore  # noqa: E402
from verify_claim_support import split_claims, verify_claims  # noqa: E402
from medsafe_skill.scripts.run_answer_experiment import (  # noqa: E402
    answer_text as pooled_answer_text,
    call_chat_api as pooled_call_chat_api,
    parse_model_pool,
    response_model_route,
)


def import_retriever_module(project_root: Path):
    module_path = project_root / "retriever_critic.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find retriever_critic.py at {module_path}. "
            "Run this script from the project root or pass --project-root."
        )
    spec = importlib.util.spec_from_file_location("retriever_critic_live", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import retriever_critic.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["retriever_critic_live"] = module
    spec.loader.exec_module(module)
    return module


def maybe_set_chat_config(module: Any, args: argparse.Namespace) -> None:
    api_url = args.chat_api_url or os.getenv("CHAT_API_URL") or os.getenv("DASHSCOPE_CHAT_API_URL")
    api_key = (
        args.chat_api_key
        or os.getenv("CHAT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY_ID")
    )
    model = args.chat_model or os.getenv("CHAT_MODEL") or os.getenv("DASHSCOPE_CHAT_MODEL")
    model_pool = parse_model_pool(
        model or getattr(module, "CHAT_MODEL", "qwen3.7-max"),
        getattr(args, "chat_models", None) or os.getenv("CHAT_MODELS") or "",
    )

    if api_url:
        module.CHAT_API_URL = api_url
    if api_key:
        if os.getenv("DASHSCOPE_API_KEY") and not api_key.lower().startswith("bearer "):
            api_key = f"Bearer {api_key}"
        module.CHAT_API_KEY = api_key
    if model:
        module.CHAT_MODEL = model
    module.MEDSAFE_MODEL_ROUTE_EVENTS = []

    def pooled_text_call(
        messages: list[dict[str, str]],
        model: str = model_pool[0],
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> str:
        response = pooled_call_chat_api(
            api_url=module.CHAT_API_URL,
            api_key=module.CHAT_API_KEY,
            model=model,
            models=model_pool,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=0.0,
            max_retries=getattr(args, "llm_max_retries", 3),
        )
        route = response_model_route(response)
        if route:
            module.MEDSAFE_MODEL_ROUTE_EVENTS.append(route)
        return pooled_answer_text(response)

    module.call_chat_api = pooled_text_call
    if HRC_CRITIC_PROMPT.exists():
        module.CRITIC_PROMPT = HRC_CRITIC_PROMPT.read_text(encoding="utf-8").strip()
        module.CriticEvaluator.STRICT_SYSTEM = (
            "你是临床用药检索相关性审查器。只输出一个合法JSON对象，不输出Markdown或解释文字。"
        )


def path_or_default(value: str | None, fallback: str | Path) -> str:
    if value:
        return str(Path(value))
    return str(fallback)


def build_config(module: Any, args: argparse.Namespace):
    project_root = args.project_root.resolve()
    default_config = module.RetrievalConfig()

    metadata_default = project_root / "vector_store" / "drug_metadata.json"
    cat_default = project_root / "vector_store" / "drug_category_hnsw.index"
    ind_default = project_root / "vector_store" / "indication_hnsw.index"
    query_default = project_root / "query.json"

    metadata_path = metadata_default if metadata_default.exists() else default_config.metadata_path
    category_path = cat_default if cat_default.exists() else default_config.faiss_category_path
    indication_path = ind_default if ind_default.exists() else default_config.faiss_indication_path
    query_path = query_default if query_default.exists() else default_config.query_path

    embedding_path = (
        args.embedding_model_path
        or os.getenv("EMBEDDING_MODEL_PATH")
        or default_config.embedding_model_path
    )

    output_dir = args.retriever_output_dir or str(MEDSAFE_DIR / "output")

    return module.RetrievalConfig(
        metadata_path=path_or_default(args.metadata_path, metadata_path),
        faiss_category_path=path_or_default(args.faiss_category_path, category_path),
        faiss_indication_path=path_or_default(args.faiss_indication_path, indication_path),
        query_path=path_or_default(args.query_path, query_path),
        output_dir=output_dir,
        embedding_model_path=str(embedding_path),
        name_match_top_k=args.retrieval_top_k,
        category_top_k=args.retrieval_top_k,
        indication_top_k=args.retrieval_top_k,
        rrf_top_k=args.retrieval_top_k,
        rrf_k=args.rrf_k,
        llm_max_retries=args.llm_max_retries,
        api_model=args.chat_model or getattr(module, "CHAT_MODEL", default_config.api_model),
        api_max_tokens=args.api_max_tokens,
        api_timeout=args.api_timeout,
        critic_max_rounds=args.critic_max_rounds,
        critic_rerank_top_k=args.retrieval_top_k,
        critic_api_model=args.chat_model or getattr(module, "CHAT_MODEL", default_config.critic_api_model),
        critic_api_max_tokens=args.critic_api_max_tokens,
        warn_print_llm_output=args.warn_print_llm_output,
        warn_llm_output_max_chars=args.warn_llm_output_max_chars,
        verbose_progress_each=True,
    )


def make_draft_answer(query: str, hits: list[dict[str, Any]]) -> str:
    lines = [f"问题：{query}"]
    for hit in hits:
        if not hit.get("matched"):
            lines.append(f"{hit.get('drug')}：未匹配到 Evidence Card，不能生成安全结论。")
            continue
        card = hit["card"]
        csrp = hit["csrp"]
        source = card.get("source_trace", {})
        safety = card.get("safety", {})
        lines.append(
            f"{card.get('drug')}：CSRP标签为{csrp.get('action_label')}（{csrp.get('color_label')}），"
            f"妊娠分级为{safety.get('pregnancy_category') or '未知'}，"
            f"哺乳分级为{safety.get('breastfeeding_category') or '未知'}，"
            f"建议复核级别为{csrp.get('review_level')}，"
            f"证据页码为文档第{source.get('page_doc')}页/PDF第{source.get('page_pdf')}页。"
        )
    lines.append("本回答仅为基于指南证据的草稿，高风险或个体化用药需由医生或药师确认。")
    return "\n".join(lines)


def is_verifiable_claim(claim: str) -> bool:
    return not (
        claim.startswith("问题：")
        or claim.startswith("本回答仅")
        or claim.startswith("高风险或个体化用药")
    )


def build_skill_output(
    query: str,
    hrc_result: dict[str, Any],
    store: EvidenceCardStore,
    skill_top_k: int,
    include_raw_hrc: bool,
    observation_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched_items, summary = attach_cards(
        [hrc_result],
        store=store,
        top_k=skill_top_k,
        observation_cache=observation_cache,
    )
    enriched = enriched_items[0]
    skill = enriched["medsafe_skill"]
    matched_cards = [
        store.get(hit["card"]["card_id"])
        for hit in skill["retrieved_cards"]
        if hit.get("matched")
    ]
    matched_cards = [card for card in matched_cards if card]

    draft_answer = make_draft_answer(query, skill["retrieved_cards"])
    claims = [claim for claim in split_claims(draft_answer) if is_verifiable_claim(claim)]
    claim_verification = verify_claims(claims, matched_cards)
    finalized_workflow = finalize_after_claim_verification(
        skill["clinical_workflow"],
        all_supported=all(item.get("supported") for item in claim_verification),
    )
    skill["clinical_workflow"] = finalized_workflow
    skill["execution_plan"]["clinical_workflow"] = finalized_workflow

    result = {
        "query": query,
        "medsafe_skill": skill,
        "draft_answer": draft_answer,
        "claim_verification": claim_verification,
        "unsupported_claims": [
            item["claim"] for item in claim_verification if not item.get("supported")
        ],
        "summary": summary,
    }
    if include_raw_hrc:
        result["hrc_result"] = hrc_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one live HRC-RAG query with MedSafe-Skill enrichment.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--output", type=Path, default=MEDSAFE_DIR / "output" / "skill_pipeline_single.json")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--cards", type=Path, default=MEDSAFE_DIR / "resources" / "evidence_cards.jsonl")
    parser.add_argument("--observation-cache", type=Path)
    parser.add_argument("--skill-top-k", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--metadata-path")
    parser.add_argument("--faiss-category-path")
    parser.add_argument("--faiss-indication-path")
    parser.add_argument("--query-path")
    parser.add_argument("--retriever-output-dir")
    parser.add_argument("--chat-api-url")
    parser.add_argument("--chat-api-key")
    parser.add_argument("--chat-model")
    parser.add_argument("--api-timeout", type=int, default=60)
    parser.add_argument("--api-max-tokens", type=int, default=512)
    parser.add_argument("--critic-api-max-tokens", type=int, default=1024)
    parser.add_argument("--critic-max-rounds", type=int, default=2)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=40)
    parser.add_argument("--warn-print-llm-output", action="store_true")
    parser.add_argument("--warn-llm-output-max-chars", type=int, default=2500)
    parser.add_argument("--no-raw-hrc", action="store_true", help="Do not include raw HRC-RAG result in output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()

    retriever_module = import_retriever_module(args.project_root)
    maybe_set_chat_config(retriever_module, args)
    config = build_config(retriever_module, args)

    print("[MedSafe] Loading Evidence Cards...")
    store = EvidenceCardStore.from_jsonl(args.cards)
    print(f"[MedSafe] Loaded {len(store.cards)} Evidence Cards.")

    observation_cache = {}
    if args.observation_cache:
        with args.observation_cache.open("r", encoding="utf-8") as f:
            observation_cache = json.load(f)
        if not isinstance(observation_cache, dict):
            raise ValueError("--observation-cache must contain a JSON object")

    print("[MedSafe] Loading HRC-RAG retriever...")
    retriever = retriever_module.HybridRetriever(config)

    print(f"[MedSafe] Running live HRC-RAG query: {args.query}")
    hrc_result = retriever.retrieve_single(args.query)
    hrc_result["id"] = args.query

    skill_output = build_skill_output(
        query=args.query,
        hrc_result=hrc_result,
        store=store,
        skill_top_k=args.skill_top_k,
        include_raw_hrc=not args.no_raw_hrc,
        observation_cache=observation_cache,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(skill_output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[MedSafe] Saved: {args.output}")
    print(json.dumps(skill_output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
