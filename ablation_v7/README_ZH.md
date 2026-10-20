# Matched three-condition answer comparison

This directory implements the 51-consultation comparison reported in the
manuscript: `llm_only`, `hrc_rag_cards`, and `medsafe_full`. Both
evidence-based conditions receive the same candidate Evidence Cards. All
three conditions share the clinical task and answer structure and use the
same specified generation model. MedSafe-Skill additionally appraises
evidence by subquestion, applies decision constraints, verifies the final
answer, and repairs contract violations when possible. The comparison
therefore evaluates complete workflows; it cannot isolate the effect of any
single rule. Additional model calls and retries remain in the cost audit.

The shared and direct-generation prompts are in
`prompts/clinical_task_shared.txt` and `prompts/direct_answer.txt`. The full
workflow also uses `../prompts/consultation_appraisal_v7.txt` and
`../prompts/consultation_answer_v7.txt`. `run.py` is the implementation entry
point; `run_autodl.sh` provides `smoke`, `repair-smoke`, and `full` modes.
`repair-smoke` uses six development-stage failure cases for regression
checks, not for the primary comparison.

## Inputs

- `medsafe_skill/resources/evidence_cards.jsonl`: saved guideline-derived
  Evidence Cards. They can be rebuilt from an authorized extraction with
  `scripts/build_evidence_cards.py`.
- `medsafe_skill/output/clinical_cases_full_v6/clinical_cases_public.json`:
  de-identified consultation questions, without historical human replies.
- `hrc_retrieval_with_cards.json` in the same directory: frozen HRC-RAG
  candidates and Card references.
- An API key supplied as `DASHSCOPE_API_KEY` or `CHAT_API_KEY` in the
  environment. Never commit a key. The model checkpoint for live embedding
  retrieval is not included; saved FAISS indexes are provided separately.

The primary 51-case comparison used the frozen retrieval results and no
interactive user supplementation. The audit records requested and returned
models, prompt/input hashes, API time, token usage, failures, and repairs.
Do not pool results from different implementation versions or models without
recording the change.

## Run from the companion project root

```bash
export ABLATION_MODEL=qwen3.7-plus
export DASHSCOPE_API_KEY='YOUR_KEY'
export RUN_DIR=medsafe_skill/output/reproduction_smoke
bash medsafe_skill/ablation_v7/run_autodl.sh smoke
```

`smoke` runs case IDs 1--3. Change it to `full` and choose a **new**
`RUN_DIR` for all cases. `SOURCE_RUN_DIR` overrides the frozen input
directory. If complete baselines already exist and experiment identities
match, `REUSE_BASELINES_FROM` can reuse them while rerunning only
`medsafe_full`; original baseline API costs remain in the audit. A fingerprint
mismatch prevents accidental merging of incompatible runs.

Main outputs include `experiment_manifest.json`, `frozen_cases.json`,
`comparison_records.json`, per-condition answers and audit records, and a
blinded review sheet. The review sheet contains no clinical scores. Clinical
ratings were collected and analyzed separately.

Offline tests need no API key or clinical workbook:

```bash
python -B -m unittest medsafe_skill.tests.test_consultation_v7 medsafe_skill.tests.test_ablation_v7
```
