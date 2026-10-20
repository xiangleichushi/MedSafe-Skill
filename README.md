# MedSafe-Skill

MedSafe-Skill is a research workflow for source-attributed medication consultation
in pregnancy and lactation. It builds Evidence Cards from a structured drug
reference, appraises retrieved evidence for each clinical subquestion, records
decision constraints and provenance, and generates a draft for clinician review.
It is not an autonomous prescribing system.

The workflow reuses text extraction, embedding, and retrieval components
from [HRC-RAG](https://github.com/xiangleichushi/HRC-RAG). This repository
adds the clinical consultation workflow, a BM25 retrieval baseline, and a
matched three-condition answer comparison.

## Relationship to HRC-RAG

This repository contains the **MedSafe-Skill extension**, not a second copy of
the HRC-RAG retriever. The companion
[HRC-RAG repository](https://github.com/xiangleichushi/HRC-RAG) provides the
medical-text extraction, embedding/index construction, three-channel retrieval,
RRF fusion, Critic loop, reranking, and retrieval-evaluation code. MedSafe-Skill
imports its `my_project/retriever_critic.py` at runtime for live retrieval; it
can also use a previously saved HRC-RAG retrieval result. The Evidence Card
builder, consultation workflow, prompts, three-condition answer comparison,
and offline contract tests are maintained here. The 75-query retrieval
benchmark belongs to HRC-RAG; the 51-question answer comparison is implemented
in this repository.

The recommended checkout puts this repository inside `HRC-RAG/my_project` with
the directory name `medsafe_skill`:

```bash
git clone https://github.com/xiangleichushi/HRC-RAG.git
cd HRC-RAG/my_project
git clone https://github.com/xiangleichushi/MedSafe-Skill.git medsafe_skill
```

This placement preserves the paths expected by the scripts. Saved vector
indexes and metadata are included under `medsafe_skill/vector_store/`; copy
them to `HRC-RAG/my_project/vector_store/` for live HRC-RAG retrieval. Frozen
retrieval outputs can be audited without rebuilding those indexes. The
fine-tuned embedding-model checkpoint is not present locally, so live
retrieval still requires `EMBEDDING_MODEL_PATH`.

## Repository contents

- `core/`, `scripts/`, and `prompts/`: Evidence Card construction, case import,
  retrieval attachment, evidence appraisal, decision constraints, API transport,
  optional supplementation, answer generation, and audit checks.
- `resources/clinical_observation_schema.json`,
  `resources/clinical_workflow_schema.json`,
  `resources/evidence_card_schema.json`, and `resources/rules/`: versioned schemas
  and research-prototype policies.
- `ablation_v7/`: the matched LLM-only, HRC-RAG plus Cards, and full
  MedSafe-Skill comparison. The same model is configured for all three
  conditions; see `ablation_v7/README_ZH.md`.
- `tests/`: synthetic, offline regression tests for the V7 workflow and
  comparison. `SKILL.md` and `docs/CONSULTATION_V7_EN.md` describe the
  evaluated workflow and its limitations.
- `resources/evidence_cards.jsonl`: the 822 saved Cards used at runtime, plus
  their index and build summary. Separate per-drug JSON files are provided
  in `evidence_cards.zip`; extracting them is optional.
- `vector_store/`: saved FAISS indexes and drug metadata, not model weights.
- `output/`: frozen inputs, consultation traces, three-condition answers, and
  summaries. Historical runs are retained for audit; use the run identified in
  the manuscript for reported values.
- `bm25/`: the lexical retrieval baseline, its inputs, tests, and saved
  75-query results. See `bm25/README.md`.

The older `ablation/` and development-stage documents are not required for
the reported V7 comparison. `scripts/run_answer_experiment.py` is retained
because V7 imports its API transport helpers.

## Directory guide

| Path | Contents |
| --- | --- |
| `core/`, `prompts/`, `scripts/` | Decision rules, prompts, data preparation, and the V7 entry point |
| `resources/`, `evidence_cards.zip` | Versioned schemas and rules, runtime Card JSONL, and archived per-drug Cards |
| `vector_store/` | Two FAISS indexes and drug metadata; no embedding-model checkpoint |
| `output/clinical_cases_full_v6/` | Frozen inputs for 51 consultations and attached HRC-RAG retrieval results |
| `output/revision_ablation_v7_repaired/` | Matched comparison, objective summary, and case-level audit after extraction |
| `ablation_v7/` | Same-model LLM-only, HRC-RAG + Cards, and MedSafe-Skill comparison |
| `bm25/` | Code, inputs, tests, and results for the 75-query BM25 baseline |
| `tests/`, `docs/` | Offline tests and V7 architecture documents |

The saved V7 comparison includes 51 answers for each direct-generation
baseline and 50 MedSafe-Skill answers; the incomplete case is identified in
`output/revision_ablation_v7_repaired/objective_summary.json`. Older output
directories are retained as development/run history and must not be mixed
with the reported comparison.

For the GitHub browser-upload edition, the two revision-result directories
are supplied as ZIP archives because of browser file-size limits. Follow
`RESTORE_RESULTS.md` after downloading to recreate the paths above.

## Quick start

On Linux/AutoDL, clone this repository into
`HRC-RAG/my_project/medsafe_skill` as shown above. Run the following commands
from `HRC-RAG/my_project`. The offline Python checks can also be run from
PowerShell on Windows.

**1. Offline checks (no API key required):**

```bash
python -B -m unittest medsafe_skill.tests.test_consultation_v7 medsafe_skill.tests.test_ablation_v7
python -B -m unittest discover -s medsafe_skill/bm25 -p test_bm25_baseline.py
python -B medsafe_skill/bm25/bm25_baseline.py --output-dir /tmp/medsafe_bm25_check
```

The last command recomputes BM25 over 75 queries in a temporary directory,
leaving the saved results untouched. For an optional cross-check against
HRC-RAG's historical metrics, run
`python -B medsafe_skill/bm25/verify_bm25_baseline.py`. That command requires
`report_critic.py`, `report_rerank.py`, and the saved historical rankings in
the companion project.

**2. Regenerate answers for the first three consultations:**

```bash
export DASHSCOPE_API_KEY='YOUR_KEY'
export ABLATION_MODEL=qwen3.7-plus
export RUN_DIR=medsafe_skill/output/reproduction_smoke
bash medsafe_skill/ablation_v7/run_autodl.sh smoke
```

To process all 51 cases, change `smoke` to `full` and set `RUN_DIR` to a new
directory. This command uses frozen cases and retrieval results and fixes the
same `ABLATION_MODEL` across all three conditions. It does not rerun HRC-RAG
retrieval. Provide the API key only through an environment variable; never
place it in a repository file or shell script. Outputs include
`objective_summary.json`, `comparison_records.json`, and answers for each
condition. Do not point `RUN_DIR` at the saved study results.

**3. Run one interactive V7 consultation (optional):**

```bash
python medsafe_skill/scripts/run_consultation_v7.py prepare \
  --source-run-dir medsafe_skill/output/clinical_cases_full_v6 \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
python medsafe_skill/scripts/run_consultation_v7.py supplement \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
python medsafe_skill/scripts/run_consultation_v7.py finalize \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
```

During the supplement stage, enter additional case or drug information in
the terminal, then enter `/done`. You may enter `/done` without adding any
information. The natural-language answer is saved to
`final_answers_readable.md`; state history, provenance, and API audit are
saved to `consultation_cases_v7.json`. The primary matched experiment did not
include interactive supplementation. This demonstration is not a fourth
experimental condition.

The offline tests use synthetic fixtures and need no clinical workbook, GPU,
or API key. To perform **live** HRC-RAG retrieval instead of replaying frozen
results, put the included index files in the companion project's
`vector_store/`, install the HRC-RAG dependencies, and set
`EMBEDDING_MODEL_PATH` to a compatible model checkpoint. That checkpoint is
not present in this repository. Rebuilding Cards from scratch also requires
the authorized source-book extraction; see `scripts/build_evidence_cards.py`.

The answer experiment logs case/card hashes, model identity, prompts, API
usage, failures, and per-condition outputs. Its recorded counts are 51/51/50
for LLM-only, HRC-RAG + Cards, and MedSafe-Skill. Technical completion does
not establish clinical correctness, and a rerun can produce different text.

## Data and safety

Saved outputs, Cards, BM25 results, and vector indexes are trackable in this
repository. The local inspection found no embedding-model checkpoint, so no
model weights are provided by this release. Add weights only after confirming
their redistribution terms and use Git LFS if needed. API keys and local
credential files must never be committed. Held-out historical consultation
replies and the private review key remain excluded.

**Public-release check:** de-identified consultation questions, generated
traces, and guide-derived Cards may still be subject to institutional approval
and source-book copyright. Confirm permission to redistribute these artifacts
before pushing to a public repository. This README does not grant those
rights. Do not publish raw source PDFs or original clinical workbooks.

The policies in this repository are research prototypes requiring clinical
review. Generated answers must not replace a clinician's or pharmacist's
individual assessment.
