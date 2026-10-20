# MedSafe-Skill

MedSafe-Skill is a research workflow for source-attributed medication consultation
in pregnancy and lactation. It builds Evidence Cards from a structured drug
reference, appraises retrieved evidence for each clinical subquestion, records
decision constraints and provenance, and generates a draft for clinician review.
It is not an autonomous prescribing system.

中文概览：本仓库包含妊娠期与哺乳期用药咨询的 Evidence Card、检索结果
接入、逐问题证据评估、约束性决策、可选人工补充、答案生成和审计记录。
它复用 [HRC-RAG](https://github.com/xiangleichushi/HRC-RAG) 的文本提取、
嵌入及检索代码；本仓库另附 BM25 检索基线与三组答案对照实验。
系统输出是供临床人员审核的研究草稿，不是自动处方建议。

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
  their index and build summary. The separate per-drug JSON files are provided
  as `evidence_cards.zip` for download; unpacking them is optional.
- `vector_store/`: saved FAISS indexes and drug metadata, not model weights.
- `output/`: frozen inputs, consultation traces, three-condition answers, and
  summaries. Historical runs are retained for audit; use the run identified in
  the manuscript for reported values.
- `bm25/`: the lexical retrieval baseline, its inputs, tests, and saved
  75-query results. See `bm25/README.md`.

The older `ablation/` and development-stage documents are not required for
the reported V7 comparison. `scripts/run_answer_experiment.py` is retained
because V7 imports its API transport helpers.

## Directory guide / 目录说明

| Path | Contents |
| --- | --- |
| `core/`, `prompts/`, `scripts/` | 核心规则、模型提示词、数据准备与 V7 三阶段入口 |
| `resources/`, `evidence_cards.zip` | 版本化 schema、规则、运行用 JSONL 和单条卡片压缩包 |
| `vector_store/` | 两个 FAISS 检索索引及药物元数据，不含嵌入模型权重 |
| `output/clinical_cases_full_v6/` | 冻结的 51 例输入与 HRC-RAG 检索/卡片结果 |
| `output/revision_ablation_v7_repaired/` | 本次三组对照结果、客观汇总及逐例审计 |
| `ablation_v7/` | LLM-only、HRC-RAG + Cards、MedSafe-Skill 的同模型实验入口 |
| `bm25/` | 75 问 BM25 基线代码、输入、测试和结果 |
| `tests/`, `docs/` | 离线测试及 V7 架构文档 |

The saved V7 comparison includes 51 answers for each direct-generation
baseline and 50 MedSafe-Skill answers; the incomplete case is identified in
`output/revision_ablation_v7_repaired/objective_summary.json`. Older output
directories are retained as development/run history and must not be mixed
with the reported comparison.

For the GitHub browser-upload edition, the two revision-result directories
are supplied as ZIP archives because of browser file-size limits. Follow
`RESTORE_RESULTS.md` after downloading to recreate the paths above.

## Quick start / 运行说明

在 Linux/AutoDL 上，先按上面的命令把本仓库克隆到
`HRC-RAG/my_project/medsafe_skill`，再从 `HRC-RAG/my_project` 执行下列命令。
Windows 上的离线测试可用 PowerShell 执行相同的 `python` 命令。

**1. 无 API 的检查：**

```bash
python -B -m unittest medsafe_skill.tests.test_consultation_v7 medsafe_skill.tests.test_ablation_v7
python -B -m unittest discover -s medsafe_skill/bm25 -p test_bm25_baseline.py
python -B medsafe_skill/bm25/bm25_baseline.py --output-dir /tmp/medsafe_bm25_check
```

最后一条重新计算 75 问 BM25；仓库内已保存原结果，故示例输出到临时目录。
需要与 HRC-RAG 的历史指标函数交叉核对时，运行
`python -B medsafe_skill/bm25/verify_bm25_baseline.py`，并确保配套仓库中有
相应的 `report_critic.py`、`report_rerank.py` 与历史排名文件。

**2. 重新生成前三例咨询答案：**

```bash
export DASHSCOPE_API_KEY='YOUR_KEY'
export ABLATION_MODEL=qwen3.7-plus
export RUN_DIR=medsafe_skill/output/reproduction_smoke
bash medsafe_skill/ablation_v7/run_autodl.sh smoke
```

运行 51 例时将最后一条的 `smoke` 改为 `full`，并把 `RUN_DIR` 设为新的
输出目录。这个入口使用仓库内冻结的病例和检索结果，三组答案使用同一个
`ABLATION_MODEL`；它不会重新运行 HRC-RAG 检索。实际密钥只应设在环境变量，
不要写进仓库文件或命令脚本。主结果见新目录的 `objective_summary.json`、
`comparison_records.json` 和各组答案。`RUN_DIR` 不要指向已保存的正式结果。

**3. 单独运行交互式 V7 流程（可选）：**

```bash
python medsafe_skill/scripts/run_consultation_v7.py prepare \
  --source-run-dir medsafe_skill/output/clinical_cases_full_v6 \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
python medsafe_skill/scripts/run_consultation_v7.py supplement \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
python medsafe_skill/scripts/run_consultation_v7.py finalize \
  --run-dir medsafe_skill/output/interactive_check --case-ids 1
```

第二步在终端输入病例或药物补充资料，输入 `/done` 结束本例；没有资料也可
直接 `/done`。最终自然语言答案在 `final_answers_readable.md`，状态历史、
证据来源和 API 审计在 `consultation_cases_v7.json`。论文的主对照实验没有
实际补充资料输入，不要把此交互演示当作论文的第四个实验组。

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
