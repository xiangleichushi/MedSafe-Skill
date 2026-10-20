---
name: medsafe-skill
description: Source-attributed consultation workflow for pregnancy and lactation medication questions.
---

# MedSafe-Skill

This file describes the evaluated V7 consultation skill. The implementation
is in `core/consultation_v7.py` and `scripts/run_consultation_v7.py`; the
matched three-condition evaluation is in `ablation_v7/`.

1. Separate a consultation into medication and decision-specific subquestions.
2. Assess each retrieved Card's role and applicability to each subquestion.
   Patient facts must be grounded in the question or authorized user input.
3. Distinguish a missing patient fact from missing general drug knowledge.
   A user may supply external information; any model-derived drug knowledge
   is recorded separately and cannot establish unreported patient details.
4. Apply the recorded decision constraints to clinical action, external
   evidence status, and clarification or professional-review needs.
5. Generate a conclusion, rationale, and precautions with source references.
   Validate the structured decision and the cited source spans. Store the
   accepted answer and audit trail for review.

These checks do not independently establish medical correctness. The workflow
produces consultation drafts for clinician review, not autonomous prescribing.
See `docs/CONSULTATION_V7_EN.md` for the detailed method and its limitations.
