# MedSafe-Skill objective metrics

> These are objective process, traceability, coverage, safety-policy, and resource metrics. They must not be described as clinical accuracy.

## Cohort and pipeline completion

| Metric | Result |
|---|---:|
| Real-world consultation cases | 51 |
| Semantically verified observation records | 51/51 (100.0%) |
| Automated HRC Critic gate passed | 24/51 (47.1%) |
| Final answers passing deterministic validation | 41/51 (80.4%) |
| Complete structured reasoning traces | 51/51 (100.0%) |

## Evidence and claim traceability

| Metric | Result |
|---|---:|
| Cases with a directly cited Evidence Card in the final answer | 34/51 (66.7%) |
| Appraised cards traceable to source pages | 255/255 (100.0%) |
| Stage-1 claims passing deterministic verification | 264/270 (97.8%) |
| Card-bound claims with document and PDF pages | 190/190 (100.0%) |
| Card-bound claims with a field path | 190/190 (100.0%) |
| Rejected Stage-1 claims isolated from final answers | 6/6 (100.0%) |
| Cases with all requested facts supported by retrieved cards | 21/51 (41.2%) |

## Safety and governance

| Metric | Result |
|---|---:|
| Cases with valid contiguous state-machine traces | 51/51 (100.0%) |
| Cases passing deterministic-control audit | 51/51 (100.0%) |
| Unsafe source-to-bounded action downgrades | 0/529 (0.0%) |
| Conservative source-to-bounded action escalations | 36/529 (6.8%) |
| Case-level avoid-to-continue/caution violations | 0/51 (0.0%) |
| Review-level under-escalations | 0/51 (0.0%) |
| Internal provenance labels leaked into clinical prose | 0/51 (0.0%) |

## Evidence gaps and controlled fallback

| Metric | Result |
|---|---:|
| Cases requiring external evidence requests | 5/51 (9.8%) |
| Cases with user-supplied supplemental values | 13/51 (25.5%) |
| Cases using programmatic derived estimates | 7/51 (13.7%) |
| Cases using disclosed low-confidence model priors | 37/51 (72.5%) |
| Model-prior claims carrying fixed risk disclosures | 0/55 (0.0%) |

## Output and resource use

| Metric | Result |
|---|---:|
| Cases requiring one final automatic revision | 38/51 (74.5%) |
| Final-answer API responses | 108 |
| Final-answer recorded total tokens | 935970 |
| Mean final-answer tokens per case | 18352.353 |
| Median final-answer length, Chinese characters | 654 |
| Historical-answer character-token ROUGE-L F1, mean | 0.123 |

## Distributions

- Reproductive context: `{"None": 2, "lactation": 31, "preconception": 2, "pregnancy": 16}`
- Consultation intent: `{"breastfeeding_interruption": 5, "dose_or_exposure": 4, "medication_safety_and_compatibility": 34, "monitoring_and_adverse_effects": 2, "safer_alternative": 6}`
- Final action: `{"caution": 15, "continue": 2, "undetermined": 34}`
- Review level: `{"pharmacist_review": 36, "specialist_review": 15}`
- Final state: `{"awaiting_external_evidence": 5, "human_review_required": 46}`
- Source-native classification labels: `{"A": 5, "B": 43, "B（3个月和6个月时），D（9个月）": 1, "B（前20周）；D（20周后）": 1, "C": 116, "C（妊娠前3个月和第3～6个月），D（第6～9个月）": 1, "D": 26, "NS": 15, "NS(可能)": 1, "NS（可能的）": 1, "NS（可能）": 3, "S": 110, "S(likely)": 2, "S(possibly)": 2, "S（likely）": 6, "S（可能）": 56, "S（局部），U（口服）": 1, "S（局部）；U（口服）": 1, "S（有可能）": 1, "U": 129, "U；S（可能）": 1, "X": 6, "无显著意义/不充分": 1}`
- Source-to-bounded action transitions: `{"avoid->avoid": 26, "caution->caution": 212, "continue->caution": 36, "continue->continue": 122, "undetermined->undetermined": 133}`
- Missing-information slots: `{"allergy_reaction": 2, "dose": 11, "gestational_age": 3, "infant_age": 9, "prematurity_status": 12, "time_since_last_dose": 6}`
- Model-prior target fields: `{"authoritative_breastfeeding_resumption_recommendation": 2, "five_half_life_inference": 10, "general_clinical_interpretation": 22, "infant_oral_bioavailability": 1, "lactation_compatibility_assessment": 7, "milk_transfer_or_relative_infant_dose": 1, "monitoring_recommendation": 6, "provisional_clinical_answer": 5, "unsupported_fact:breastfeeding_interval": 1}`

## Interpretation boundary

A 100% deterministic validation rate means schema, provenance, numeric, and policy checks passed; it does not establish medical correctness.

The held-out workbook answers are marked source_workbook_text_not_gold. BLEU, ROUGE, or BERTScore against them may be secondary descriptive similarity only, not clinical correctness.
