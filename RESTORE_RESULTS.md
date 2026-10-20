# Restore saved results

The GitHub browser upload limit required two saved V7 result directories to
be archived. These are data archives, not model weights. The repository also
contains the smaller frozen inputs under `output/clinical_cases_full_v6/`.

From the repository root on Linux:

```bash
mkdir -p output
unzip revision_ablation_v7_full_answer.zip -d output
unzip revision_ablation_v7_repaired.zip -d output
```

On Windows PowerShell:

```powershell
Expand-Archive .\revision_ablation_v7_full_answer.zip -DestinationPath .\output
Expand-Archive .\revision_ablation_v7_repaired.zip -DestinationPath .\output
```

After extraction, the final comparison summary is at
`output/revision_ablation_v7_repaired/objective_summary.json`. The ZIP files
exclude private review keys. `evidence_cards.zip` contains the 822 separate
per-drug JSON files and is optional for running the workflow, which uses
`resources/evidence_cards.jsonl`. To restore the separate files, extract it
at the repository root.
