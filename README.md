# Synthetic Data Evaluation Framework

Utility, privacy, and fidelity diagnostics for synthetic data.

## Quickstart
- populate `examples/real_sample.csv` and `examples/synth_sample.csv`
- run `python scripts/run_evaluate.py --real examples/real_sample.csv --synth examples/synth_sample.csv --target target --task classification --outdir for_github/outputs`

## Inputs
- real/synthetic CSVs with matching schema
- CLI options to configure task and target

## Outputs
- `outputs/tables/summary.csv`
- `outputs/figures/` PNG plots

## Reproducibility
- uses fixed seeds for sampling (42)

## Results Preview
- Tables: outputs/tables/summary.csv
- Figures: outputs/figures/summary_metrics.png
```
tstr_score_real   : 0.6
tstr_score_synth  : 0.6
privacy_advantage : -0.3470167042950927
corr_mad          : 0.0
ks_mean           : 0.0
```
