Generated: 2026-01-14 21:52:50
Source: outputs/master_results.csv (canonical merged results).
Tables:
- main_table_overall.csv: best DP vs best overall and deltas vs best non-DP.
- summary_mean_ci95.csv: bootstrap 95% CI per dataset+method (long format).
- summary_mean_std.csv: mean/std per dataset+method.
- dpcr_effect_summary.csv: DP-CR paired deltas with CI.
- money_table_<DS>.csv, winners_overall.csv, winners_dp_only.csv.
Figures:
- utility/privacy/correlation bar charts (mean +/- std).
- frontier plots with mean +/- 95% CI and baseline hull.
Privacy advantage: lower is better (|AUC-0.5| in [0, 0.5]).
Utility higher is better. CI is bootstrap over seeds.
Floating-point equality: comparisons use tolerance (1e-9) to avoid rounding artifacts.
Apparent equality at 6 decimals can occur even when exact metrics differ.