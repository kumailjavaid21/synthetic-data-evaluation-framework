Generated: 2026-01-23 04:53:59
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
Primary privacy indicator is log10 of normalized NN distance median (higher is safer). Raw nn_dist_norm_median is retained in artifacts.
Utility higher is better. CI is bootstrap over seeds.
Note: membership_audit_adv (from mia_attack.json) is real-only and not synth-dependent.
Floating-point equality: comparisons use tolerance (1e-9) to avoid rounding artifacts.
Apparent equality at 6 decimals can occur even when exact metrics differ.