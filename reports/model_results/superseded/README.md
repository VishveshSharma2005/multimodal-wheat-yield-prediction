# Superseded results — pre-leakage-fix (2026-06-06 run)

**Do not cite anything in this folder.** These files are kept only as provenance
for the corrections applied on 2026-09-05/06, so the before/after of the leakage
fix is auditable.

They were produced *before*:

* `actual_available` (a target-availability indicator) was removed from the
  feature sets,
* 8 exact-duplicate feature aliases were removed,
* the `"text"` substring bug that silently dropped
  `image_texture_proxy_stage_avg` was fixed,
* the LSTM best-checkpoint (`copy.deepcopy`) bug was fixed and the run was seeded.

| File | Superseded by |
|---|---|
| `stagewise_regression_metrics_original.csv` | `../stagewise_regression_metrics.csv` |
| `stagewise_regression_summary_original.md` | `../stagewise_regression_summary.md` |
| `combined_regression_metrics_original.csv` | `../combined_regression_metrics.csv` |
| `combined_regression_summary_original.md` | `../combined_regression_summary.md` |
| `quintile_classification_metrics_original.csv` | `../quintile_classification_metrics.csv` |
| `quintile_classification_summary_original.md` | `../quintile_classification_summary.md` |
| `lstm_metrics_original.csv` | `../lstm_metrics.csv` |

Original model artefacts from the same run are in `archive/models_pre_leakage_fix/`
(local only, not tracked in Git).
