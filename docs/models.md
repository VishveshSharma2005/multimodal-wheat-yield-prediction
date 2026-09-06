# Models

Four model families exist in this repository. Three are wired into the yield
pipeline; one (the CNN) is a standalone experiment. This document states which
is which and gives the evidence.

---

## 1. CNN image classifiers — trained, evaluated, and NOT fused into regression

### 1.1 What they are

Five ImageNet-pretrained backbones fine-tuned as a **5-class wheat growth-stage
image classifier**. They do not predict yield.

* Architecture (identical for all five, `src/deep_learning/train_*.py`):
  `Backbone(include_top=False, weights="imagenet", trainable=False)`
  -> `GlobalAveragePooling2D(name="cnn_gap")`
  -> `Dropout(0.2)`
  -> `Dense(5, activation="softmax")`
* Optimiser Adam, lr 1e-3, batch size 16, seed 42.
* Augmentation: `RandomFlip("horizontal")`.
* Callbacks: `EarlyStopping(monitor="val_loss", patience=5,
  restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint(save_best_only=True)`.
* Configured for `--epochs 100`; **early stopping actually halted them at 14-29
  epochs** (see the `epochs` column of `reports/model_results/*_history.csv`).
  Describing them as "trained for 100 epochs" would be inaccurate — they were
  *allowed up to* 100 and converged sooner.

### 1.2 Data

`data/interim/cnn_dataset_split.csv` — 5,211 images, 5 classes, split
3,647 train / 782 validation / 782 test. Verified: **no image path appears in
more than one split**, and there are no duplicate rows.

| Class | Images |
|---|---|
| `1_Tillering` | 1,037 |
| `2_Jointing` | 1,202 |
| `3_BH` | 1,192 |
| `4_Flowering` | 1,164 |
| `5_Filling` | 616 |

### 1.3 Held-out test results (`reports/model_results/cnn_evaluation_metrics.csv`)

Computed by `evaluate_cnn_models.py` on the **test** split (782 images) — a
genuine held-out evaluation.

| Model | Accuracy | Precision | Recall | F1 | Train time (s) | Epochs run | Size |
|---|---|---|---|---|---|---|---|
| **xception** | **0.9923** | 0.9924 | 0.9923 | **0.9923** | 3,808.8 | 29 | 84.1 MB |
| densenet121 | 0.9847 | 0.9848 | 0.9847 | 0.9846 | 999.8 | 20 | 29.7 MB |
| vgg16 | 0.9731 | 0.9731 | 0.9731 | 0.9731 | 2,377.7 | 21 | 59.0 MB |
| inceptionv3 | 0.9616 | 0.9659 | 0.9616 | 0.9621 | 1,156.2 | 21 | 88.4 MB |
| mobilenetv2 | 0.9015 | 0.9245 | 0.9015 | 0.8987 | 223.6 | 14 | 9.7 MB |

**Best CNN: Xception, 99.23% test accuracy.**

### 1.4 Why they are NOT fused into the yield regression

The intended fusion path exists in code but is broken by a key mismatch:

| Table | `stage` vocabulary |
|---|---|
| `data/processed/cnn_stage_features.csv` | `1_Tillering`, `2_Jointing`, `3_BH`, `4_Flowering`, `5_Filling` |
| `data/processed/stage_features.csv` | `sowing`, `early_vegetative`, `tillering`, `stem_elongation`, `booting_heading`, `flowering`, `grain_filling_initial`, `grain_filling`, `maturity_preharvest` |

`build_stage_table.py::_merge_cnn_stage_features()` does
`stage_df.merge(cnn_stage, on="stage", how="left")`. **No key matches**, so all
6,913 `cnn_*` columns arrive 100% NaN.

Four independent artefacts already record this:

1. `stage_features_with_cnn.csv`: 6,913 `cnn_*` columns, **0 of 1485 rows** carry any value.
2. `reports/model_results/ml_selected_features.csv`: 62,217 `cnn_*` rows, all with
   `status = dropped`, `reason = "missing > 90%"`. Zero selected.
3. `reports/model_results/lstm_summary.md`: every `cnn_feat_*` listed under
   "Dropped features — missing 100.00%".
4. `reports/model_results/cnn_ablation_study.csv`: the "Weather + Satellite +
   Handcrafted Images" and "... + CNN" rows are **byte-identical** (feature_count
   31 in both, RMSE 653.3290200178633 in both).
   `cnn_improvement_report.md` states "Feature count increase: 0, RMSE improvement: 0.0".
5. Every row of every prediction CSV has `cnn_available_flag = 0`.

**Conclusion to state in any write-up:** the CNN is a *separate stage-classification
experiment*. There is no CNN-to-regression multi-modal fusion in this repository.

A second, deeper problem would remain even after fixing the key: the merge is on
`stage` alone, so a single mean-pooled 6,912-dim vector per stage would be
broadcast identically to all 33 districts and all 5 years. Being constant within
each per-stage model, it would still carry zero predictive information. Real
fusion requires per-case (district x season x stage) imagery, which this project
does not have — the image sets are external and carry no district or date link
(`case_id` and `district` are NaN throughout `image_features.csv`).

### 1.5 Artefacts (locked — read-only)

```
models/cnn/xception_best.keras       84.1 MB
models/cnn/inceptionv3_best.keras    88.4 MB
models/cnn/vgg16_best.keras          59.0 MB
models/cnn/densenet121_best.keras    29.7 MB
models/cnn/mobilenetv2_best.keras     9.7 MB
                                    ------
                                    271.0 MB
```

**These files are excluded from Git** (271 MB is far past a comfortable
repository size). They are not deleted, not modified, and not retrained. See
section 5 for distribution options.

Loading them requires **Keras 3 (TensorFlow >= 2.16)**. Verified: they fail to
deserialize under Keras 2 with
`Could not deserialize class 'Functional' because its parent module
keras.src.models.functional cannot be imported`.

Note: `models/cnn/densenet121_best.keras` (2026-06-06 14:59) is *newer* than
`data/processed/cnn_features/densenet121_features.csv` (2026-06-05 17:20), so the
stored DenseNet feature file came from an earlier DenseNet run. This is
inconsequential today because no CNN features reach any model, but it matters if
fusion is ever repaired.

---

## 2. Stage-wise regression — the primary yield model

`src/models/train_stagewise_regression.py` fits **one model per stage** (9 stages),
trying five candidates and keeping the lowest validation RMSE.

Pipeline: `ColumnTransformer(median impute numeric | most-frequent impute +
one-hot categorical)` -> regressor. Columns >90% missing are dropped.

Candidates: `RandomForestRegressor`, `GradientBoostingRegressor`,
`HistGradientBoostingRegressor`, `LGBMRegressor`, `XGBRegressor` (all
`random_state=42`).

Feature inputs come from the shared policy in `src/features/feature_policy.py`,
which blocks target-derived columns and exact-duplicate aliases. 27-34 numeric
features per stage plus 3 categoricals, depending on image-feature coverage.

Train 99 rows/stage (33 districts x 2020-21..2022-23); validation 33 rows/stage
(2023-24).

**GradientBoostingRegressor won all 9 stages** on the 2023-24 hold-out. Full grid
in `reports/model_results/stagewise_regression_metrics.csv` (45 rows). See
`docs/results.md` for the numbers.

> Under walk-forward temporal CV the ranking changes: **LightGBM** is best and
> GradientBoosting drops to 3rd of 5 — and **no model beats a district-mean
> baseline**. See `reports/summaries/research_ready_results.md` section E.

A pooled all-stage model is also fitted (`_train_combined`), 891 train / 297
validation; GradientBoosting also wins there.

Artefacts (tracked in Git, ~130 KB each):

```
models/regression/best_model_<stage>.joblib   x 9
models/regression/best_combined_model.joblib  x 1
```

**Caveat:** the winner is chosen on the validation set and then scored on the
same set, so the headline metrics are selection-optimistic. With 33 validation
cases per stage this matters. See `docs/pipeline.md` section 4.1.

---

## 3. LSTM temporal model — implemented, trained, evaluated

`src/models/train_lstm_sequence_model.py` (PyTorch).

* One sequence per case: the 9 stages in DAS order -> one final yield.
* 99 train / 33 validation / 33 forecast sequences, 9 timesteps, **76 input
  features** (from 6,954 candidates after dropping the 6,913 all-NaN CNN columns,
  `area_ha`, the target-derived `actual_available`, and 8 exact-duplicate
  aliases; district/crop/season one-hots added).
* `nn.LSTM(input_size=76, hidden_size=64, num_layers=2, dropout=0.2,
  batch_first=True)` -> `Linear(64, 1)`, on packed sequences.
* Adam lr 1e-3, batch 16, MSE loss, up to 100 epochs, early stopping patience 15.
* Imputer, scaler, district one-hot encoder and target normalisation are all
  fitted on the **training split only** — no leakage.

Validation (2023-24) metrics — `reports/model_results/lstm_metrics.csv`:

| MAE | RMSE | MAPE | R2 |
|---|---|---|---|
| 92.29 | **113.38** | 3.12% | 0.9778 |

Restored from the **best** epoch (28 of 43 run). Seeded with 42 and verified
reproducible: two consecutive runs produced byte-identical artefacts.

> Superseded LSTM results, in order: RMSE 202.51 (unseeded, pre-fix) ->
> 166.69 (unseeded, final-epoch bug) -> 148.81 (bug fixed, leaky feature still
> present) -> **113.38 (current: bug fixed + `actual_available` and duplicate
> aliases removed)**. Only the last is citable.

Artefact: `models/deep_learning/lstm_yield_model.pt` (295 KB, tracked). The
checkpoint stores the state dict plus `feature_columns`, `scaler_mean`,
`scaler_scale`, `imputer_statistics`, `target_mean`, `target_std`, and (since the
2026-09-05 fix) `best_epoch`, `best_val_rmse`, `epochs_run` and `random_seed` —
enough to run inference standalone and to audit which epoch was kept.

### 3.1 RESOLVED defect: "best weights" were not actually the best

**Fixed 2026-09-05.** Training tracked `best_val_rmse` and did:

```python
best_state = model.state_dict()      # NOT deepcopy'd
```

`state_dict()` returns tensors that alias the live parameters, so subsequent
epochs mutated the snapshot in place. The checkpoint therefore held the **final**
epoch's weights rather than the best epoch's, and early stopping's selection was
silently discarded.

Evidence from the pre-fix run (`lstm_training_log.csv`, 34 epochs): best epoch 19
at `val_rmse` 113.90, last epoch 34 at 166.69 — and `lstm_metrics.csv` reported
**166.69**, exactly the last epoch. The reported numbers were honest about the
model that was saved, but the wrong model was saved.

The fix is `best_state = copy.deepcopy(model.state_dict())`, plus
`torch.manual_seed` / NumPy / Python seeding and a seeded DataLoader generator so
the corrected run is reproducible.

**Proof the fix works, from the corrected run (55 epochs):**

| | Epoch | val_rmse |
|---|---|---|
| Best epoch (restored) | **40** | **148.8119** |
| Last epoch (previously saved by mistake) | 55 | 166.8315 |
| Reported in `lstm_metrics.csv` | — | **148.8119** |

The reported metric now matches the best epoch and **not** the last epoch; before
the fix the reverse was true.

Note the corrected run is a fresh, seeded training run, so its trajectory differs
from the unseeded pre-fix run — the old 166.69 and the new 148.81 come from
different runs and are not two evaluations of the same weights. The comparison
that matters is structural: the checkpoint now holds the epoch early stopping
chose.

Even at RMSE 148.81 the stage-wise GradientBoosting regressors remain better on
RMSE (183.11-254.42 per stage is not directly comparable; the pooled model's
129.39 is the like-for-like figure and still beats the LSTM). See
`docs/results.md`.

---

## 4. Quintile classification (Q1-Q5) — implemented, trained, evaluated

`src/models/train_quintile_classifier.py`, one classifier per stage.

* Thresholds: `pd.qcut(train_y, q=5, retbins=True)` on the **training split
  only** (`train_quintile_classifier.py:187`) — verified, no leakage.
* Labels Q1 (lowest yield) ... Q5 (highest).
* Candidates: `RandomForestClassifier`, `GradientBoostingClassifier`.
* 99 train / 33 validation per stage.

Best per stage: GradientBoosting for `sowing`, `early_vegetative`, `tillering`,
`booting_heading`, `grain_filling`; RandomForest for `stem_elongation`,
`flowering`, `grain_filling_initial`, `maturity_preharvest`.

Accuracy across stages: **0.4545 - 0.8485** (best `early_vegetative` 0.8485,
worst `sowing` RandomForest 0.4545). Full grid in
`reports/model_results/quintile_classification_metrics.csv`; per-stage confusion
matrices in `reports/model_results/confusion_matrix_<stage>.csv`.

Artefacts: `models/classification/best_classifier_<stage>.joblib` x 9 (~1 MB
each, tracked).

With 33 validation cases spread over 5 classes (~6-7 per class), these accuracy
figures have very wide confidence intervals. Treat them as indicative.

---

## 5. Artefact distribution

| Artefact | Size | In Git? | How to obtain |
|---|---|---|---|
| `models/regression/*.joblib` (10) | ~1.3 MB total | **Yes** | Cloned with the repo |
| `models/classification/*.joblib` (9) | ~9.5 MB total | **Yes** | Cloned with the repo |
| `models/deep_learning/lstm_yield_model.pt` | 294 KB | **Yes** | Cloned with the repo |
| `models/cnn/*.keras` (5) | 271 MB | **No** | See below |

For the CNN weights, pick one:

1. **Git LFS** — `git lfs track "models/cnn/*.keras"`, then commit. Note GitHub's
   free LFS quota is 1 GB storage / 1 GB bandwidth per month; 271 MB per clone
   will exhaust bandwidth quickly.
2. **A release asset or external store** (GitHub Releases, Zenodo, Google Drive,
   institutional storage) with the download link recorded here. **Recommended** —
   and a Zenodo DOI is also citable in a thesis.
3. **Retrain** — `python src/deep_learning/run_cnn_pipeline.py` (requires the
   5,211-image dataset and a GPU; ~2.5 h total for all five). Results will not be
   bit-identical.

Whichever is chosen, the expected local path is `models/cnn/<name>_best.keras`;
`extract_cnn_features.py` and `evaluate_cnn_models.py` look there and skip any
model they cannot find.

---

## 6. Environment compatibility of saved artefacts

Verified on this machine:

| Artefact | Requires | Symptom if too old |
|---|---|---|
| `*.joblib` | scikit-learn >= 1.8, NumPy >= 2.0 | `ModuleNotFoundError: No module named 'numpy._core.multiarray'` |
| `*.keras` | Keras 3 / TensorFlow >= 2.16 | `Could not deserialize class 'Functional'` |
| `*.pt` | torch >= 2.0 | — (loads under torch 2.x) |

These pins are reflected in `requirements.txt` and `requirements_colab_cnn.txt`.
