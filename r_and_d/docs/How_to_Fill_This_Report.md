# How to Fill This Report (Beginner Guide)

This guide tells you *what to write* in each section of the main report, even if your dataset is small.

Main template: `docs/Field-Season_Case-Study_Report.md`

---

## 1) If you only have ONE field-season
You can still write a strong case study.
- Your “model results” will be limited (not enough data to generalize).
- Focus on: data pipeline, temporal features, weed correction, and stage-wise prediction logic.

Write clearly in Limitations: “single case, results are illustrative”.

---

## 2) How to fill each section quickly

### Section 0: Quick Summary
Write 5–8 bullets:
- where the farm is
- what you predicted (yield kg/ha + optional quintile)
- what data sources you used
- what model you tried first (baseline)
- one sentence about best stage

### Section 2: Farm Profile
Keep it simple:
- irrigation type
- soil report summary (even 3–5 numbers)
- field boundary source

### Section 3: Timeline
This is very important.
If you don’t know exact stage dates, write approximate dates based on:
- your farm visit notes
- visible satellite/drone changes
- days after sowing

### Section 4: Data Sources
For each source, write:
- where it came from
- how often you collected it
- what variables you extracted

### Section 5: Yield Ground Truth
Be specific:
- harvested area
- weight measurement method
- moisture correction method (or “not corrected”)

### Section 6: Preprocessing
Write 3 rules you followed, for example:
- all features computed only up to the stage date
- cloud-masked satellite points removed
- missing satellite values interpolated linearly (or left missing)

### Section 7: Weed correction
Even a simple approach is acceptable if described honestly.
Minimum outputs per date:
- weed_cover_%
- ndvi_crop_only

### Section 8: Features
Explain “crop state features” in plain language:
- level: NDVI today
- slope: NDVI increasing or decreasing
- stress: canopy temperature anomaly

### Sections 9–11: Models + Experiments
Start with one baseline model per stage.
Then add ablations:
- without NDVI
- without weed correction

### Section 10: Evaluation
If you have few cases, you can still do:
- leave-one-field-out (if multiple fields)
- or time split (if multiple seasons)

If you cannot split properly, write it as a limitation and report training/validation carefully.

---

## 3) What your professor usually wants to see
- Stage-wise setup + “no leakage” statement
- Crop state included (trajectory features, not a snapshot)
- Weed correction explained and compared
- Clear evaluation protocol and ablations

---

## 4) What not to do
- Don’t compute quintiles using test yields
- Don’t use late-season images when predicting early stages
- Don’t mix yield measurement methods across cases without noting it
