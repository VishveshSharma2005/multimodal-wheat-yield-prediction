# Stage Definitions (Wheat) — Simple Options

You can define stages in two beginner-friendly ways. Choose one and stay consistent.

---

## Option A: Calendar-based stages (easy)
Use days after sowing (DAS). Example only (adjust locally):
- Sowing: DAS 0
- Emergence: DAS 7–14
- Tillering: DAS 20–35
- Stem elongation: DAS 35–55
- Booting/Heading: DAS 55–75
- Flowering: DAS 75–90
- Grain filling: DAS 90–115
- Pre-harvest: DAS 115–130

Pros: easy, no extra computation.
Cons: seasons differ; heat/cold shifts stages.

---

## Option B: GDD-based stages (better science)
Compute cumulative GDD from sowing and use thresholds.

You must choose:
- Base temperature $T_{base}$ (wheat often ~0–5°C, follow your advisor)

Daily:
- $T_{mean}=(T_{max}+T_{min})/2$
- $GDD=\max(0, T_{mean}-T_{base})$

Then define stage ends by GDD thresholds (example placeholders):
- Emergence end: `[__]` GDD
- Tillering end: `[__]` GDD
- Stem elongation end: `[__]` GDD
- Heading end: `[__]` GDD
- Flowering end: `[__]` GDD
- Grain filling end: `[__]` GDD
- Pre-harvest end: `[__]` GDD

Pros: consistent across years.
Cons: needs daily temperature data.
