# Response to the round-4 review (BSPC mock review, Major 3/3/3/3)

The R4 text is in main.tex (backup main_r4.tex). The new runs are Kaggle p56t0–t2 (zero-phase, seeds 0–2) and p56u0–u1 (causal, seeds 0–1): 25 fold models in total. Each fold model was converted with 10 draws × 2 calibration sizes × 4 calibration forms (min/max, and inputs clipped at 0.01/99.99, 0.1/99.9 or 1/99).

**M1. Clipping on all folds, percentile sweep, level chosen on validation. Done.**
- The results are in the new Table 7 and Section 4.5.
- Min/max calibration: 22% and 30% of draws lost more than 0.05 AUROC (300 and 2000 windows), with the worst draws losing 0.220 and 0.327.
- Clipping at 1/99: 1.2% and 0.4% of draws lost more than 0.05, and the worst draws lost 0.065 and 0.089.
- Choosing the clipping level on validation picked 1/99 in 33 of 50 choices. Mean loss was 0.004 and the worst model lost 0.065.
- Clipping at 0.1/99.9 against min/max: the per-model mean loss improved in 22 of 25 models (Wilcoxon p < 0.001), and the improvement was positive in all 5 runs.

**M2. Comparison with standard calibrators. Partly done.**
- Added the baseline of choosing the best min/max draw on validation. It helps with 300 windows (4% of models lose more than 0.05) but not with 2000 windows (12%).
- Per-layer percentile, entropy or MSE ranges and AdaRound were not run. This is stated in the Discussion and Limitations.

**M3. Mechanism. Done.** The internal-activation explanation is now stated as a hypothesis, with the reason it was not measured.

**M4. Evaluation noise. Done.**
- Draw-level gains of up to 0.066 AUROC are reported, as is the roughly 0.04 spread in FP32 AUROC between retrainings.
- With about 16 controls per test fold, a single-draw difference below about 0.05 is within evaluation noise. This is why the main statistic is the share of draws losing more than 0.05.
- Decision-level results added: balanced accuracy, and agreement with FP32 decisions (0.82 rising to 0.87).

**M5. Claims beyond the CIs. Done.**
- Filter mismatch: "tended to", "suggested but not established" (Conclusions, highlight 5, abstract).
- QAT: "usually fewer large losses, not in every run (z0)".
- Equivalence under shift: shown for a single conversion. (The 16×8 and clipping results give no reason to expect otherwise.)

**M6. Non-independence of units. Done.** Improvements are reported per run (positive in all 5), alongside the model-level Wilcoxon test. The 5 new runs are a third independent set of retrainings, after z and r.

**M7. Code and plan before acceptance. Not done (authors' decision).** Everything will be released on acceptance, as the data and code availability statement says.

**M8. Justifying the network over the RF. Done.** The Discussion now explains that the network gives a score for each 4-s window from one wrist, whereas the RF needs all 22 recordings. It states this advantage was not tested, and narrows the generalization to "one small network and one converter".

**Numeric fixes.**
- H2/H3: now described as two-sided bootstrap p values.
- CIs: equivalence results use 90% CIs; all other CIs are 95%.
- Latency ratio: 49× in a matched benchmark (562.5 ms against 11.6 ms).
- Input-scale gain: "three to five times".
- Abstract: 239 words.
