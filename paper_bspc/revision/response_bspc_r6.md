# Response to the round-6 review (BSPC mock review, Major 3/4/2/3)

**M1. Comparison with standard calibrators. Done.**
- New runs w0–w2 and y0–y1 (25 fold models). Each draw was converted in six ways: min/max, inputs clipped at 0.1/99.9, inputs clipped at 1/99, and three per-tensor estimators (99.99th percentile, MSE-optimal, KL-entropy).
- The per-tensor estimators work by recording float statistics in calibrate-only mode, replacing each tensor's range, and then running the converter's own MLIR quantizer. With unchanged ranges, this reproduces the default conversion bit-for-bit.
- Results (share of draws losing more than 0.05, for 300 / 2000 calibration windows):

  | Calibration | 300 windows | 2000 windows |
  |---|---|---|
  | min/max | 24% | 30% |
  | inputs clipped 1/99 | 0.8% | 0.4% |
  | per-tensor KL | 2.8% | 0.4% |
  | per-tensor 99.99th percentile | 5.2% | 4.8% |
  | per-tensor MSE | 18% | 17% |

- Clipping at 1/99 gave a lower per-model loss than MSE in 18 of 25 models (p ≤ 0.001). Against KL it was better in 18 of 25 models with 300 windows (p = 0.008) and 14 of 25 with 2000 windows (p = 0.31).
- Choosing the method on validation data gave a mean loss of 0.004 and 0.003.
- The contribution is reframed: input clipping is a simple step that matches KL ranges.
- AdaRound and bias correction were not run; this is stated in the Discussion and Limitations.

**M2. Restructure. Done.**
- New Table 3 lists every run family.
- Runs with the script error (s, c), the Monipar analysis and the c0 per-window firmware moved to the Supplementary material (`supplementary.tex`: Table S1, Sections S2 and S3).
- Table 4 now lists only runs of the specified network.

**M3. Dependent units. Done.**
- The Brown–Forsythe test over 50 units (runs sharing seeds and folds) was removed and replaced with per-run counts.
- Results stating that z and r runs share folds and seeds and are repeats.
- The clipping effect is reported separately for the zero-phase group (13 of 15 models) and the causal group (10 and 9 of 10).
- The text also states that w0–w2 and y0 reproduce r0–r2 and u0 exactly, while y1 is a new set of models.

**M4. Deployment motivation. Partly done.** The Discussion explains the per-window, single-wrist rationale and states that it was not tested. Single-wrist analyses were not run.

**M5. Code and analysis plan during review. Not done (authors' decision).** Both will be released when the article is published. The project is kept in a private repository until then.

**M6. Whole-fold validation of the deployable causal model. Done.**
- Fold 0 of run k0 was replayed on the ESP32-S3 with causally filtered input.
- Identical outputs: 8371/8378 windows (QAT) and 8372/8378 (PTQ).
- Device AUROC 0.884 and 0.908, identical to the desktop. Latency 10.68 ms.
- Added to Table 8. The text states that host-side causal filtering was used, and that its equivalence with the firmware filter rests on the 27-window streaming test.

**Minor inconsistencies. Fixed.**
- Table 2 caption now notes the separate benchmark run.
- The abstract compares the same 25 models.
- CLE figures are labelled by model set.
