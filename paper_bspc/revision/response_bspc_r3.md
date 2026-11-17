# Response to the round-3 review (BSPC, mock review)

Reviewer recommendation: Major revision (scores 3/3/3/3). The changes below are in main.tex (R3); the R2 text is kept as main_r2.tex.

## Major comments

**M1. PTQ mechanism and fixes. Done (a, c), partly done (b).**
- New experiment: runs q0 and q1 (Kaggle p56q0 and p56q1). Fold 0 was retrained five times for each of seeds 0 and 1, giving ten models. Each model got 10 pairs of 300- and 2000-window calibration draws, and each draw was converted twice: as drawn (default min/max) and with every input channel clipped to the 0.1/99.9 percentiles of normalized training windows. For each int8 model we recorded the input and output scales and the fraction of test input values that saturate.
- Saturation-underestimation hypothesis: rejected. With min/max calibration, at most 0.005% of test input values saturated.
- Loss versus range: across draws, the loss did not correlate with the input scale (Spearman −0.03 for 300 windows, +0.14 for 2000) or with the output scale.
- Clipping fixed it:

  | Calibration | Mean loss | Draws losing > 0.05 | Worst draw |
  |---|---|---|---|
  | 300 windows, min/max | 0.023 | 15% | 0.207 |
  | 300 windows, clipped | 0.010 | 2% | 0.059 |
  | 2000 windows, min/max | 0.051 | 41% | 0.317 |
  | 2000 windows, clipped | 0.009 | 2% | 0.057 |

  The worst draw of every model stayed within 0.06. Per-model paired comparison: 2000 windows 9/10 models better (Wilcoxon p = 0.006); 300 windows 7/10 (p = 0.19).
- Changes in the paper: Methods (new paragraph), Section 4.5 (new paragraph), Discussion (mechanism, recommendation), Abstract, highlight 3, Conclusions and Limitations. The recommendation is now "clipped calibration ranges or QAT, checked across draws".
- (b) Full decomposition: not done. The 16×8 variant (int16 activations, int8 weights) isolates activation quantization and removes the loss. The input is not saturated and the input scale does not predict the loss, so input quantization is not the main factor. We did not build float-input / int8-internal variants.

**M2. Overclaims. Done.**
- Abstract and Platt: "agreed" has become "close" in Section 4.3.
- Bit-exact count: "all but four or five windows".
- Filter mismatch is now worded "tended to lose", and the CI is given in the abstract and Results. Results states that all CIs include zero.
- The QAT spread is described as not significant for the specified network (Discussion).
- The abstract recommendation is now based on the probability that a draw loses more than 0.05.

**M3. Run IDs. Done.**
- Table 4 has a Run column with labels a, z0–z2, r0–r2, k0–k2, s1–s2 and c0–c2, plus a note explaining r0–r2.
- Every relevant caption now names its runs: Table 5 (r0–r2, k0–k2), Table 6 (r0–r2), Figure 2 (r0–r2), Figure 3 (r0–r2), Table 7 (a, z0), and q0/q1 in Section 4.5.
- (c) The summary for z0–z2 has not been added as a supplement. Table 4 already gives every run.

**M4. End-to-end validation. Partly done.**
- The streaming firmware was re-run with the fold-0 QAT model of the specified causal run k0. Result: 27/27 windows bit-exact with the causal Python pipeline; 12.6 ms per 2-s step (inference 10.9 ms); 40.3 kB RAM.
- The per-window end-to-end firmware still uses c0. This is stated, together with the scope sentence: "validates the on-device preprocessing chain and the integer inference; on-device accuracy over a whole fold was measured only for run z0".
- The 16×8 mismatch (46/50) is explained in the Table 7 note: the four outputs differ by one step of the int16 output scale, from fixed-point rounding in the reference kernels.

**M5. Code and plan. Not done; this is the authors' decision.**
- The code, fold assignments, the dated plan and the amendment log will be released when the paper is accepted, as the Data and code availability section states.
- The external-cohort (Monipar) result is now reported in Deviations (ii) with its CIs: 21 PD, 7 HC, exploratory, no-BN network.

## Minor comments
1. Latency ratio: corrected to 53× throughout.
2. 12.2 ms versus 10.7 ms: stated that the build and model differ and that the difference was not decomposed. The streaming result now uses k0 (10.9 ms inference).
3. Sidedness: H2 and H3 are one-sided in the stated direction, with two-sided percentile bootstrap CIs (Methods). TOST is equivalent to the 90% CI lying inside the margin.
4. H3 confounding: one sentence added to Methods.
5. Pooled versus within-fold: restricted to PD versus HC; feature models corrected to 0.003–0.005.
6. 13-shift mean: reported (equivalence p = 0.032 and 0.004).
7. 90% CIs reported, and the margin is justified with its limitation stated.
8. ECE: commented on. Fixed-threshold agreement: not computed. The caption now states that thresholds are refitted for each variant.
9. Deployed RF: AUROC 0.923 in fold 0 and 0.898 across folds.
10. Varghese: balanced accuracy 91.2% and 72.4% reported, with why the numbers are not comparable.
11. Dilation details: not added.
12. Receptive field: 127 samples in both designs.
13. and 14. Table 1 and Table 2 details: not changed.
15. Figure 1: fonts enlarged; the figure is now full width.
16. Figure 2: shared x-axis. Absolute FP32 AUROC values are kept in the text.
17. Figure 3: the original conversion is marked (×), and the mean of the 300-window draws is added.
18. Per-sex CIs: not added.
19. Introduction: one sentence on the one-wrist window model with two-wrist, 11-task aggregation.
20. AI use: described in Methods.
21. Density: partly addressed.

References: early-access JBHI entries are fixed; the PADS dataset entry is marked [dataset] and its capitalisation is corrected.
