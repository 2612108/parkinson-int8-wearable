# Response to the mock review (BSPC, round 1)

Recommendation received: Major revision. Summary of changes and point-by-point replies.

## Major comments

**M1 – Confirmatory analysis.** Removed "This is the primary result". The abstract, highlights and conclusions now say that the planned confirmatory evidence is inconclusive. Added a fold-normalized sensitivity analysis (Table 3): H1 is not shown in any seed, H2 remains null, and H3 is not supported in any seed after Holm adjustment. The title no longer says "pre-specified". The cross-seed rule is stated: seed 0 was primary and seeds 1 and 2 were replications, with no rule set across seeds. The dated plan and its amendment log will be released with the code upon acceptance.

**Disclosure.** While answering M4 we found a script error. Seeds 1 and 2, and all early exploratory runs, had lost the batch normalization after the depthwise convolutions. All runs were repeated with the specified architecture (three seeds × two filters). Every run is reported in Table 4, and the error is described in Section 5.1.

**M2 – Pooling.** The text now says only that fold models produce scores on different scales, and no longer attributes this to temperature scaling alone. The pooling problem is placed in the literature (Forman & Scholz 2010; Airola et al. 2011). We report pooled, fold-normalized and within-fold AUROC, the fold-level SD (Table 2) and per-unit dot plots (Fig. 2). Rank normalization reproduces the within-fold mean.

**M3 – Mechanism.** The quantization variants (2000-window calibration, PTQ of the fine-tuned model, 16x8) were re-run in both filter settings. Added a float-stem/input variant to separate input quantization, plus layer-wise errors and per-channel range statistics. Main changes in the claims:
- The claim about depthwise channel ranges was dropped.
- The new conclusion is: activation rounding accumulated across layers, together with range setting during calibration. A larger calibration set worsened PTQ in every run in which it was tested.
- Input quantization alone is not the cause.
- Range-adjustment methods (DFQ, AdaRound) were not tested and are discussed as a limitation.
- The PTQ loss after sensor randomization is now discussed.

**M4 – FP32 reference.** The within-fold FP32 AUROC agrees with the benchmark (0.879 versus 0.882). The 0.762 figure was a pooling artefact: pooled FP32 ranged from 0.700 to 0.816 across six runs, whereas within-fold AUROC ranged from 0.849 to 0.881. Seed-to-seed variation is reported.

**M5 – RF versus network.** The single-window/single-wrist advantage claim was removed. The text now states that it was not tested and that the random forest remained more accurate and also runs on the device.

**M6 – Statistics.**
- The repeated k-fold correction (Bouckaert & Frank 2004) is stated with its ratio and degrees of freedom.
- Exploratory p values are declared descriptive, and "equivalent" was removed.
- Table 2 now uses seed-0 point estimates with matching CIs.
- Reuse of the validation set is acknowledged.
- The one-seed generalizations were dropped.

**M7 – Confounding and framing.** Added a within-sex AUROC and an age + sex ± score analysis (Section 4.5). The paper is framed as "detection in a referred population" from the Data section onwards, and "screening" was removed.

**M8 – Shift.** Changed the keyword. Explained the 25-Hz condition. Stated which device differences the simulation does not model. Moved Monipar to Appendix A and noted the recording-setting confound.

**M9 – Reproducibility.** Added the details:
- sensor-randomization parameters;
- baseline architectures and grids;
- windows per participant and the imputation method;
- TF/TFMOT versions and the QAT scheme;
- the composition of the calibration set;
- the fact that the seed sets both folds and initialization.

Code, folds, firmware and the plan will be released in a public repository upon acceptance.

**M10 – Device.**
- (a) Bit-exactness on whole folds was not added. It is listed as a limitation, and the inference argument is stated.
- (b) The random forest was re-estimated with causal features: 0.881 when trained zero-phase, 0.900 when trained causal (0.898 with matched zero-phase features).
- (c) Energy claims were removed.
- New: the 16x8 model was measured on the board at 562 ms per window (48× int8), with 45/50 identical outputs and the rest within 1 LSB.

## Minor comments
1. Abstract rewritten (232 words).
2. Highlights rewritten (≤ 82 characters each).
3. Changed to "among the fastest-growing".
4. Preprints flagged.
5. Table 1 cell now reads "descriptive only".
6. Dilation finding kept.
7. Shared frame stated.
8. Kept as is (window-level early stopping is the training objective).
9. Agreement discussed (85–98%; fine-tuned FP32 91%).
10. Table restructured: the reference is given in footnotes.
11. 14.6 ms = 2.4 + 12.2 (normalization, input quantization, inference).
12. The extrapolated 0.23-s assessment time was removed; only measured per-recording times are reported.
13. Fig. 2 redrawn as a per-unit difference plot; the per-shift figure states the CI method.
14. Pipeline caption updated.
15. The 50-Hz shift is noted where H2/H3 are defined.
16. `arena_used_bytes` stated.
17. Gyroscope-removal numbers given.
18. Stride deviation expanded.
19. AI declaration now describes how outputs were verified.
20. Access terms stated.
21. References fixed: peer-reviewed versions, capitalization, et al., early access.
22. Each Results subsection now opens with its finding, and settings are labelled.
