# Response to the mock review (BSPC, round 2)

New experiments (Kaggle runs p56r0–r2 and p56rr; board runs B6 and B5bn):
- 10 calibration draws (300 and 2000 windows) for every trained model of three seeds.
- PTQ after BN folding and cross-layer equalization (CLE).
- Zero-phase-trained models evaluated on causally filtered input in all folds.
- Storage of temperatures, thresholds and validation logits, used for the Platt-scaling analysis.
- Five retrainings of fold 0.
- Whole-fold replay on the ESP32-S3 (8378 windows).
- 16x8 latency re-measured with the specified network.

**M1 Pooling.**
- Temperatures ranged from 0.12 to the search bound of 20 (reached in 4 of 15 folds), and the median raw logits per fold ranged from −0.86 to +1.99.
- Platt scaling with an intercept brought pooled AUROC back to the within-fold level (FP32 0.832–0.880 vs within-fold 0.856–0.885).
- The plan is now stated to have been mis-specified.
- ECE is reported (Table 6).
- The highlight about pooling has been reframed.

**M2 Run versus calibration variance.**
- Across 150 draws on 15 models, the SD within a model (0.033) exceeded the SD of the per-model means (0.020).
- In 5 retrainings of fold 0, single calibration draws changed AUROC by −0.106 to +0.028.
- The specified network and the BN-less runs are now reported separately everywhere (Table 4).

**M3 Calibrator.**
- Claims are now restricted to TFLite's default min/max calibration, in the title, abstract, discussion and limitations.
- CLE was added and did not help (mean −0.033; worst −0.248).
- Percentile, entropy and MSE calibration and AdaRound are discussed as the next step (Wu 2020; Nagel 2020).
- The "every run" claim for 2000-window calibration has been removed; the averages are given instead.

**M4 Device.**
- A complete fold was replayed: 8373/8378 windows were identical, and the participant AUROC on the device equalled the desktop value (0.889 QAT, 0.856 PTQ).
- Filter mismatch is now evaluated in all folds and seeds with corrected-t CIs (QAT −0.044, FP32 −0.020). The random-forest comparison is evaluated in the same way (−0.017).
- The two-wrist, 11-task requirement is stated explicitly.

**M5 Nulls.**
- H2 is now supported by an equivalence test within ±0.02 (p = 0.023 zero-phase, p = 0.002 causal).
- CIs are labelled as unadjusted.
- The 16x8 equivalence test is done per architecture and filter with the corrected t.
- Variance comparisons use Brown–Forsythe; it is not significant for the specified network (p = 0.16), and the text now says so.
- The claim about the float-stem mechanism was dropped.

**M6 Benchmark.**
- Paired within-fold differences are reported (RF +0.024, CI −0.029 to 0.076; XGB +0.036, CI −0.001 to 0.074), and the wording is now "at least as accurate".
- The pooled-gap statement was corrected.
- The networks still use a single seed; this is listed as a limitation.

**M7 Deviations.**
- The "During revision" wording was removed.
- The Monipar appendix (BN-less, only 7 HC) is no longer reported.
- The streaming firmwares are disclosed as using the BN-less model.
- The 16x8 latency was re-measured with the specified network.
- The analysis plan and code will be released on acceptance (decision of the authors).

**M8 Confounding.** Age by group has been added; analyses within each sex are reported.

**M9 Decision metrics.** Table 6 reports BA, sensitivity, specificity, ECE and agreement. Agreement of a fine-tuned FP32 model (91%) separates the effect of threshold refitting.

**Minor comments.**
- The abstract is 248 words.
- Highlights were rewritten (69–82 characters).
- The real-time duty fraction was corrected to 0.5% of the 2-s step.
- "about 49 times" is used.
- Tables were set without resizebox.
- Figures were redrawn: the per-shift differences now include FP32+SR and the causal filter, the calibration draws have their own figure, and the pipeline figure shows the fold structure.
- Model sizes are now given as flatbuffer sizes.
- GBD is cited as a group author.
- References were added: Wu 2020, AdaRound, Bouthillier 2021, Varoquaux 2018 and Fawcett 2006.
- The AI declaration now describes independent recomputation.
