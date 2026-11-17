# Response to the round-5 review (BSPC mock review, Major 2/3/3/2)

**M1. Originality and comparison with standard calibrators. Reframed; comparison not run.**
- Added Banner et al. 2019 (ACIQ) and Zhao et al. 2019 (OCS).
- The Discussion now says plainly that the outlier effect and the clipping remedy are already known. The contribution is to measure how large and how variable the effect is for a small clinical wearable network evaluated per participant, and to provide a label-free screen.
- Per-layer MSE ranges and AdaRound are not part of the default TFLite converter and were not compared. This is stated in Limitations.

**M2. Mechanism. Done.**
- New runs v0 and v1 (seeds 2 and 3): fold 0 retrained 5 times per seed, 10 draws × 3 calibration forms per model.
- Activation scales: min/max calibration gave scales 1.6–2.2× larger than clipped calibration in the convolutional and residual layers, 1.4× after pooling, 1.2× at the output and 5.5× at the input.
- Within a model, larger scales went with larger losses in every layer (ρ = −0.16 to −0.36). The link was strongest in the first pointwise convolutions and the first residual additions, and weakest at the output.
- The mechanism is therefore now shown, not hypothesized, and it acts across the network rather than in one layer.

**M3. Label-free fidelity and window-level AUROC. Done.**
- Window-level AUROC, computed on about 8400 test windows: min/max calibration lost 0.049 (300 windows) and 0.068 (2000 windows); clipping at 1/99 lost 0.012 and 0.013.
- Window decision agreement with FP32: 79% and 71% with min/max calibration, 91% and 90% with clipping at 1/99.
- Within a model, logit RMSE correlated with the participant-level loss (ρ = −0.40), and window-level and participant-level losses correlated (ρ = +0.51). The participant-level losses are therefore not artefacts of the 16 test controls.
- The Discussion recommends a label-free logit-agreement check.

**M4. t runs matching r runs. Explained and corrected.**
- Training on Kaggle CPUs is deterministic for a fixed script and seed, so t0–t2 reproduce r0–r2 exactly (verified: largest difference over 960 results is 0). Nothing was copied.
- Methods now states this. The zero-phase and causal groups are reported as the two independent groups.
- The sentence "training was not deterministic" was removed.

**M5. Exploratory status. Done in the abstract.** The abstract now says that the three planned tests were inconclusive and that the remaining results are exploratory. The dated plan will be released on acceptance (authors' decision).

**M6. Single-wrist use case. Not run.** It is stated as untested in the Discussion and Limitations.

**M7. Thresholds based on about 13 validation controls. Partly addressed.** The new label-free and window-level metrics need no threshold. The participant-level decision metrics still use thresholds refitted for each variant, and this is stated.
