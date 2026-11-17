# Referee report

**Journal:** Biomedical Signal Processing and Control (Elsevier)
**Manuscript:** "Eight-bit quantization of tiny wrist-inertial networks for Parkinson's disease detection: an empirical evaluation with microcontroller validation"
**Files reviewed:** main.pdf (25 pp., line-numbered), main.tex, highlights.txt, Figure_1–3.pdf

Line numbers below refer to the line numbers printed in main.pdf; section/table numbers follow the compiled PDF.

---

## 1. Summary

The authors study how eight-bit quantization affects a 22.7k-parameter depthwise-separable temporal convolutional network (DS-TCN) that separates Parkinson's disease (PD, n = 276) from healthy controls (HC, n = 79) using bilateral Apple Watch inertial data from the public PADS dataset, with participant-level nested 5-fold cross-validation, three seeds, and 14 simulated sensor shifts. Three hypotheses (PTQ non-inferior to FP32; quantization does not interact with shift; QAT + sensor randomization beats PTQ under shift) were fixed in a dated plan and tested on fold-pooled predictions; the results were inconsistent across seeds and vanished after per-fold rank normalization, which the authors attribute to instability of fold-pooled AUROC. In exploratory within-fold analyses over twelve runs (five of which used an unintended architecture without depthwise batch normalization), the PTQ loss varied from 0.000 to 0.087 AUROC, was larger with a 2000-window calibration set, and disappeared with 16-bit activations; QAT varied less. On an ESP32-S3, int8 inference was bit-exact on 50 windows and took 11.6 ms per 4-s window (16×8: 562 ms), a zero-phase-trained network lost 0.074 AUROC on one fold when fed causally filtered input, and a feature-based random forest was both more accurate and deployable.

The manuscript is unusually transparent (planned vs exploratory labelling, deviation log, reporting of a script bug and of negative results), and the on-device engineering is careful. However, the confirmatory part failed by the authors' own account, the central exploratory claim ("PTQ is unreliable across retraining runs") rests largely on one outlier run plus runs of a different architecture, several mechanistic and device-level claims rest on a single fold or a few windows, and the "pooling instability" finding appears to be at least partly a consequence of the authors' own calibration protocol rather than a property of cross-validation in general.

---

## 2. Recommendation: **Major revision**

Justification against BSPC's standard. The topic (embedded inference for wearable inertial signals, preprocessing matched to the device, evaluation methodology) is within BSPC scope, and BSPC does publish careful empirical/negative studies. The work is not fatally flawed: the data are public, the splits are participant-level, the leakage analysis is useful, and the device measurements are real. But at present (a) the claims in the abstract, highlights and conclusions are stronger than the evidence (see M1–M5), (b) the key confound between training-run variance and calibration-sample variance in PTQ has not been separated (M2), (c) the main methodological "lesson" on pooling is presented as a general finding when it is plausibly produced by temperature-only calibration on ~13 validation controls per fold (M1), and (d) PTQ was evaluated only with TensorFlow Lite's default min/max calibrator, so "PTQ is unreliable" is really "default TFLite PTQ is unreliable for this network" (M3). These issues are fixable with additional experiments that are cheap relative to what has already been done (no new data needed). I would not accept the paper in its current form, and I would lean towards rejection if the revision does not address M1–M4 with new analyses rather than rewording.

---

## 3. Major comments

**M1. The "pooling is unstable" finding is plausibly an artifact of the calibration protocol, and the large pooled-vs-within-fold gap is unexplained.**
(Section 4.3, lines ~253–262; Discussion lines ~345–352; abstract; highlight 2.)
- In the benchmark (Table 2, seed 0) the DS-TCN had pooled AUROC 0.856 vs within-fold 0.882 (gap 0.026). In the six "corrected" quantization runs, the same FP32 architecture had pooled 0.700–0.816 vs within-fold 0.849–0.881, i.e. gaps of up to ~0.17. The manuscript does not explain why the same model/protocol produced a ~5× larger pooling penalty in the quantization runs than in the benchmark. Please state exactly what differs between the benchmark pipeline and the quantization pipeline (temperature fitting, class weighting, normalization, early-stopping data, architecture/BN status of the benchmark model, software versions).
- Temperature scaling only rescales logits (no bias term). With class-weighted BCE, each fold model's logits carry a fold-specific offset that temperature scaling cannot remove; with ~57 validation participants (~13 HC) per fold the fitted temperature is itself noisy. Pooling such scores will necessarily be unstable. This is a property of the chosen post-processing, not an intrinsic property of cross-validated AUROC. Required: (i) report per-fold logit location/scale (e.g., median logit of HC and PD per fold) for FP32 and PTQ; (ii) repeat the pooled analysis after Platt scaling (with intercept) or after centering on the fold threshold, and show whether pooled ≈ within-fold then; (iii) report ECE, which the Methods promise (line ~193) but which is never reported anywhere.
- The phenomenon is well known (Forman & Scholz 2010; Airola et al. 2011, both cited) — the manuscript should state clearly that the novelty is limited to "this happened to a pre-specified analysis plan", and the highlight "Pooled cross-validated AUROC varied by 0.12 across runs" should be reframed accordingly. It is not a finding about quantization.
- Consequence for the planned analysis: the authors chose a primary estimand (pooled AUROC with per-fold, per-variant temperature) that is known to be fragile. The honest conclusion is that the plan was mis-specified, not that "the planned confirmatory evidence is inconclusive" in a neutral sense. Please say so explicitly in the Discussion.

**M2. "PTQ loss depends on the training run" is not separated from calibration-sample variance and is driven by one outlier run and by runs of a different architecture.**
(Section 4.4, lines ~263–287; Table 4; abstract "over twelve runs ... 0.000 to 0.087"; highlight 3; Conclusions.)
- PTQ is calibrated on 300 randomly drawn windows. The variance attributed to "the run" therefore mixes training non-determinism and calibration-sample randomness. Required: for each saved FP32 fold model (at minimum the planned seed-0 run with the −0.219 fold), re-convert with ≥10 independent calibration draws (300 and 2000 windows) and report the distribution of PTQ loss per model. If the −0.219 fold can be reproduced (or not) by redrawing the calibration set from the *same* weights, that changes the interpretation entirely.
- Of the four runs with mean PTQ loss > 0.03, three are runs without depthwise BN (a different network, 21.8k parameters, produced by a script error) and one is the planned seed-0 run whose own repeat lost only 0.010. In the six runs of the specified architecture, mean PTQ loss was 0.000–0.028 (mean 0.013 zero-phase, 0.017 causal; CIs include zero; Table 5). The abstract, highlight 3 ("ranged from none to 0.087 across repeated training runs") and Conclusions ("varied from none to large between runs of the same configuration") overstate this. The phrase "same configuration" is incorrect for the pooled set of twelve runs. Required: separate the specified-architecture runs from the BN-less runs in all summary statements, abstract and highlights; treat BN-less runs as a distinct (and possibly informative: absence of BN after depthwise convs changes activation ranges) condition rather than as replications.
- One seed per configuration was rerun once (planned vs repeat). Run-to-run variance should be estimated with more repeats of an identical configuration (e.g., ≥5 retrainings of fold 0 with fixed fold assignment and different initialisations), otherwise "PTQ is unreliable across retraining runs" rests on n = 2.

**M3. PTQ was evaluated only with TensorFlow Lite's default (min/max) calibrator; standard remedies were not tested.**
(Methods lines ~159–172; Discussion lines ~355–372; Limitations.)
- The observation that a larger calibration set makes PTQ worse is the expected behaviour of min/max range estimation (more samples → more extreme ranges → coarser grid) and has a standard remedy (percentile/MSE/KL range setting). Channel range ratios up to 8.4 within a layer also point to well-known fixes (cross-layer equalization, bias correction, AdaRound, per-channel activation handling). The paper's practical conclusion ("PTQ unreliable; prefer QAT") is therefore conditional on the least robust PTQ variant. Required: add at least percentile (e.g., 99.9/99.99) or MSE-based activation ranges and one of CLE/bias correction/AdaRound (available in AIMET, or implementable for this small network), re-run on the six specified-architecture runs, and report whether the run-to-run variance remains. If these are not added, the title/abstract/conclusions must explicitly restrict claims to default min/max TFLite PTQ.
- Note also the text claim "Calibrating on 2000 instead of 300 windows made PTQ worse in every run in which it was tested" (line ~270) and "consistently increased the loss" (line ~360) are contradicted by Table 4: in the BN-less causal runs with seeds 0 and 2, the 2000-window calibration was *better* (−0.021 vs −0.030; −0.016 vs −0.037). The claim holds only for the six specified-architecture runs. Please correct.

**M4. Device-level claims rest on a single fold and a handful of windows; the filter-mismatch comparison is not like-for-like.**
(Section 4.6, lines ~320–342; abstract; highlight 5; Conclusions.)
- The −0.074 AUROC loss for a zero-phase-trained network fed causally filtered input is from one model on the 71 fold-0 test participants (~16 HC), without a CI. It is then compared with a −0.017 loss for the random forest computed over full cross-validation (0.898 → 0.881) and with "0.900" for a causally trained forest, without stating whether these are pooled or within-fold, which forest (500 trees benchmark vs 100-tree/depth-8 deployed), or how many seeds. Required: run the filter-mismatch experiment for all folds and seeds of the network (the causal pipeline already exists in Python), report within-fold mean difference with corrected-t CIs, and do the same for the random forest under identical evaluation. Then compare the two mismatch penalties with a paired analysis.
- "Accurate on-device inference required training with the device's causal filter" (highlight 5, abstract, Conclusions) is a strong claim from one fold; also, the causal FP32 model is itself slightly worse than the zero-phase one (0.860 vs 0.872, Table 5). Please quantify the causal-trained vs zero-phase-trained gap on causal input across all folds.
- Bit-exactness was checked on 50, 8 and 27 windows. Given that the network only has ~22k parameters and inference takes 11.6 ms, it is cheap to replay an entire test fold (≈8,000 windows) on the board (or at least one full fold per variant) and report participant-level on-device AUROC directly. This would turn "inferred" accuracy into measured accuracy and justify the "microcontroller validation" in the title.
- Practical deployability: both the network's participant decision and the random forest's 2112-feature vector require both wrists and all 11 tasks, i.e. data from two separate watches. A single wrist-worn MCU cannot produce either decision. The manuscript acknowledges this only in passing (lines ~392–396). Either evaluate a single-wrist (and ideally reduced-task) configuration for both the network and the feature model, or remove/soften the framing that this is a "wrist-worn classifier" running on-device.

**M5. Over-interpretation of null hypothesis tests and of exploratory statistics.**
- H2 was a superiority test ("quantization increases the loss under shift"). Failure to reject is not evidence of no interaction, yet the abstract states "quantization did not add loss under shift" and line ~287 says "The planned finding that quantization does not interact with shift (H2) held in every analysis". Required: report an equivalence/TOST analysis with a stated margin (the CIs in Table 3 suggest equivalence at ±0.02 may well hold in the fold-normalized analysis) or reword as "no evidence of an interaction".
- Table 3, fold-normalized H3 seed 0: 95% CI (+0.003, +0.117) excludes 0 but p_Holm = 0.14. State that CIs are unadjusted, or give Holm-consistent (e.g., 1−α/m) intervals, to avoid confusing readers.
- The 16×8 TOST "p < 0.001 over 45 units" treats 45 fold–run units from nine runs (including BN-less runs) as the sample; the units are not independent (shared folds within a seed, overlapping training sets). Use the corrected-t framework consistently, per architecture.
- Counts such as "PTQ lost >0.05 in 18 of 60 fold-run units vs QAT in 2" mix architectures and include runs with no corrected-architecture replication. Report these per architecture and give a formal comparison of variances (e.g., Levene/Brown–Forsythe on fold-level differences, or a mixed model with run as random effect).
- Single-fold differences with ~16 HC per fold are noisy: the float-stem experiment (lines ~295–301) shows the float-stem variant on average *worse* than full int8 PTQ (mean −0.0185 vs −0.0108), which the authors interpret as "did not remove the loss". With six fold values this experiment cannot discriminate hypotheses; please temper the mechanistic conclusion ("the loss therefore accumulated from activation rounding across the network") or support it with a larger experiment (e.g., per-layer mixed precision sweep across all folds).

**M6. The benchmark (RQ1) is single-seed for networks and lacks paired comparisons; comparisons with the original PADS results are missing.**
(Section 4.1, Table 2.)
- Given that the paper's main message is run-to-run variability, conclusions drawn from one seed per network (and from untuned baseline architectures) are inconsistent with the paper's own recommendations. Run at least three seeds for each network in Table 2.
- "A feature-based random forest was more accurate" (abstract, Conclusions) is asserted without a paired test or CI on the difference. Within folds the gap is 0.905 vs 0.882 (SD 0.025 and 0.023); the DS-TCN's own within-fold range over runs is 0.849–0.881. Provide a paired within-fold difference with corrected-t CI. Also note that XGBoost (0.915/0.918), not the random forest, was the best model.
- Text (line ~235): "Pooled AUROC was lower than within-fold AUROC for every network (by 0.026 to 0.045)". Table 2 contradicts this: attention pooling 0.846 vs 0.847 (gap 0.001) and CNN-LSTM 0.839 vs 0.864 (gap 0.025). Correct the statement.
- Report the results of Varghese et al. (2024) on the same PD-vs-HC and PD-vs-DD tasks next to Table 2, with their metric, so readers can judge whether your pipeline is competitive.
- PD vs DD (the clinically relevant task in a referral clinic) is benchmarked but excluded from the quantization study; justify or extend.

**M7. Architecture/protocol deviations undermine the "planned" framing and parts of the device results.**
(Section 4.2 lines ~240–244; Section 5.1; Appendix A.)
- The specified architecture was not executed as planned in any run (stride placement, deviation i), five runs lacked depthwise BN (deviation ii), and the 16×8 latency, the first streaming firmware and the entire Monipar appendix use the BN-less model. Please (a) re-measure 16×8 latency and the streaming firmware with the specified-architecture model (cheap), (b) state the BN status of the Table 2 benchmark DS-TCN, and (c) either rerun Appendix A with the corrected model or drop it.
- "During revision we found ..." (line ~240) — this wording reveals manuscript history and is confusing for a first-round reader. Rephrase as a factual deviation report.
- The analysis plan and amendment log are not available to reviewers ("will be released with the code", line ~207; "upon acceptance", line ~476). A claim of pre-specification cannot be assessed without the dated document. Please provide the plan (with a verifiable timestamp, e.g., OSF/Zenodo/Git commit hash) as supplementary material now, together with the code, fold assignments and per-fold result files.

**M8. Confounding by age and sex is insufficiently addressed.**
(Section 3.1, line 98; Section 4.5.)
- HC are 63% women vs 29% in PD; age distribution of the groups is not reported at all. Report age (mean/SD) by group. Age+sex alone gave within-fold AUROC 0.68–0.69, and within women the network's AUROC drops to 0.79–0.84. Provide an age- and sex-matched (or reweighted) evaluation of the FP32 and quantized variants, and check that quantization effects are not different between sexes (e.g., because activation ranges differ with movement amplitude). The statement "The network therefore does not merely encode the demographic imbalance" is reasonable, but "how much" matters for interpreting 0.86–0.88.

**M9. Decision-level metrics for quantized variants are missing.**
- The paper reports AUROC almost exclusively, though the Methods promise BA, ECE and agreement. The deployed device outputs a decision. Please add balanced accuracy, sensitivity, specificity (with CIs) and ECE for FP32/PTQ/QAT/16×8 in Table 5, and separate the decision disagreement due to re-fitted thresholds (85% agreement) from that due to score changes (e.g., by applying the FP32 temperature/threshold mapped through the quantized output scale, or reporting agreement at fixed specificity).

---

## 4. Minor comments

1. Abstract: "lost 0.074" — add the unit (AUROC) and that this is one fold. Use "AUROC" consistently (the abstract alternates "area under the curve").
2. Abstract and Discussion: "48 times slower" — 562.5/11.56 = 48.7; say "about 49 times" or "~50×". Also note that it is 8× slower than int8 with portable kernels; the 48× figure is specific to the absence of ESP-NN int16 kernels.
3. Line ~311: "0.3% of real time" — with 50% overlap a window is classified every 2 s, so the duty cycle is ~0.6%.
4. Discussion line ~351: "varied by at most 0.03 across six runs" — Table 4 gives 0.849–0.881 = 0.032.
5. Table 2: give the parameter count for the attention-pooling DS-TCN ("--"); PD-vs-DD within-fold SD of 0.119/0.124 for the DS-TCN suggests a failed fold — report and discuss.
6. Table 4: replace "−0.000" with "0.000"; specify how many folds/units contribute to "worst fold".
7. Tables 2–5 are typeset with \resizebox and are barely legible at print size (Table 4/5 font is ~5 pt). Split or transpose them; avoid resizebox per Elsevier guidance.
8. Figure 2: the left-panel x-axis is truncated at about ±0.10 and the QAT+SR seed-0 point (+0.095) sits at the edge; ensure no points are clipped and use the same x-range in both panels for comparability. Add the number of units per row.
9. Figure 3: shows absolute AUROC with wide CIs that hide the paired effects; plot within-fold *differences* from FP32 per shift instead, and add the causal-filter (device-relevant) version. Also add FP32 + SR, which the text discusses (0.844 vs 0.789 at 180°).
10. Figure 1 is a generic flowchart; it could show the fold structure (outer/inner/validation split, where calibration windows come from), which is the paper's methodological core.
11. Section 3.4: specify how the 300 calibration windows are drawn (from training participants excluding validation? stratified by class/task?), and the TFLite converter options (representative dataset, optimization flags, per-tensor vs per-channel activations, input/output scale handling).
12. Section 3.3/3.4: give normalization details (per-channel z-score? statistics over windows or participants?), class weights, hardware and determinism flags; state the number of windows per class.
13. Section 3.2: thresholds and temperatures are fitted on ~57 participants (~13 HC) — report the variance of fitted temperatures across folds; this is directly relevant to M1.
14. Section 3.6: the 50-Hz shift equals clean data; keeping it in the 14-shift mean as a "negative control" dilutes the shift mean. Report the 13-shift mean as a sensitivity analysis.
15. Section 3.6: "different accelerometer" simulation re-adds gravity in a random direction to gravity-removed Apple Watch data; please justify that this produces realistic clipping (real gravity direction is not random relative to the wrist during seated tasks).
16. Table 6: report the model flatbuffer size (flash) for each variant, whether the tensor arena is in internal SRAM or PSRAM, and the ESP-NN version. "Image (kB)" includes the whole application and is less informative than model size.
17. Random forest deployment: the deployed forest (100 trees, depth 8) differs from the benchmarked one (500 trees, tuned); report its cross-validated AUROC under the same protocol, and its flash size.
18. Section 4.6: "outputs matched a causal Python implementation bit for bit on eight windows" — eight windows is very small; see M4.
19. Appendix A: Monipar raw acceleration contains gravity, PADS acceleration is gravity-removed by the device; a 0.5 Hz high-pass edge does not reproduce Apple's gravity removal during orientation changes. With 7 HC and a supervised/home confound, the appendix adds little; consider moving it to supplementary material.
20. Ethics: PADS and Monipar data-use terms (PhysioNet credentialed access/licence) should be named explicitly.
21. Generative-AI declaration: the stated use goes beyond language editing (writing analysis and firmware code, running statistical analyses). This is acceptable under Elsevier policy if declared, but the authors should describe how the AI-written analysis code was independently verified (beyond bit-exactness of firmware), e.g., by recomputing key statistics with independent code.
22. Introduction, first sentence: cite the GBD 2016 Parkinson's Disease Collaborators as a group author for ref. [1] rather than as "E. R. Dorsey, A. Elbaz, E. Nichols, et al."
23. Keywords: consider adding "post-training quantization", "quantization-aware training", "PADS".
24. Language is generally clear and concise; a few sentences are dense with numbers (e.g., Section 4.4 first paragraph, ~20 numbers). Move run-by-run numbers to the tables and keep the text to the key estimates with CIs.

---

## 5. Checks

### 5.1 Internal numeric consistency

I recomputed the summary values in Table 5 from Table 4 and checked abstract/highlights/text against tables and figures.

Consistent:
- Table 5 means vs Table 4 rows (specified-architecture runs): FP32 0.872/0.860; PTQ −0.013/−0.017; 2000-calibration −0.039/−0.055; 16×8 +0.001/−0.001; QAT −0.010/0.000; worst units −0.052/−0.089, −0.174 — all match.
- Abstract FP32 within-fold range 0.849–0.881 matches Table 4 (six specified runs); pooled range span 0.116 ≈ "0.12" (highlight 2).
- Table 3 Holm-adjusted p values are internally consistent with a 3-hypothesis Holm step-down within each seed.
- Fold-normalized H1 estimates (−0.088, −0.014, −0.015) agree with within-fold PTQ means in Table 4 (−0.087, −0.013, −0.015).
- Highlights ≤ 85 characters (74–82); abstract 232 words (< 250).
- Figure 2 point clouds match Table 5 worst units; float-stem values in Fig. 2 match lines ~295–297.
- Fig. 3 values for FP32 at 180° (0.789) and no-gyro (0.843) match the text.

Inconsistent / unsupported:
- Line ~235: "Pooled AUROC was lower than within-fold AUROC for every network (by 0.026 to 0.045)" — attention pooling gap 0.001; CNN-LSTM 0.025 (Table 2).
- Line ~270 and ~360: 2000-window calibration "worse in every run"/"consistently" — false for causal BN-less seeds 0 and 2 (Table 4).
- Abstract/highlight 3/Conclusions: "from none to 0.087 across repeated training runs (of the same configuration)" — range includes BN-less runs (different architecture).
- Line ~351: "at most 0.03" vs 0.032.
- Pooled FP32 AUROC 0.700–0.816 in quantization runs vs 0.856 in the Table 2 benchmark for the same architecture/seed: unexplained (M1).
- −0.017 (forest, full CV) vs −0.074 (network, fold 0) compared as if equivalent (M4).
- ECE and BA promised in Methods; ECE never reported, BA only in Appendix Table A.7.

### 5.2 Are claims supported?
- "Planned evidence inconclusive": supported.
- "Pooling across folds was unstable" (as a general lesson): partially; likely protocol-dependent (M1).
- "PTQ loss depended on the run": weakly supported for the specified architecture (one outlier run, n = 2 repeats; confounded with calibration sampling) (M2).
- "Grew with larger calibration set": supported for specified architecture only; expected for min/max calibration (M3).
- "16-bit activations removed the loss": supported.
- "QAT varied less": suggestive; no formal variance comparison; averages similar to PTQ in specified runs.
- "Quantization did not add loss under shift": not supported as stated (null superiority test) (M5).
- "Accurate on-device inference required training with the causal filter": one fold, no CI (M4).
- "Random forest more accurate": no paired test; XGBoost was best (M6).
- "Bit-exact on ESP32-S3": supported for the windows tested; small samples.

### 5.3 Statistical soundness
- Planned vs exploratory: clearly labelled — a strength. However, the plan cannot be inspected (M7), the plan had no cross-seed decision rule, and the primary estimand was fragile (M1).
- Multiplicity: Holm within seed for planned tests; exploratory p values explicitly unadjusted — acceptable if claims are worded as descriptive, which in the abstract/highlights they are not always.
- CIs: bootstrap for pooled, Nadeau–Bengio corrected t for within-fold — appropriate; but n = 15 fold–seed units with ~16 HC per fold gives wide intervals; most exploratory CIs include 0 (Table 5), which should be reflected in the wording. Independence assumptions violated in the "45 units" TOST (M5).
- Equivalence: only for 16×8; should be applied to H2 and to PTQ/QAT vs FP32.

### 5.4 Reproducibility
- Positives: public data, software versions (TF 2.21, TFMOT 0.8.1, ESP-IDF 5.3.2), architecture description, seeds, bit-exactness tests.
- Gaps: code, fold assignments, analysis plan only "upon acceptance"; training non-deterministic; calibration-window sampling and converter flags under-specified; normalization details missing; benchmark network hyperparameters minimal. Please provide code and plan to reviewers now.

### 5.5 Scope fit
Good fit for BSPC (wearable inertial signal processing, preprocessing–device matching, embedded deployment). The contribution is primarily methodological/evaluative; this is acceptable for BSPC provided the evidence is solid and generalizable beyond one network and one default calibrator, which is currently not the case.

### 5.6 Length and format
- Abstract 232 words (OK, < 250). Highlights 5 bullets, 74–82 characters (OK).
- Main text ~4,600 words excluding tables, references and appendix (OK for ~5,000).
- Elsevier elsarticle preprint, line numbers, CRediT, competing interests, funding, data availability and generative-AI declaration present (OK).
- Tables use \resizebox and become unreadably small (minor 7). Seven tables and three figures for a ~4,600-word paper is dense; consider moving Table 4 or Table A.7 to supplementary material.

### 5.7 Reference quality
- 48 references; mostly appropriate primary sources for quantization (Jacob 2018, Krishnamoorthi 2018, Nagel 2019/2021), CV evaluation (Forman 2010, Airola 2011, Nadeau & Bengio 2003, Bouckaert & Frank 2004) and PD wearables.
- BSPC papers cited: El Allam & Hamlich 2026 (used substantively — post-INT8 calibration shift), Coyle-Asbil et al. 2026 (used substantively — filtering and inter-brand equivalence), Ortone et al. 2026 (sEMG on STM32; tangential), Magni et al. 2026 (PD vs NPH gait with shoe sensors; used only to illustrate small-cohort variance — tangential and reads as a journal-fit citation). Two of four BSPC citations are meaningful; the other two could be replaced by more relevant BSPC work on wearable PD detection/tremor or embedded biosignal inference.
- Two preprints (Zannat 2026; Zhao et al. 2025) are used as the only performance comparators on PADS; preprints should be labelled as such in the reference list (Zannat is labelled in Table 1 only) and peer-reviewed PADS results, if any, should be preferred.
- Several 2026 references (JBHI early access, BSPC 2026 volumes) could not be verified from this review environment; the authors should ensure all DOIs resolve. A quick search located the Magni et al. work (also as a medRxiv preprint), but not the El Allam & Hamlich article; please double-check that citation.
- Missing relevant literature: activation-range calibration methods for PTQ (percentile/KL/MSE; e.g., Wu et al. 2020 "Integer quantization for deep learning inference"), AdaRound (Nagel et al., ICML 2020) and BRECQ; variance in ML benchmarks across seeds (Bouthillier et al., MLSys 2021) and small-sample CV error bars (Varoquaux, NeuroImage 2018), both directly relevant to the paper's central claims; Fawcett (2006) on ROC averaging across folds.
- Ref. [1] should be listed under the GBD 2016 Parkinson's Disease Collaborators group author.

---

## 6. Scores (1 = poor, 5 = excellent)

| Criterion | Score | Comment |
|---|---|---|
| Originality | 2 | Known effects (fold-pooling bias, min/max PTQ sensitivity of depthwise nets, causal vs zero-phase mismatch) tested in a new domain; the new element is the stability analysis, which is under-powered. |
| Methodological soundness | 2.5 | Strong leakage control and transparency; but confounded variance sources (training vs calibration draw), single default PTQ calibrator, single-fold device claims, null tests over-interpreted, protocol-induced pooling instability. |
| Clarity | 3.5 | Concise, honest writing; dense numeric paragraphs, tiny tables, and mixing of architectures in summaries reduce clarity. |
| Significance to BSPC readers | 3 | Practical lessons (match device preprocessing; evaluate per fold and across runs; check 16-bit kernel availability) are useful for embedded wearable work, if generalized and properly supported. |
