# Review — Biomedical Signal Processing and Control

**Manuscript:** "Eight-bit quantization of tiny wrist-inertial networks for Parkinson's disease detection: a pre-specified empirical study with microcontroller validation"

**Material reviewed:** main.pdf (25 pp., line-numbered preprint), main.tex, highlights.txt, Figure_1.pdf, Figure_2.pdf. Line numbers below refer to the PDF.

---

## 1. Summary

The authors examine whether a 22.7k-parameter depthwise-separable temporal CNN (DS-TCN) for Parkinson's disease (PD) versus healthy-control (HC) classification from bilateral smartwatch IMU data (PADS, n = 469) keeps its accuracy after INT8 conversion, both in distribution and under 14 simulated device shifts. They frame it as a study with a plan fixed in advance: three hypotheses (H1: PTQ non-inferior to FP32 at margin −0.02 AUROC; H2: quantization adds loss under shift; H3: QAT + sensor randomization beats PTQ under shift), tested on AUROC pooled across outer folds, for three seeds. H1 fails in all seeds, H2 is not supported, and H3 is supported in two of three seeds. Exploratory analyses then switch to within-fold AUROC, attribute the PTQ loss to 8-bit activations (a 16×8 experiment), report that QAT avoids the loss, find that sensor randomization does not help consistently, and report that pooling understates AUROC by > 0.1. The model and a random-forest baseline are deployed on an ESP32-S3 with bit-exact outputs, 11.6 ms network latency and a 12.6 ms streaming step. On-device preprocessing showed that a network trained with zero-phase filtering loses 0.074 AUROC when given causally filtered input. A 28-participant external cohort (Monipar) is reported descriptively only.

## 2. Recommendation: **Major revision**

The study is unusually transparent: participant-level nested CV, a stated analysis plan, a leakage control, an honest deviations section, real bit-exactness checks on hardware, and a stronger feature baseline that the authors openly say beats their network. Methodologically careful empirical work of this kind fits BSPC. However, the paper's headline claims do not rest on the confirmatory analysis.

- **The pre-specified primary estimand is one the authors themselves call distorted.** Pooled AUROC is the basis of H1–H3, and Discussion l. 356–358 still calls the pooled H3 result "the primary result", yet l. 365–368 says pooling distorts results.
- **The mechanistic claim is extrapolated.** The 16-bit-activation explanation in the highlights and abstract comes from a different, exploratory setting (causal filter), where the PTQ loss was itself not significant.
- **The FP32 reference is not reproducible within the paper.** It differs by ~0.1 AUROC between two "implementations" of the same model.
- **Code and the analysis plan are not available.**

All of this can be fixed, but it needs new analyses and substantial reframing, not just editing.

---

## 3. Major comments

**M1. The confirmatory analysis uses an estimand the authors later reject, so the "pre-specified" framing does not hold up.**
- H1–H3 are tested on AUROC pooled across five temperature-scaled outer folds (l. 197–198). Section 4.3 and the Discussion (l. 250–255, 365–374) argue that this estimand mixes ranking with fold-specific score scales and understates AUROC by > 0.1.
- Yet Discussion l. 356–358 still calls pooled H3 ("QAT+SR beat PTQ in two of three seeds") "the primary result". Within folds, the H3 contrast is +0.038 (−0.054 to 0.130) with the zero-phase filter and −0.009 with the causal filter, i.e. essentially null. The pooled H3 effects of +0.133 and +0.122 are therefore probably largely artefacts of the very problem the paper describes.
- The paper cannot hold both positions. The title, abstract, highlights and conclusions lead with "pre-specified" findings, but every substantive message is exploratory: the activation mechanism, QAT ≈ FP32, no SR benefit, and the causal-filter requirement.

The authors must:
- (a) state plainly in the abstract, highlights and conclusions that the confirmatory H3 result is not supported by the within-fold analysis and should not be read as evidence for the proposed method;
- (b) drop "This is the primary result" or qualify it heavily;
- (c) reconsider "pre-specified" in the title, since most reported conclusions are post hoc;
- (d) deposit the dated analysis plan as supplementary material, with a verifiable timestamp (OSF, Zenodo, a git commit hash or similar). "A dated document" (l. 191) that reviewers cannot inspect carries no weight as pre-registration;
- (e) state the confirmatory decision rule across seeds. Table 3's footnote says "Seed 0 is primary", but the text reports outcomes "in two of three seeds" as if seeds were replications. Which rule was fixed in advance?

**M2. The explanation of the pooling discrepancy is asserted, not shown, and the phenomenon is already known.**
- Temperature scaling divides the logit by a single scalar. It cannot reorder participants within a fold, and by itself it does not create between-fold offsets. Offsets arise from fold-specific logit biases and differing score distributions, with or without temperature scaling.
- To support the attribution to "temperature-scaled folds" (abstract, highlight 4, l. 204, 366), the authors must report pooled AUROC computed on: (i) raw logits before temperature scaling; (ii) per-fold standardized or rank-normalized scores; and (iii) per-fold AUROC with its between-fold SD.
- Within-fold AUROC with ~16 HC per test fold (79 HC / 5) is itself very noisy and biased in small samples. The paper should report the fold-level distribution (e.g., a dot plot) rather than only the mean.
- The problem of pooling versus averaging in cross-validation is documented (e.g., Forman and Scholz, SIGKDD Explorations 2010; Airola et al., Computational Statistics & Data Analysis 2011). The "methodological lesson" (l. 365) must be placed in that literature, not presented as a new finding.

**M3. The claim that the loss comes from 8-bit activations is extrapolated from a different experimental setting and is confounded with input quantization.**
- In the abstract, highlight 2 and l. 276–278 / 350–352, the 0.066 zero-phase PTQ loss is said to have "vanished when activations were kept at 16 bits" and "was not removed by more calibration data". But the 16×8 and 2000-window experiments were run only with the causal filter (Table 5). There, the 300-window PTQ loss was only −0.025 (−0.074 to 0.024), a CI that includes zero. Showing that a non-significant loss "was not removed" (−0.017, CI also includes zero) is not evidence. The abstract's sentence splices results from two pipelines.
- The models are converted with int8 input and output (l. 162). The 16×8 mode also quantizes the input tensor to 16 bits. Raw gyroscope and acceleration channels with heavy tails, quantized per tensor to 8 bits at the input, are an equally plausible cause. "Activations, not weights" is therefore not separated from "input quantization".
- Required:
  - (i) run the 16×8 and calibration-size experiments in the zero-phase setting where the loss is large, or restrict the claim to the causal setting;
  - (ii) run a float-input/int8-internal variant, or an int16-input/int8-activation variant, to separate input from internal-activation quantization;
  - (iii) provide layer-wise quantization error (e.g., SQNR per layer) and activation-range statistics per channel for the depthwise layers. Without these, the explanation at l. 352–353 is speculation;
  - (iv) apply standard PTQ remedies (cross-layer equalization, bias correction, AdaRound) or justify their omission. Limitations l. 407–408 acknowledges they are missing, but they are the obvious control for "PTQ is unreliable";
  - (v) explain why the PTQ loss more than halves when the filter changes from zero-phase to causal (0.066 vs 0.025). This is not discussed and bears directly on the mechanism.
- Note also that PTQ after sensor randomization lost only 0.007 (l. 265). This suggests SR regularizes activation ranges, i.e., SR does help quantization, which contradicts the blanket statement that "sensor randomization gave no consistent benefit". Discuss this.
- Finally, that depthwise-separable networks quantize poorly per tensor and that QAT fixes it is well established (Krishnamoorthi 2018 [17]; Sheng et al., "A quantization-friendly separable convolution for MobileNets", 2018). The novelty of this finding is low and should be presented as a confirmation in a new domain.

**M4. The FP32 reference is inconsistent across the paper.**
- Table 2 reports DS-TCN (raw) pooled AUROC 0.856 (one seed).
- The quantization study's FP32 pooled AUROC for seed 0 is 0.762 (l. 250), and its within-fold value is 0.879.
- L. 251–252 dismisses the gap: the benchmark "came from a different implementation and training run".

The same architecture, data and protocol give pooled AUROCs ~0.09 apart. That gap is larger than every quantization effect the paper is about. It suggests the pooled metric, or one of the pipelines, is unstable, and it raises the question of which implementation was deployed on the device. The authors must run Table 2 and the quantization study through one code path, explain the gap concretely, and report seed-to-seed variability of the FP32 model itself (SD across seeds for pooled and within-fold AUROC). If FP32 training variance is comparable to the PTQ loss, H1 must be interpreted in that light. Limitations l. 407 already admits non-deterministic training.

**M5. The network's practical advantage over the random forest is asserted, not evaluated.**
- The paper justifies the tiny network by saying it "scores any 4-s window from one wrist" (l. 338–339, 385–386), whereas the RF needs the full two-wrist, 11-task protocol.
- But every network result in Tables 2–5 averages window probabilities over all windows from both wrists and all 11 tasks (l. 153–154). The network uses the full protocol too.
- The only single-window figure is the window AUROC of 0.748 (l. 237), far below the RF's 0.906.
- To support the stated use case, report participant-level performance of the network and the RF under matched reduced protocols: single wrist (dominant / more-affected), single task, and a fixed number of windows. A feature RF trained on one task and one wrist is an obvious baseline. As it stands, the RF dominates on accuracy, also runs on the MCU, needs no quantization, and is 4–5× cheaper per participant in total compute (≈ 0.23 s vs 11.6 ms × several hundred windows). The paper should state this clearly in the Conclusions rather than leaving it in Results.

**M6. Statistical treatment of the exploratory analyses.**
- (a) The 15 "fold–seed units" are not independent. State exactly which correction is used for repeated k-fold. Nadeau–Bengio [47] targets repeated random subsampling; for repeated 5-fold, the Bouckaert–Frank corrected repeated k-fold t-test is the usual choice. Give the n_test/n_train ratio and the degrees of freedom.
- (b) Dozens of exploratory contrasts are reported (6 variants × 15 conditions in Fig. 2, Tables 4–5, TOST tests, per-shift claims such as "helped for a 180° rotation") with no multiplicity control and no stated family. Either control the false discovery rate or explicitly describe all exploratory p values as descriptive. Words such as "equivalent" (l. 276, 278) should not be used for exploratory TOST results, e.g., p = 0.016 for QAT vs fine-tuned.
- (c) Table 2: point estimates are three-seed means but CIs come from seed 0 (footnote a), and the networks have one seed. The CIs ignore training variance and do not belong to the reported point estimate. Use the same seeds and the same CI method for all models, or compare models with paired tests that include seed and fold variance.
- (d) Report how participant-level CIs were obtained when the "threshold of the fold that tested it" is retained (l. 186–187). The bootstrap ignores threshold-estimation uncertainty.
- (e) The 20% validation split (~57 participants, ~16 HC) is used at the same time for early stopping, temperature fitting, threshold selection and QAT early stopping. This reuse biases calibration and thresholds and should be acknowledged or avoided.
- (f) PD vs DD and the plain-CNN contrast rest on one seed (l. 282–287), yet the Discussion generalizes from them ("For a plain convolutional network, post-training quantization was harmless", l. 349–350). Either run three seeds or drop the generalization.

**M7. Confounding and clinical framing.**
- HC are 63% female versus 29% for PD (l. 239), and age and sex alone give AUROC 0.674. The network results are not adjusted or stratified (Limitations l. 409–411). Since the network is the object of study, at minimum report network AUROC within sex and an age-adjusted analysis, or a logistic model with the network score plus age and sex.
- The abstract opens with "screening" and the Introduction with "screening at scale" (l. 9). But the data come from a referral clinic, the PD-vs-HC task excludes the clinically hard comparison, and on PD vs DD the network is weak (AUROC 0.635). Rewrite the framing to match the evidence (detection in a referred population, l. 411–412) from the abstract onward, not only in the Limitations.

**M8. The device-shift component is only weakly validated, so the robustness claims need scaling back.**
- The "different accelerometer" simulation (l. 176–177) adds gravity, clips, requantizes and removes gravity. It does not model frequency response, anti-aliasing, output-data-rate mismatch, clock drift or sensor-specific noise spectra, which are exactly the differences reported in [22].
- The 25-Hz shift interacts with the 0.5–20 Hz band-pass (Nyquist 12.5 Hz). Explain how a 25-Hz signal is brought back to 50 Hz and what the 12.5–20 Hz band contains afterwards.
- The external cohort (7 HC) gives CIs 0.3–0.4 wide. It mixes 6 supervised and 15 home-recorded PD participants with HC whose recording setting is not stated (a possible setting confound). It uses a different wrist rule and a gravity-removal method different from PADS. It is analysed with a separately trained model (three tasks, acceleration only, single training run).
- Remove "Domain shift" as a keyword or clearly qualify it, and state in the abstract that robustness to real device change was not established. Also consider dropping Monipar from the main text (move it to the supplement). In its current form it adds noise, not evidence. If kept, report the PD-supervised vs PD-home vs HC breakdown.

**M9. Reproducibility.**
For a paper whose main contribution is a methodological warning, "code available on reasonable request" (l. 448–449) is insufficient. The authors must release:
- training and evaluation code, fold assignments and seeds;
- the frozen analysis plan;
- the conversion scripts and the firmware (TFLM/ESP-NN versions, compiler flags).

Several details are missing from the text:
- sensor-randomization parameters (rotation distribution, noise SD, gain distribution, l. 167–169);
- architecture and hyperparameters of DS-CNN, 1-D CNN and CNN-LSTM (Table 2);
- the RF and XGBoost search grids;
- the number of windows per participant;
- how missing task–wrist recordings are imputed for the RF;
- TensorFlow and TFLite versions;
- the exact QAT configuration (per-channel or per-tensor, whether BN is folded);
- the composition of the 300 calibration windows (class balance, participants).

Does the "seed" in the three-seed design change only network initialization, or also the outer-fold assignment (l. 111–113)? This matters for the interpretation of every between-seed difference.

**M10. On-device evaluation needs to be completed, given the paper's own lesson about preprocessing mismatch.**
- (a) Bit-exactness is checked on 50, 8 and 27 windows (l. 222, 315, 323–324). Run the full fold-0 test set (all windows) through the device pipeline and report participant-level AUROC from on-device outputs.
- (b) The deployed RF was trained on zero-phase features but run with causal features, and its on-device accuracy was not re-estimated (l. 412–415). This is precisely the mismatch the paper identifies as critical for the network. It must be evaluated, not left as a limitation, especially since the RF is presented as the stronger on-device option.
- (c) Energy: a 10-mA-resolution USB meter on the whole board (l. 325–329) cannot support any power statement for a wearable. Measure current with a proper instrument (shunt with oscilloscope, or a power profiler) and report energy per inference and an estimated battery life for a duty-cycled wearable, or drop power from the claims. BSPC readers will expect this for a deployment paper.

---

## 4. Minor comments

1. **Abstract:** at 242 words it is near the 250-word limit and very dense with numbers. Remove the pooled-vs-within-fold digression or the RF sentence. The final sentence ("Eight-bit activations, not weights, made ... unreliable") states the exploratory mechanism as fact; qualify it (see M3).
2. **Highlights:** highlight 1 is 86 characters including spaces, above Elsevier's 85-character limit. Highlights 2 and 3 state exploratory findings as established ("removed it in every fold" applies only to the causal runs; "kept full-precision accuracy", whereas equivalence within ±0.02 was *not* established in the zero-phase setting, p = 0.12, l. 259).
3. **l. 2–3:** "fastest-growing neurological disorder" with citations from 2018. Update, or soften to "among the fastest-growing".
4. **l. 56–57:** "the abstract does not describe the split". Reviewers expect the authors to read the full paper [26] before characterizing it in Table 1 ("n.r."). Similarly, [31] is mentioned without saying what it reports.
5. **Table 1:** the "External" column for this work is "(✓)a" based on 7 controls with no conclusion drawn. Marking it as an external evaluation overstates the contribution; use "–" or "descriptive only" in the cell itself.
6. **l. 146–148:** the dilation finding (0/50 bit-exact for space-to-batch) is interesting and practically useful. Specify the TFLM/ESP-NN versions and whether this is a known converter or kernel issue. It deserves a sentence in the Discussion.
7. **Eq. (1):** a^T ω and ||a × ω|| combine quantities in different units (g and rad/s) under different frames. Comment on whether both sensors share an axis frame on the Apple Watch, and on scaling before int8 input quantization.
8. **l. 160:** early stopping on *window* AUROC while the endpoint is participant AUROC. Justify.
9. **Table 4:** report SDs or CIs next to the means; the table alone does not show the fold-level instability the text emphasizes. Also report the agreement result: QAT agrees with FP32 on only 85.6% of decisions despite equal AUROC. This decision-level instability (refitted thresholds) is clinically relevant and should be discussed.
10. **Table 5:** "PTQ of fine-tuned" uses a different reference row than the column header ("Clean − reference"). Split it into a separate column or row for clarity.
11. **l. 312–314 vs Table 7:** end-to-end 14.6 ms with 2.4 ms preprocessing leaves 12.2 ms, but Table 7 inference is 11.56 ms; the streaming firmware gives 11.6 ms for normalization, quantization and inference together. Account for the ~0.6 ms difference.
12. **l. 334–336:** the 0.23 s assessment time assumes 10-s recordings, but PADS steps last 10–20 s (l. 94). Recompute using actual durations.
13. **Fig. 2:** points for the six variants overlap heavily. Apply a horizontal dodge, use larger markers, or plot differences from FP32. The y-axis label "AUROC within fold (mean, 95% CI)" should state the CI method (corrected t). The x labels are rotated 60°; a horizontal layout or a table may read better.
14. **Fig. 1:** informative but generic. The row-3 arrows suggest all six variants go to the ESP32, yet only fold-0 PTQ/QAT/QAT+SR were deployed. The causal-filter branch, which is central to the conclusions, is absent.
15. **l. 182:** keeping a shift that is identical to the clean condition (50 Hz) in the 14-shift mean dilutes H2/H3. It is pre-specified, so keep it, but say so where H2/H3 are defined.
16. **l. 225–226:** "tensor-arena size actually used". Specify how it was measured (TFLM `arena_used_bytes()`?).
17. **l. 266:** "All variants lost accuracy without the gyroscope". Give numbers. Fig. 2 suggests the QAT+SR drop is the largest (≈ 0.72), a notable SR side effect.
18. **Section 5.1:** the stride-placement deviation changes the receptive field and the downsampling behaviour, not just the parameter count. State whether it could affect activation ranges, and therefore PTQ behaviour.
19. **Generative-AI declaration:** the authors state Claude was used for "writing analysis and firmware code, running the statistical analyses". Under Elsevier policy, AI use in the research process (as opposed to writing assistance) should be described in the Methods, with how its outputs were verified.
20. **Data availability:** PADS is on PhysioNet under a credentialed or restricted licence (check). State the licence and access conditions accurately.
21. **References:**
    - Capitalization errors: "Mcunet", "Mlperf tiny", "mpower", "Tensorflow lite micro".
    - Missing space before "doi" in [25] and [41] ("1–13doi", "1–14doi").
    - Peer-reviewed versions exist for [12] (MLSys 2021), [13] (NeurIPS 2020) and [14] (NeurIPS Datasets & Benchmarks 2021). Cite those instead of arXiv.
    - [26] and [31] are non-peer-reviewed preprints that support comparative claims; flag them as preprints in the text.
    - [1] lists ~70 authors; Elsevier style permits "et al." truncation.
22. **Language:** the prose is clear and economical, but Results sections 4.3–4.4 are a run of numbers that is hard to follow. Each subsection should open with one sentence stating the finding, then the numbers. Distinguish zero-phase from causal runs consistently, e.g., label every number with its setting.

---

## 5. Checks

### 5.1 Internal numeric consistency

| Item | Abstract / Highlights | Text | Tables / Figures | Verdict |
|---|---|---|---|---|
| H1 pooled losses | 0.033–0.064 | −0.063, −0.064, −0.033 (l. 245–246) | Table 3 identical | Consistent |
| Within-fold PTQ loss | 0.066; 0.219 max | 0.066 (l. 256), 0.219 | Table 4: 0.874 − 0.808 = 0.066 | Consistent |
| QAT vs FP32 | "no detectable loss" | +0.003, equivalence **not** shown (p = 0.12) | Table 4: 0.877 vs 0.874 | Numbers consistent; wording overstated in abstract and highlight 3 |
| QAT − PTQ (shift) | – | 0.068 | Table 4: 0.861 − 0.794 = 0.067 | Rounding, acceptable |
| H3 within-fold | – | +0.038 | Table 4: 0.832 − 0.794 = 0.038 | Consistent |
| SR effect | "no consistent benefit" | −0.027 clean; −0.030 QAT shift | 0.847 − 0.874; 0.832 − 0.861 = −0.029 | Consistent (rounding) |
| PTQ+SR loss | – | 0.007 | 0.840 − 0.847 | Consistent |
| Pooling gap | "more than 0.1" | 0.762 vs 0.879 (seed 0 only) | Not in any table | Supported for seed 0 only; give all seeds |
| FP32 pooled reference | – | 0.762 (seed 0) | Table 2: 0.856 | **Inconsistent** (M4) |
| "Not removed by more calibration data", "vanished at 16 bits" | Abstract, highlight 2, attached to the 0.066 loss | Causal runs only (l. 269–278) | Table 5 (causal; base loss 0.025, n.s.) | **Settings conflated** (M3) |
| Causal-run deltas | – | −0.025, −0.017, −0.001, +0.015, −0.020, +0.019, −0.039 | Table 5 rows match within ±0.001 rounding | Consistent |
| Causal H3 | – | −0.009 | 0.819 − 0.828 | Consistent |
| 180° rotation | – | 0.837 vs 0.772 | Fig. 2 matches visually | Consistent |
| Latency | 12.6 ms / 2-s step; 39.4 kB | 12.6 = 1.0 + 11.6 | Table 7: 11.56 ms | Consistent |
| End-to-end window | – | 14.6 ms incl. 2.4 ms preprocessing | 11.56 ms inference → 0.6 ms unexplained | Minor gap (m11) |
| Real-time fraction | – | 0.3% | 11.56 / 4000 = 0.29% | Consistent |
| Portable slowdown | – | 6.1× | 70.29 / 11.56 = 6.08 | Consistent |
| RF features | – | 2112 = 22 × 96 = 22 × 8 × 12 | – | Consistent |
| Fold-0 test size | – | 71 participants | (276 + 79) / 5 = 71 | Consistent |
| Causal-filter loss | – | 0.890 → 0.816 (l. 317–318) = 0.074; Discussion 0.074 | – | Consistent |
| Leakage gain | – | 0.748 → 0.862 (+0.114), 0.639 → 0.827 (+0.188); Discussion "0.11 to 0.19" | – | Consistent |
| Monipar BA values | – | – | Table 6 values achievable with 21/7 counts | Plausible |
| Fig. 2 caption | "15 fold–seed units"; 50 Hz omitted | – | 15 x-ticks (clean + 13 shifts + no-gyro = 14 shifts minus 50 Hz) | Consistent |

### 5.2 Are the claims supported?
- **Supported:**
  - PTQ of this DS-TCN loses AUROC in the zero-phase setting, both pooled and within fold;
  - QAT closes the gap within fold (point estimate);
  - quantization does not interact with simulated shift;
  - bit-exact TFLM/ESP-NN inference with the stated latency;
  - a zero-phase-trained model degrades under causal filtering;
  - the RF outperforms the networks.
- **Not adequately supported:**
  - that 8-bit activations, not weights, are the cause (M3; different setting, input quantization confounded, no layer-wise evidence);
  - that pooling distortion comes from temperature scaling (M2);
  - that QAT+SR is beneficial (the H3 "primary result", M1);
  - that a single-window, single-wrist network is a practical advantage (M5);
  - any claim about real device shift (M8);
  - any power or energy feasibility (M10c);
  - the RF's on-device accuracy (M10b).

### 5.3 Statistical soundness
- **Pre-specified vs exploratory.** Labelled clearly in the Methods, which is commendable. But the abstract, highlights and conclusions blend the two, and every highlight except the first rests on exploratory results. The plan is not provided.
- **Multiple comparisons.** Holm within seed for the three confirmatory tests is appropriate. There is no control for the many exploratory contrasts, the per-shift results or the TOST tests, and the across-seed decision rule is ambiguous (M1e, M6b).
- **CIs.**
  - Participant-level stratified bootstrap ignores training and threshold variance.
  - The corrected resampled t needs the exact correction for repeated k-fold to be specified.
  - Table 2 mixes three-seed point estimates with seed-0 CIs (M6).
- **Sample size per fold.** About 16 HC per test fold makes within-fold AUROC noisy. Fold-level dispersion should be shown.
- **Non-inferiority margin.** −0.02 AUROC is reasonable but not justified clinically or empirically. Relate it to FP32 seed-to-seed variance (M4).

### 5.4 Reproducibility
Insufficient as submitted (M9): no code, plan, fold lists or firmware, and several hyperparameters are missing. The hardware description (ESP-IDF 5.3.2, module specification) is good.

### 5.5 Fit to BSPC scope
Good. Wearable inertial signal processing, filtering (the causal vs zero-phase finding is a genuine signal-processing point), embedded deployment and evaluation methodology are all within scope. The quantization component is generic ML engineering. The paper would fit BSPC better if it developed the signal-processing aspects (filter-phase mismatch, sensor-heterogeneity modelling, input scaling for int8) more than the generic quantization findings.

### 5.6 Length and format compliance
- **Format:** elsarticle preprint, line numbers, CRediT, competing interests, funding, ethics, data availability and AI declaration are all present.
- **Length:** body text (Introduction to Conclusions) is roughly 4,500–5,000 words, within the ~5,000-word guideline. 7 tables and 2 figures is table-heavy; consider moving Table 1 or Table 6 to the supplement.
- **Abstract:** 242 words, compliant but at the limit.
- **Highlights:** 5 bullets (compliant count); highlight 1 exceeds 85 characters (86).
- **Graphical abstract:** none provided (optional at BSPC, but recommended).

### 5.7 Quality of references
- **Size and mix.** 48 references, mostly relevant and recent, with DOIs.
- **Preprints.** Several key ML and TinyML references are cited as arXiv preprints although peer-reviewed versions exist. Two PADS comparators are preprints.
- **Missing literature.**
  - quantization of depthwise-separable networks (Sheng et al. 2018) and PTQ remedies (Nagel et al., DFQ 2019; AdaRound 2020);
  - pooling vs averaging AUROC in CV (Forman and Scholz 2010; Airola et al. 2011);
  - the corrected repeated k-fold test (Bouckaert and Frank 2004);
  - causal vs zero-phase filtering in real-time wearable pipelines.
- **BSPC citations.** Four BSPC papers are cited ([20], [22], [36], [42]), all from 2026:
  - [20] (quantization-aware conformal prediction for ECG on ESP32-S3) is used meaningfully and is the closest prior work.
  - [22] (accelerometer brand equivalence) is relevant, but its findings on frequency response are not reflected in the shift simulation (M8).
  - [36] is cited only to show that small cohorts give wide CIs (a gait PD-vs-NPH study), which is tangential.
  - [42] (sEMG on STM32) is tangential.

  BSPC has published a large body of work on wearable IMU-based PD tremor and bradykinesia detection and on embedded biosignal classification. None of it is engaged in the Related Work. The current selection reads as journal-matching rather than substantive engagement; please engage with BSPC work directly relevant to IMU-based PD classification.

---

## 6. Scores (1 = poor, 5 = excellent)

| Criterion | Score | Comment |
|---|---|---|
| Originality | 2 | PTQ fragility of depthwise-separable nets, QAT as a fix, CV pooling artefacts and filter-phase mismatch are each known individually. The combination in a PD wearable setting with real MCU verification is the contribution. |
| Methodological soundness | 2 | Strong protocol hygiene (participant-level nested CV, leakage control, plan, bit-exact checks), undermined by the invalid primary estimand, conflated settings for the mechanism claim, the unexplained FP32 discrepancy and missing code. |
| Clarity | 3 | Concise, honest prose. Results are an overload of numbers across four settings (pooled / within-fold × zero-phase / causal) that readers will struggle to keep apart. |
| Significance to BSPC readers | 3 | Practically useful warnings for embedded wearable developers (validate per fold, match on-device preprocessing, beware dilation conversions). Clinical significance is limited because the RF baseline dominates and real device shift is untested. |
