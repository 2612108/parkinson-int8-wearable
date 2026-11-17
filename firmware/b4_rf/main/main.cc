// b4_rf — Random Forest (emlearn float, 100 cây, max_depth=8, 2112 đặc trưng)
//         + Trích xuất đặc trưng trên bo (12 đặc trưng/kênh x 8 kênh = 96 đặc trưng cho 1 bản ghi 10 s x 6 kênh @ 100 Hz)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_chip_info.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "generated/b4_data.h"
#include "generated/pd_rf.h"
#include "generated/pd_rf_hard.h"

#define WARMUP_RF 20
#define N_TIMED_RF 1000
#define WARMUP_FEAT 5
#define N_TIMED_FEAT 100

#define WELCH_NPERSEG 256
#define WELCH_NSTEP   128
#define WELCH_NFREQ   129   // 256/2 + 1
#define FS_DECIM      50.0f

// Bộ đệm tĩnh cho đường ống trích xuất đặc trưng 1 bản ghi 10 s (1000 mẫu @ 100 Hz -> 500 mẫu @ 50 Hz)
alignas(16) static float s_x50[B4_REC_RAW_C][B4_REC_DECIM_T];     // 6 x 500 x 4B = 12,000 B
alignas(16) static float s_xf[B4_REC_RAW_C][B4_REC_DECIM_T];      // 6 x 500 x 4B = 12,000 B
alignas(16) static float s_ch8[B4_REC_OUT_C][B4_REC_DECIM_T];     // 8 x 500 x 4B = 16,000 B
alignas(16) static float s_feats96[B4_REC_N_FEATS];               // 96 x 4B = 384 B
alignas(16) static float s_sort_buf[B4_REC_DECIM_T];              // 500 x 4B = 2,000 B
alignas(16) static float s_hann256[WELCH_NPERSEG];                // 256 x 4B = 1,024 B
alignas(16) static float s_twiddle_re[WELCH_NPERSEG / 2];         // 128 x 4B = 512 B
alignas(16) static float s_twiddle_im[WELCH_NPERSEG / 2];         // 128 x 4B = 512 B
alignas(16) static uint8_t s_bitrev256[WELCH_NPERSEG];            // 256 B
alignas(16) static float s_fft_re[WELCH_NPERSEG];                 // 1,024 B
alignas(16) static float s_fft_im[WELCH_NPERSEG];                 // 1,024 B
alignas(16) static float s_psd[WELCH_NFREQ];                      // 516 B
static float s_win_norm = 1.0f;

static void init_welch_tables() {
  const float two_pi_over_n = (float)(2.0 * M_PI / (double)WELCH_NPERSEG);
  float sum_w2 = 0.0f;
  for (int n = 0; n < WELCH_NPERSEG; ++n) {
    // scipy.signal.get_window('hann', 256, fftbins=True) -> periodic Hann
    const float w = 0.5f - 0.5f * cosf(two_pi_over_n * (float)n);
    s_hann256[n] = w;
    sum_w2 += w * w;

    uint8_t rev = 0;
    uint8_t x = (uint8_t)n;
    for (int b = 0; b < 8; ++b) {
      rev = (uint8_t)((rev << 1) | (x & 1));
      x >>= 1;
    }
    s_bitrev256[n] = rev;
  }
  s_win_norm = FS_DECIM * sum_w2;

  for (int k = 0; k < WELCH_NPERSEG / 2; ++k) {
    const float ang = -two_pi_over_n * (float)k;
    s_twiddle_re[k] = cosf(ang);
    s_twiddle_im[k] = sinf(ang);
  }
}

// Radix-2 FFT 256 điểm in-place trên s_fft_re, s_fft_im
static inline void fft256_radix2(float re[WELCH_NPERSEG], float im[WELCH_NPERSEG]) {
  for (int len = 2; len <= WELCH_NPERSEG; len <<= 1) {
    const int half = len >> 1;
    const int step = WELCH_NPERSEG / len;
    for (int i = 0; i < WELCH_NPERSEG; i += len) {
      int tw_idx = 0;
      for (int j = 0; j < half; ++j) {
        const float wr = s_twiddle_re[tw_idx];
        const float wi = s_twiddle_im[tw_idx];
        tw_idx += step;
        const int u = i + j;
        const int v = u + half;
        const float tr = wr * re[v] - wi * im[v];
        const float ti = wr * im[v] + wi * re[v];
        re[v] = re[u] - tr;
        im[v] = im[u] - ti;
        re[u] = re[u] + tr;
        im[u] = im[u] + ti;
      }
    }
  }
}

// Bước 1: Hạ mẫu 100 -> 50 Hz cho bản ghi 10 s (1000 mẫu -> 500 mẫu) bằng FIR polyphase 41 taps
static void __attribute__((noinline)) step_downsample_10s(
    const float raw_in[B4_REC_RAW_C][B4_REC_RAW_T],
    float out_50[B4_REC_RAW_C][B4_REC_DECIM_T]) {
  const int half_len = (B4_FIR_LEN - 1) / 2;  // 20
  for (int c = 0; c < B4_REC_RAW_C; ++c) {
    const float* x = raw_in[c];
    float* y = out_50[c];
    for (int n = 0; n < B4_REC_DECIM_T; ++n) {
      const int center = 2 * n + half_len;
      int k_min = 0;
      if (center - (B4_REC_RAW_T - 1) > k_min) k_min = center - (B4_REC_RAW_T - 1);
      int k_max = B4_FIR_LEN - 1;
      if (center < k_max) k_max = center;
      float acc = 0.0f;
      for (int k = k_min; k <= k_max; ++k) {
        acc += g_b4_fir_kaiser41[k] * x[center - k];
      }
      y[n] = acc;
    }
  }
}

// Bước 2: Lọc thông dải 0.5-20 Hz bằng IIR Butterworth bậc 4 nhân quả (4 tầng biquad DF2T)
static void __attribute__((noinline)) step_bandpass_10s_causal(
    const float in_50[B4_REC_RAW_C][B4_REC_DECIM_T],
    float out_bp[B4_REC_RAW_C][B4_REC_DECIM_T]) {
  for (int c = 0; c < B4_REC_RAW_C; ++c) {
    memcpy(out_bp[c], in_50[c], sizeof(float) * B4_REC_DECIM_T);
    float* buf = out_bp[c];
    for (int s = 0; s < B4_SOS_SECTIONS; ++s) {
      const float b0 = g_b4_butter4_bp_sos[s][0];
      const float b1 = g_b4_butter4_bp_sos[s][1];
      const float b2 = g_b4_butter4_bp_sos[s][2];
      const float a1 = g_b4_butter4_bp_sos[s][4];
      const float a2 = g_b4_butter4_bp_sos[s][5];
      float z1 = 0.0f, z2 = 0.0f;
      for (int n = 0; n < B4_REC_DECIM_T; ++n) {
        const float x = buf[n];
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        buf[n] = y;
      }
    }
  }
}

// Bước 3: Dựng 8 kênh (ax, ay, az, wx, wy, wz, |a|, |w|), mỗi kênh 500 mẫu @ 50 Hz
static void __attribute__((noinline)) step_make_8channels_10s(
    const float in_bp[B4_REC_RAW_C][B4_REC_DECIM_T],
    float out_ch8[B4_REC_OUT_C][B4_REC_DECIM_T]) {
  for (int c = 0; c < B4_REC_RAW_C; ++c) {
    memcpy(out_ch8[c], in_bp[c], sizeof(float) * B4_REC_DECIM_T);
  }
  float* acc_mag = out_ch8[6];
  float* gyr_mag = out_ch8[7];
  for (int n = 0; n < B4_REC_DECIM_T; ++n) {
    const float ax = in_bp[0][n], ay = in_bp[1][n], az = in_bp[2][n];
    const float wx = in_bp[3][n], wy = in_bp[4][n], wz = in_bp[5][n];
    acc_mag[n] = sqrtf(ax * ax + ay * ay + az * az);
    gyr_mag[n] = sqrtf(wx * wx + wy * wy + wz * wz);
  }
}

static inline float percentile_linear(const float* sorted_x, int n, float q_pct) {
  const float pos = (float)(n - 1) * (q_pct / 100.0f);
  const int lo = (int)floorf(pos);
  const int hi = (int)ceilf(pos);
  const float frac = pos - (float)lo;
  return sorted_x[lo] + frac * (sorted_x[hi] - sorted_x[lo]);
}

static inline float trapz_uniform(const float* p, int idx_start, int idx_end_incl, float df) {
  const int count = idx_end_incl - idx_start + 1;
  if (count < 2) return 0.0f;
  float sum_mid = 0.0f;
  for (int k = idx_start + 1; k < idx_end_incl; ++k) {
    sum_mid += p[k];
  }
  return df * (0.5f * (p[idx_start] + p[idx_end_incl]) + sum_mid);
}

// Bước 4: Tính 12 đặc trưng cho 1 kênh 500 mẫu @ 50 Hz (khớp src/features.py: channel_features)
// Thứ tự 12 đặc trưng:
//   0: std, 1: iqr, 2: skew, 3: kurt, 4: rms_diff,
//   5: log_pow, 6: peak_f, 7: spec_ent, 8: bp_0.5_3, 9: bp_3_7, 10: bp_7_12, 11: bp_12_20
static void __attribute__((noinline)) extract_12_features_one_channel(
    const float* x,
    int n_samples,
    float out_12[B4_FEATS_PER_CH]) {
  // 1) Các mô-men thống kê (std, skew, kurtosis Fisher) và RMS sai phân bậc 1
  float sum = 0.0f;
  for (int i = 0; i < n_samples; ++i) sum += x[i];
  const float mean = sum / (float)n_samples;

  float m2 = 0.0f, m3 = 0.0f, m4 = 0.0f;
  for (int i = 0; i < n_samples; ++i) {
    const float d = x[i] - mean;
    const float d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= (float)n_samples;
  m3 /= (float)n_samples;
  m4 /= (float)n_samples;
  const float std_val = sqrtf(m2);
  const float skew_val = (m2 > 0.0f) ? (m3 / (m2 * std_val)) : 0.0f;
  const float kurt_val = (m2 > 0.0f) ? (m4 / (m2 * m2) - 3.0f) : -3.0f;

  float sum_diff2 = 0.0f;
  for (int i = 0; i < n_samples - 1; ++i) {
    const float df = x[i + 1] - x[i];
    sum_diff2 += df * df;
  }
  const float rms_diff_val = sqrtf(sum_diff2 / (float)(n_samples - 1));

  // 2) IQR (분위수 75% - 25%, nội suy tuyến tính)
  memcpy(s_sort_buf, x, sizeof(float) * n_samples);
  std::sort(s_sort_buf, s_sort_buf + n_samples);
  const float iqr_val = percentile_linear(s_sort_buf, n_samples, 75.0f) -
                        percentile_linear(s_sort_buf, n_samples, 25.0f);

  // 3) Welch PSD: nperseg=256, nstep=128, detrend='constant', Hann window, scaling='density'
  const int n_segs = (n_samples - WELCH_NPERSEG) / WELCH_NSTEP + 1;
  memset(s_psd, 0, sizeof(s_psd));

  for (int seg = 0; seg < n_segs; ++seg) {
    const float* seg_ptr = x + seg * WELCH_NSTEP;
    float seg_sum = 0.0f;
    for (int i = 0; i < WELCH_NPERSEG; ++i) seg_sum += seg_ptr[i];
    const float seg_mean = seg_sum / (float)WELCH_NPERSEG;

    for (int i = 0; i < WELCH_NPERSEG; ++i) {
      const float v = (seg_ptr[i] - seg_mean) * s_hann256[i];
      const uint8_t br = s_bitrev256[i];
      s_fft_re[br] = v;
      s_fft_im[br] = 0.0f;
    }
    fft256_radix2(s_fft_re, s_fft_im);

    for (int k = 0; k < WELCH_NFREQ; ++k) {
      const float re = s_fft_re[k];
      const float im = s_fft_im[k];
      float p = (re * re + im * im) / s_win_norm;
      if (k > 0 && k < WELCH_NFREQ - 1) p *= 2.0f;
      s_psd[k] += p;
    }
  }
  const float inv_nsegs = 1.0f / (float)n_segs;
  for (int k = 0; k < WELCH_NFREQ; ++k) s_psd[k] *= inv_nsegs;

  // Tần số f[k] = k * (50.0 / 256.0) = k * 0.1953125 Hz
  // Dải:
  //   tot      : k = 0 .. 128
  //   (0.5, 3) : f >= 0.5 & f < 3.0   -> k = 3 .. 15
  //   (3, 7)   : f >= 3.0 & f < 7.0   -> k = 16 .. 35
  //   (7, 12)  : f >= 7.0 & f < 12.0  -> k = 36 .. 61
  //   (12, 20) : f >= 12.0 & f < 20.0 -> k = 62 .. 102
  //   m = (f >= 0.5) & (f <= 20.0)    -> k = 3 .. 102
  const float df = FS_DECIM / (float)WELCH_NPERSEG;
  const float tot = trapz_uniform(s_psd, 0, WELCH_NFREQ - 1, df) + 1e-12f;
  const float bp_0p5_3 = trapz_uniform(s_psd, 3, 15, df) / tot;
  const float bp_3_7   = trapz_uniform(s_psd, 16, 35, df) / tot;
  const float bp_7_12  = trapz_uniform(s_psd, 36, 61, df) / tot;
  const float bp_12_20 = trapz_uniform(s_psd, 62, 102, df) / tot;

  int best_k = 3;
  float best_p = s_psd[3];
  float sum_pm = 0.0f;
  for (int k = 3; k <= 102; ++k) {
    const float pk = s_psd[k];
    sum_pm += pk;
    if (pk > best_p) {
      best_p = pk;
      best_k = k;
    }
  }
  const float peak_f = (float)best_k * df;
  const float denom_pm = sum_pm + 1e-12f;
  float sent = 0.0f;
  for (int k = 3; k <= 102; ++k) {
    const float pn = s_psd[k] / denom_pm;
    sent -= pn * logf(pn + 1e-12f);
  }

  out_12[0]  = std_val;
  out_12[1]  = iqr_val;
  out_12[2]  = skew_val;
  out_12[3]  = kurt_val;
  out_12[4]  = rms_diff_val;
  out_12[5]  = logf(tot);
  out_12[6]  = peak_f;
  out_12[7]  = sent;
  out_12[8]  = bp_0p5_3;
  out_12[9]  = bp_3_7;
  out_12[10] = bp_7_12;
  out_12[11] = bp_12_20;
}

static void __attribute__((noinline)) step_extract_96_features(
    const float in_ch8[B4_REC_OUT_C][B4_REC_DECIM_T],
    float out_feats96[B4_REC_N_FEATS]) {
  for (int c = 0; c < B4_REC_OUT_C; ++c) {
    extract_12_features_one_channel(in_ch8[c], B4_REC_DECIM_T, &out_feats96[c * B4_FEATS_PER_CH]);
  }
}

static void print_stat_line(const char* name, std::vector<int64_t>& v) {
  std::sort(v.begin(), v.end());
  const int n = (int)v.size();
  double sum = 0.0;
  for (int64_t x : v) sum += (double)x;
  const double mean_us = sum / n;
  const int64_t p50_us = v[n / 2];
  const int64_t p95_us = v[(int)(n * 0.95)];
  const int64_t min_us = v.front();
  const int64_t max_us = v.back();
  printf("#step_latency step=%s mean_us=%.2f p50_us=%lld p95_us=%lld min_us=%lld max_us=%lld mean_ms=%.4f p95_ms=%.4f n=%d\n",
         name, mean_us, (long long)p50_us, (long long)p95_us, (long long)min_us, (long long)max_us,
         mean_us / 1000.0, (double)p95_us / 1000.0, n);
}

static void run() {
  printf("\n#B4 BEGIN\n");
  init_welch_tables();

  esp_chip_info_t ci;
  esp_chip_info(&ci);
  const size_t int_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t int_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
  const size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  const size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  const size_t dsp_feat_ram = sizeof(s_x50) + sizeof(s_xf) + sizeof(s_ch8) + sizeof(s_feats96) +
                              sizeof(s_sort_buf) + sizeof(s_hann256) + sizeof(s_twiddle_re) +
                              sizeof(s_twiddle_im) + sizeof(s_bitrev256) + sizeof(s_fft_re) +
                              sizeof(s_fft_im) + sizeof(s_psd);
  const size_t rf_leaves_bytes = sizeof(pd_rf_leaves);

  printf("#chip cores=%d rev=%d cpu_mhz=%d internal_total=%u internal_free=%u psram_total=%u psram_free=%u\n",
         ci.cores, ci.revision, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
         (unsigned)int_total, (unsigned)int_free,
         (unsigned)psram_total, (unsigned)psram_free);
  printf("#memory n_features=%d n_trees=%d max_depth=%d rf_leaves_rodata_bytes=%u rf_ram_stack_bytes=64 feat_dsp_static_ram_bytes=%u\n",
         B4_N_FEAT, B4_N_TREES, B4_MAX_DEPTH, (unsigned)rf_leaves_bytes, (unsigned)dsp_feat_ram);

  // =========================================================================
  // PHẦN 1: Chạy Random Forest (emlearn float) trên 20 vector đặc trưng test fold 0
  //         So sánh nhãn và xác suất với scikit-learn
  // =========================================================================
  printf("#B4 RF COMPARISON BEGIN n_test=%d n_features=%d\n", B4_N_TEST, B4_N_FEAT);
  int match_label_soft = 0;
  int match_label_hard = 0;
  int match_eml_soft_bitexact = 0;
  float max_dp_soft_vs_sk = 0.0f;
  float mean_dp_soft_vs_sk = 0.0f;
  float max_dp_hard_vs_sk = 0.0f;

  for (int i = 0; i < B4_N_TEST; ++i) {
    float out_soft[2] = {0.0f, 0.0f};
    float out_hard[2] = {0.0f, 0.0f};
    pd_rf_predict_proba(g_b4_test_X[i], B4_N_FEAT, out_soft, 2);
    const int32_t y_soft = pd_rf_predict(g_b4_test_X[i], B4_N_FEAT);
    pd_rf_hard_predict_proba(g_b4_test_X[i], B4_N_FEAT, out_hard, 2);
    const int32_t y_hard = pd_rf_hard_predict(g_b4_test_X[i], B4_N_FEAT);

    const float p_soft = out_soft[1];
    const float p_hard = out_hard[1];
    const float p_sk = g_b4_test_prob_sklearn[i];
    const int32_t y_sk = g_b4_test_y_sklearn[i];

    const bool ok_soft = (y_soft == y_sk);
    const bool ok_hard = (y_hard == y_sk);
    if (ok_soft) match_label_soft++;
    if (ok_hard) match_label_hard++;
    if (fabsf(p_soft - g_b4_test_prob_eml_soft[i]) < 1e-6f) match_eml_soft_bitexact++;

    const float dp_soft = fabsf(p_soft - p_sk);
    const float dp_hard = fabsf(p_hard - p_sk);
    if (dp_soft > max_dp_soft_vs_sk) max_dp_soft_vs_sk = dp_soft;
    if (dp_hard > max_dp_hard_vs_sk) max_dp_hard_vs_sk = dp_hard;
    mean_dp_soft_vs_sk += dp_soft;

    printf("#rf_cmp idx=%d sid=%s y_true=%d y_sklearn=%d prob_sklearn=%.6f "
           "y_board_soft=%d prob_board_soft=%.6f dp_soft=%.6f match_soft=%d "
           "y_board_hard=%d prob_board_hard=%.6f dp_hard=%.6f match_hard=%d\n",
           i, g_b4_test_sid[i], (int)g_b4_test_y_true[i], (int)y_sk, p_sk,
           (int)y_soft, p_soft, dp_soft, (int)ok_soft,
           (int)y_hard, p_hard, dp_hard, (int)ok_hard);
    fflush(stdout);
    if ((i & 3) == 3) vTaskDelay(1);
  }
  mean_dp_soft_vs_sk /= (float)B4_N_TEST;
  printf("#rf_summary label_match_soft=%d/%d label_match_hard=%d/%d eml_ref_exact=%d/%d "
         "max_abs_dp_soft=%.6f mean_abs_dp_soft=%.6f max_abs_dp_hard=%.6f\n",
         match_label_soft, B4_N_TEST, match_label_hard, B4_N_TEST,
         match_eml_soft_bitexact, B4_N_TEST,
         max_dp_soft_vs_sk, mean_dp_soft_vs_sk, max_dp_hard_vs_sk);
  printf("#B4 RF COMPARISON END\n");
  fflush(stdout);
  vTaskDelay(2);

  // Đo thời gian suy luận RF (1000 lần lặp trên 20 vector test)
  volatile float sink_rf = 0.0f;
  for (int i = 0; i < WARMUP_RF; ++i) {
    float out_p[2];
    pd_rf_predict_proba(g_b4_test_X[i % B4_N_TEST], B4_N_FEAT, out_p, 2);
    sink_rf = out_p[1];
  }

  std::vector<int64_t> t_rf_soft(N_TIMED_RF), t_rf_hard(N_TIMED_RF);
  for (int i = 0; i < N_TIMED_RF; ++i) {
    const float* xvec = g_b4_test_X[i % B4_N_TEST];
    float out_p[2];
    const int64_t t0 = esp_timer_get_time();
    pd_rf_predict_proba(xvec, B4_N_FEAT, out_p, 2);
    const int64_t t1 = esp_timer_get_time();
    const int32_t cls = pd_rf_hard_predict(xvec, B4_N_FEAT);
    const int64_t t2 = esp_timer_get_time();
    sink_rf = out_p[1] + (float)cls;
    t_rf_soft[i] = t1 - t0;
    t_rf_hard[i] = t2 - t1;
  }
  (void)sink_rf;
  print_stat_line("rf_infer_proba_soft_100trees", t_rf_soft);
  print_stat_line("rf_infer_label_hard_100trees", t_rf_hard);

  // =========================================================================
  // PHẦN 2: Trích xuất đặc trưng trên bo cho MỘT bản ghi 10 s x 6 kênh @ 100 Hz
  //         (12 đặc trưng/kênh x 8 kênh = 96 đặc trưng) + so sánh với Python
  // =========================================================================
  printf("#B4 FEATURE EXTRACTION BEGIN file=\"%s\"\n", g_b4_rec_file);
  step_downsample_10s(g_b4_rec_raw_10s, s_x50);
  step_bandpass_10s_causal(s_x50, s_xf);
  step_make_8channels_10s(s_xf, s_ch8);
  step_extract_96_features(s_ch8, s_feats96);

  float max_rel_err = 0.0f;
  float mean_rel_err = 0.0f;
  float max_abs_err = 0.0f;
  int worst_idx = 0;

  for (int j = 0; j < B4_REC_N_FEATS; ++j) {
    const float v_bo = s_feats96[j];
    const float v_py = g_b4_ref_feats96_causal[j];
    const float abs_e = fabsf(v_bo - v_py);
    const float denom = fmaxf(fabsf(v_py), 1e-12f);
    const float rel_e = abs_e / denom;
    mean_rel_err += rel_e;
    if (abs_e > max_abs_err) max_abs_err = abs_e;
    if (rel_e > max_rel_err) {
      max_rel_err = rel_e;
      worst_idx = j;
    }
    printf("#feat_cmp idx=%d name=%s board=%.8e py_causal=%.8e abs_err=%.4e rel_err=%.4e py_zp=%.8e\n",
           j, g_b4_feat96_names[j], v_bo, v_py, abs_e, rel_e, g_b4_ref_feats96_zp[j]);
    fflush(stdout);
    if ((j & 3) == 3) vTaskDelay(1);
  }
  mean_rel_err /= (float)B4_REC_N_FEATS;
  printf("#feat_summary n_feats=%d max_rel_err=%.6e mean_rel_err=%.6e max_abs_err=%.6e worst_idx=%d worst_name=%s\n",
         B4_REC_N_FEATS, max_rel_err, mean_rel_err, max_abs_err, worst_idx, g_b4_feat96_names[worst_idx]);
  printf("#B4 FEATURE EXTRACTION END\n");
  fflush(stdout);
  vTaskDelay(2);

  // Đo thời gian tính đặc trưng trên bo cho 1 bản ghi 10 s x 6 kênh @ 100 Hz (100 lần lặp)
  volatile float sink_feat = 0.0f;
  for (int i = 0; i < WARMUP_FEAT; ++i) {
    step_downsample_10s(g_b4_rec_raw_10s, s_x50);
    step_bandpass_10s_causal(s_x50, s_xf);
    step_make_8channels_10s(s_xf, s_ch8);
    step_extract_96_features(s_ch8, s_feats96);
    sink_feat = s_feats96[0];
  }

  std::vector<int64_t> t_f_down(N_TIMED_FEAT), t_f_bp(N_TIMED_FEAT), t_f_ch8(N_TIMED_FEAT),
                       t_f_ext96(N_TIMED_FEAT), t_f_total(N_TIMED_FEAT);
  for (int i = 0; i < N_TIMED_FEAT; ++i) {
    const int64_t t0 = esp_timer_get_time();
    step_downsample_10s(g_b4_rec_raw_10s, s_x50);
    const int64_t t1 = esp_timer_get_time();
    step_bandpass_10s_causal(s_x50, s_xf);
    const int64_t t2 = esp_timer_get_time();
    step_make_8channels_10s(s_xf, s_ch8);
    const int64_t t3 = esp_timer_get_time();
    step_extract_96_features(s_ch8, s_feats96);
    const int64_t t4 = esp_timer_get_time();
    sink_feat = s_feats96[i % B4_REC_N_FEATS];

    t_f_down[i]  = t1 - t0;
    t_f_bp[i]    = t2 - t1;
    t_f_ch8[i]   = t3 - t2;
    t_f_ext96[i] = t4 - t3;
    t_f_total[i] = t4 - t0;
    if ((i & 7) == 0) vTaskDelay(1);
  }
  (void)sink_feat;

  print_stat_line("rec10s_1_downsample_100to50", t_f_down);
  print_stat_line("rec10s_2_bandpass_biquad4", t_f_bp);
  print_stat_line("rec10s_3_make_8channels", t_f_ch8);
  print_stat_line("rec10s_4_extract_96feats_welch256", t_f_ext96);
  print_stat_line("rec10s_total_feature_pipeline", t_f_total);
  printf("#B4 END\n");
  fflush(stdout);
}

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1500));
  run();
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
