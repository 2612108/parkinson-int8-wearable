// b2_e2e — Đo độ trễ ĐẦU-CUỐI (End-to-End) và chế độ đo năng lượng 3 pha trên ESP32-S3
// Mô hình: reports/p56a/models/PD_vs_HC_raw_qat_fold0.tflite (DS-TCN INT8, ESP-NN)
// Đường ống trên bo cho 1 cửa sổ 4 s (400 mẫu x 6 kênh @ 100 Hz):
//   1) Hạ mẫu 100 -> 50 Hz bằng bộ lọc FIR polyphase (41 taps Kaiser beta=5.0, lấy 1/2 -> 200 mẫu x 6)
//   2) Lọc thông dải 0.5-20 Hz bằng IIR Butterworth bậc 4 nhân quả (4 tầng biquad Direct Form II Transposed, KHÔNG zero-phase)
//   3) Tạo 8 kênh (ax, ay, az, wx, wy, wz, |a|, |w|) -> 200 x 8
//   4) Chuẩn hóa z-score bằng mu/sd (norm_PD_vs_HC_raw_fold0.npz) + lượng tử hóa INT8
//   5) Suy luận mô hình TFLite Micro (ESP-NN) + tính xác suất PD
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
#include "esp_sleep.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "generated/model_data.h"
#include "generated/test_data.h"
#include "generated/op_resolver.h"
#include "generated/e2e_data.h"

#ifndef ARENA_BYTES
#define ARENA_BYTES (160 * 1024)
#endif
#define WARMUP 10
#define N_TIMED 1000
#define PHASE_DURATION_US (300LL * 1000000LL)
#define DUTY_PERIOD_US    (4LL * 1000000LL)

// Bộ đệm tĩnh cho đường ống DSP trên bo (căn chỉnh 16 byte trong SRAM nội)
alignas(16) static float s_x50[E2E_RAW_C][E2E_DECIM_T];     // 6 x 200 x 4B = 4800 B
alignas(16) static float s_xf[E2E_RAW_C][E2E_DECIM_T];      // 6 x 200 x 4B = 4800 B
alignas(16) static float s_x8[E2E_DECIM_T][E2E_OUT_C];      // 200 x 8 x 4B = 6400 B
alignas(16) static float s_xn[E2E_DECIM_T][E2E_OUT_C];      // 200 x 8 x 4B = 6400 B

// Bước 1: Hạ mẫu 100 -> 50 Hz bằng FIR polyphase (41 taps, hệ số đối xứng Kaiser beta=5.0)
static void __attribute__((noinline)) step_downsample_polyphase(
    const float raw_in[E2E_RAW_C][E2E_RAW_T],
    float out_50[E2E_RAW_C][E2E_DECIM_T]) {
  const int half_len = (E2E_FIR_LEN - 1) / 2;  // 20
  for (int c = 0; c < E2E_RAW_C; ++c) {
    const float* x = raw_in[c];
    float* y = out_50[c];
    for (int n = 0; n < E2E_DECIM_T; ++n) {
      const int center = 2 * n + half_len;
      int k_min = 0;
      if (center - (E2E_RAW_T - 1) > k_min) k_min = center - (E2E_RAW_T - 1);
      int k_max = E2E_FIR_LEN - 1;
      if (center < k_max) k_max = center;
      float acc = 0.0f;
      for (int k = k_min; k <= k_max; ++k) {
        acc += g_fir_kaiser41[k] * x[center - k];
      }
      y[n] = acc;
    }
  }
}

// Bước 2: Lọc thông dải 0.5-20 Hz bằng IIR Butterworth bậc 4 nhân quả (4 tầng biquad DF2T)
static void __attribute__((noinline)) step_bandpass_biquad_causal(
    const float in_50[E2E_RAW_C][E2E_DECIM_T],
    float out_bp[E2E_RAW_C][E2E_DECIM_T]) {
  for (int c = 0; c < E2E_RAW_C; ++c) {
    memcpy(out_bp[c], in_50[c], sizeof(float) * E2E_DECIM_T);
    float* buf = out_bp[c];
    for (int s = 0; s < E2E_SOS_SECTIONS; ++s) {
      const float b0 = g_butter4_bp_sos[s][0];
      const float b1 = g_butter4_bp_sos[s][1];
      const float b2 = g_butter4_bp_sos[s][2];
      // a0 = 1.0f
      const float a1 = g_butter4_bp_sos[s][4];
      const float a2 = g_butter4_bp_sos[s][5];
      float z1 = 0.0f, z2 = 0.0f;
      for (int n = 0; n < E2E_DECIM_T; ++n) {
        const float x = buf[n];
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        buf[n] = y;
      }
    }
  }
}

// Bước 3: Tạo 8 kênh (ax, ay, az, wx, wy, wz, |a|, |w|)
static void __attribute__((noinline)) step_make_8channels(
    const float in_bp[E2E_RAW_C][E2E_DECIM_T],
    float out_x8[E2E_DECIM_T][E2E_OUT_C]) {
  for (int n = 0; n < E2E_DECIM_T; ++n) {
    const float ax = in_bp[0][n];
    const float ay = in_bp[1][n];
    const float az = in_bp[2][n];
    const float wx = in_bp[3][n];
    const float wy = in_bp[4][n];
    const float wz = in_bp[5][n];
    out_x8[n][0] = ax;
    out_x8[n][1] = ay;
    out_x8[n][2] = az;
    out_x8[n][3] = wx;
    out_x8[n][4] = wy;
    out_x8[n][5] = wz;
    out_x8[n][6] = sqrtf(ax * ax + ay * ay + az * az);
    out_x8[n][7] = sqrtf(wx * wx + wy * wy + wz * wz);
  }
}

// Bước 4a: Chuẩn hóa z-score bằng mu/sd từ norm_PD_vs_HC_raw_fold0.npz
static void __attribute__((noinline)) step_normalize(
    const float in_x8[E2E_DECIM_T][E2E_OUT_C],
    float out_xn[E2E_DECIM_T][E2E_OUT_C]) {
  for (int n = 0; n < E2E_DECIM_T; ++n) {
    for (int c = 0; c < E2E_OUT_C; ++c) {
      out_xn[n][c] = (in_x8[n][c] - g_norm_mu[c]) / g_norm_sd[c];
    }
  }
}

// Bước 4b: Lượng tử hóa float32 -> int8 theo scale/zero_point của tensor đầu vào
static void __attribute__((noinline)) step_quantize(
    const float in_xn[E2E_DECIM_T][E2E_OUT_C],
    int8_t* out_q,
    float inv_scale,
    int zp) {
  const float* src = &in_xn[0][0];
  const int total = E2E_DECIM_T * E2E_OUT_C;
  for (int i = 0; i < total; ++i) {
    long q = lrintf(src[i] * inv_scale) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    out_q[i] = (int8_t)q;
  }
}

struct E2EOutput {
  int8_t q_out;
  float logit;
  float prob;
  int32_t qin_cksum;
};

static E2EOutput run_e2e_once(
    const float raw_in[E2E_RAW_C][E2E_RAW_T],
    tflite::MicroInterpreter& interp,
    TfLiteTensor* in,
    TfLiteTensor* out,
    float inv_in_scale,
    int in_zp,
    float out_scale,
    int out_zp) {
  step_downsample_polyphase(raw_in, s_x50);
  step_bandpass_biquad_causal(s_x50, s_xf);
  step_make_8channels(s_xf, s_x8);
  step_normalize(s_x8, s_xn);
  step_quantize(s_xn, in->data.int8, inv_in_scale, in_zp);
  interp.Invoke();
  const int8_t q_out = out->data.int8[0];
  const float logit = ((int)q_out - out_zp) * out_scale;
  const float prob = 1.0f / (1.0f + expf(-logit));
  int32_t cks = 0;
  for (int i = 0; i < IN_SIZE; ++i) cks += in->data.int8[i];
  return E2EOutput{q_out, logit, prob, cks};
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
  printf("\n#B2 BEGIN\n");
  esp_chip_info_t ci;
  esp_chip_info(&ci);
  const size_t int_free_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t int_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
  const size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
  const size_t psram_free_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  printf("#chip cores=%d rev=%d cpu_mhz=%d internal_total=%u internal_free_before=%u psram_total=%u psram_free_before=%u\n",
         ci.cores, ci.revision, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
         (unsigned)int_total, (unsigned)int_free_before,
         (unsigned)psram_total, (unsigned)psram_free_before);
#if CONFIG_NN_OPTIMIZED
  printf("#esp_nn=optimized\n");
#else
  printf("#esp_nn=ansi\n");
#endif

  const char* arena_mem = "internal";
  uint8_t* arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) {
    arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_SPIRAM);
    arena_mem = "psram";
  }
  if (!arena) {
    printf("#ERR arena alloc\n");
    return;
  }

  const tflite::Model* model = tflite::GetModel(g_model);
  static OpResolver resolver;
  register_ops(resolver);
  static tflite::MicroInterpreter interp(model, resolver, arena, ARENA_BYTES);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("#ERR AllocateTensors (tang ARENA_BYTES)\n");
    return;
  }
  TfLiteTensor* in = interp.input(0);
  TfLiteTensor* out = interp.output(0);
  const size_t dsp_buf_bytes = sizeof(s_x50) + sizeof(s_xf) + sizeof(s_x8) + sizeof(s_xn);
  const size_t int_free_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  printf("#memory model_bytes=%u arena_alloc=%u arena_used=%u arena_mem=%s dsp_static_buf_bytes=%u total_e2e_ram_bytes=%u internal_free_after=%u in_bytes=%u out_bytes=%u\n",
         g_model_len, (unsigned)ARENA_BYTES, (unsigned)interp.arena_used_bytes(), arena_mem,
         (unsigned)dsp_buf_bytes, (unsigned)(interp.arena_used_bytes() + dsp_buf_bytes),
         (unsigned)int_free_after, (unsigned)in->bytes, (unsigned)out->bytes);

  const float in_scale = in->params.scale;
  const int in_zp = (int)in->params.zero_point;
  const float inv_in_scale = 1.0f / in_scale;
  const float out_scale = out->params.scale;
  const int out_zp = (int)out->params.zero_point;
  printf("#quant in_scale=%.8g in_zp=%d out_scale=%.8g out_zp=%d\n",
         in_scale, in_zp, out_scale, out_zp);

  // Kiểm tra nhanh bit-exact với 10 mẫu test_data (TFLM vs TFLite PC)
  int mism = 0;
  for (int i = 0; i < N_TEST; ++i) {
    memcpy(in->data.int8, g_test_in[i], IN_SIZE);
    if (interp.Invoke() != kTfLiteOk) {
      printf("#ERR invoke test %d\n", i);
      return;
    }
    if (memcmp(out->data.int8, g_ref_out[i], OUT_SIZE) != 0) mism++;
  }
  printf("#bitexact_model_only matched=%d/%d\n", N_TEST - mism, N_TEST);

  // =========================================================================
  // PHẦN 1A: So sánh đầu ra ĐẦU-CUỐI trên bo (lọc IIR nhân quả) vs Python
  // =========================================================================
  printf("#B2 FILTER COMPARISON BEGIN\n");
  printf("#primary_pads_file=%s start_sample=%d window_s=4.0 fs_in=100 fs_out=50\n",
         g_e2e_win_file[0], g_e2e_win_start[0]);
  int causal_exact = 0;
  for (int w = 0; w < E2E_N_WIN; ++w) {
    E2EOutput eo = run_e2e_once(g_e2e_raw_in[w], interp, in, out, inv_in_scale, in_zp, out_scale, out_zp);
    const bool match_causal = (eo.q_out == g_ref_q_causal[w]);
    if (match_causal) causal_exact++;
    const float diff_causal = fabsf(eo.prob - g_ref_prob_causal[w]);
    const float diff_zp_win = eo.prob - g_ref_prob_zerophase_win[w];
    const float diff_zp_full = eo.prob - g_ref_prob_zerophase_full[w];
    printf("#win_cmp idx=%d label=%s file=%s start=%d qin_cksum=%ld ref_cksum=%ld "
           "q_board=%d prob_board=%.6f q_py_causal=%d prob_py_causal=%.6f causal_match=%d abs_diff_causal=%.6f "
           "q_py_zp_win=%d prob_py_zp_win=%.6f diff_board_vs_zp_win=%+.6f "
           "q_py_zp_full=%d prob_py_zp_full=%.6f diff_board_vs_zp_full=%+.6f desc=\"%s\"\n",
           w, g_e2e_win_label[w], g_e2e_win_file[w], g_e2e_win_start[w],
           (long)eo.qin_cksum, (long)g_ref_qin_checksum[w],
           (int)eo.q_out, eo.prob,
           (int)g_ref_q_causal[w], g_ref_prob_causal[w], (int)match_causal, diff_causal,
           (int)g_ref_q_zerophase_win[w], g_ref_prob_zerophase_win[w], diff_zp_win,
           (int)g_ref_q_zerophase_full[w], g_ref_prob_zerophase_full[w], diff_zp_full,
           g_e2e_win_desc[w]);
  }
  printf("#causal_board_vs_py_matched=%d/%d\n", causal_exact, E2E_N_WIN);
  printf("#B2 FILTER COMPARISON END\n");

  // =========================================================================
  // PHẦN 1B: Đo độ trễ ĐẦU-CUỐI từng bước và tổng trên 1000 lần (cửa sổ 0)
  // =========================================================================
  printf("#B2 LATENCY BENCHMARK BEGIN n=%d warmup=%d file=%s\n",
         N_TIMED, WARMUP, g_e2e_win_file[0]);
  for (int i = 0; i < WARMUP; ++i) {
    (void)run_e2e_once(g_e2e_raw_in[0], interp, in, out, inv_in_scale, in_zp, out_scale, out_zp);
  }

  std::vector<int64_t> t_down(N_TIMED), t_bp(N_TIMED), t_ch8(N_TIMED),
                       t_norm(N_TIMED), t_quant(N_TIMED), t_norm_quant(N_TIMED),
                       t_prep(N_TIMED), t_inv(N_TIMED), t_total(N_TIMED);

  volatile float sink_prob = 0.0f;
  for (int i = 0; i < N_TIMED; ++i) {
    const int64_t t0 = esp_timer_get_time();
    step_downsample_polyphase(g_e2e_raw_in[0], s_x50);
    const int64_t t1 = esp_timer_get_time();
    step_bandpass_biquad_causal(s_x50, s_xf);
    const int64_t t2 = esp_timer_get_time();
    step_make_8channels(s_xf, s_x8);
    const int64_t t3 = esp_timer_get_time();
    step_normalize(s_x8, s_xn);
    const int64_t t4 = esp_timer_get_time();
    step_quantize(s_xn, in->data.int8, inv_in_scale, in_zp);
    const int64_t t5 = esp_timer_get_time();
    interp.Invoke();
    const int8_t q_out = out->data.int8[0];
    const float logit = ((int)q_out - out_zp) * out_scale;
    sink_prob = 1.0f / (1.0f + expf(-logit));
    const int64_t t6 = esp_timer_get_time();

    t_down[i] = t1 - t0;
    t_bp[i] = t2 - t1;
    t_ch8[i] = t3 - t2;
    t_norm[i] = t4 - t3;
    t_quant[i] = t5 - t4;
    t_norm_quant[i] = t5 - t3;
    t_prep[i] = t5 - t0;
    t_inv[i] = t6 - t5;
    t_total[i] = t6 - t0;

    if ((i & 63) == 0) vTaskDelay(1);
  }
  (void)sink_prob;

  print_stat_line("1_downsample_100to50_fir41", t_down);
  print_stat_line("2_bandpass_0p5_20_biquad4", t_bp);
  print_stat_line("3_make_8channels_normvec", t_ch8);
  print_stat_line("4a_normalize_zscore", t_norm);
  print_stat_line("4b_quantize_int8", t_quant);
  print_stat_line("4_norm_plus_quant", t_norm_quant);
  print_stat_line("preprocess_total", t_prep);
  print_stat_line("5_model_invoke_espnn", t_inv);
  print_stat_line("e2e_total", t_total);
  printf("#B2 LATENCY BENCHMARK END\n");

  // =========================================================================
  // PHẦN 2: Chế độ đo năng lượng (3 pha x 300 s: IDLE, INFER, SLEEP/INFER_DUTY)
  // =========================================================================
  printf("#B2 ENERGY MODE BEGIN phase_s=300\n");
  fflush(stdout);

  // --- Pha 1: IDLE (vòng lặp chờ 300 s, CPU 240 MHz) ---
  printf("PHASE IDLE WAIT 10s\n");
  fflush(stdout);
  vTaskDelay(pdMS_TO_TICKS(10000));
  const int64_t t_idle_start = esp_timer_get_time();
  printf("PHASE IDLE START t=%lld\n", (long long)t_idle_start);
  fflush(stdout);
  int n_infer_idle = 0;
  while (esp_timer_get_time() - t_idle_start < PHASE_DURATION_US) {
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  const int64_t t_idle_end = esp_timer_get_time();
  printf("PHASE IDLE END t=%lld duration_ms=%.1f n_infer=%d\n",
         (long long)t_idle_end, (double)(t_idle_end - t_idle_start) / 1000.0, n_infer_idle);
  fflush(stdout);

  // --- Pha 2: INFER (chạy suy luận đầu-cuối liên tục trong 300 s) ---
  printf("PHASE INFER WAIT 10s\n");
  fflush(stdout);
  vTaskDelay(pdMS_TO_TICKS(10000));
  const int64_t t_infer_start = esp_timer_get_time();
  printf("PHASE INFER START t=%lld\n", (long long)t_infer_start);
  fflush(stdout);
  int n_infer_cont = 0;
  while (esp_timer_get_time() - t_infer_start < PHASE_DURATION_US) {
    E2EOutput eo = run_e2e_once(g_e2e_raw_in[0], interp, in, out, inv_in_scale, in_zp, out_scale, out_zp);
    sink_prob = eo.prob;
    n_infer_cont++;
  }
  const int64_t t_infer_end = esp_timer_get_time();
  printf("PHASE INFER END t=%lld duration_ms=%.1f n_infer=%d\n",
         (long long)t_infer_end, (double)(t_infer_end - t_infer_start) / 1000.0, n_infer_cont);
  fflush(stdout);

  // --- Pha 3: SLEEP / INFER_DUTY (mỗi 4 s chạy 1 lần suy luận đầu-cuối rồi light-sleep, tổng 300 s) ---
  printf("PHASE INFER_DUTY WAIT 10s\n");
  fflush(stdout);
  vTaskDelay(pdMS_TO_TICKS(10000));
  const int64_t t_sleep_start = esp_timer_get_time();
  const int64_t t_sleep_target_end = t_sleep_start + PHASE_DURATION_US;
  printf("PHASE SLEEP START t=%lld mode=INFER_DUTY period_s=4\n", (long long)t_sleep_start);
  printf("PHASE INFER_DUTY START t=%lld period_s=4\n", (long long)t_sleep_start);
  fflush(stdout);
  fsync(fileno(stdout));
  vTaskDelay(pdMS_TO_TICKS(50));  // đảm bảo host nhận trọn dòng START trước khi tắt pad USB trong light-sleep

  int n_infer_duty = 0;
  int64_t total_active_us = 0;
  int64_t total_sleep_us = 0;
  int64_t slot_start = esp_timer_get_time();
  while (slot_start < t_sleep_target_end) {
    const int64_t t_a0 = esp_timer_get_time();
    E2EOutput eo = run_e2e_once(g_e2e_raw_in[0], interp, in, out, inv_in_scale, in_zp, out_scale, out_zp);
    sink_prob = eo.prob;
    n_infer_duty++;
    const int64_t t_a1 = esp_timer_get_time();
    total_active_us += (t_a1 - t_a0);

    int64_t next_boundary = slot_start + DUTY_PERIOD_US;
    if (next_boundary > t_sleep_target_end) next_boundary = t_sleep_target_end;
    const int64_t now = esp_timer_get_time();
    if (next_boundary > now + 2000LL) {
      const int64_t sleep_us = next_boundary - now;
      esp_sleep_enable_timer_wakeup((uint64_t)sleep_us);
      esp_light_sleep_start();
      const int64_t after_sleep = esp_timer_get_time();
      total_sleep_us += (after_sleep - now);
    }
    slot_start = esp_timer_get_time();
  }
  const int64_t t_sleep_end = esp_timer_get_time();

  // Chờ 1.5 s sau khi thoát light-sleep để cổng USB-Serial-JTAG (/dev/ttyACM0) re-enumerate trên PC
  // rồi in kết quả Pha 3 lặp lại 3 lần cách nhau 500 ms để script thu log luôn bắt được đầy đủ.
  for (int rep = 0; rep < 3; ++rep) {
    vTaskDelay(pdMS_TO_TICKS(600));
    printf("PHASE SLEEP END t=%lld duration_ms=%.1f n_infer=%d active_ms=%.2f light_sleep_ms=%.1f\n",
           (long long)t_sleep_end, (double)(t_sleep_end - t_sleep_start) / 1000.0,
           n_infer_duty, (double)total_active_us / 1000.0, (double)total_sleep_us / 1000.0);
    printf("PHASE INFER_DUTY END t=%lld duration_ms=%.1f n_infer=%d active_ms=%.2f light_sleep_ms=%.1f\n",
           (long long)t_sleep_end, (double)(t_sleep_end - t_sleep_start) / 1000.0,
           n_infer_duty, (double)total_active_us / 1000.0, (double)total_sleep_us / 1000.0);
    printf("#energy_phases idle_s=%.2f idle_n=%d infer_s=%.2f infer_n=%d sleep_s=%.2f sleep_n=%d active_duty_pct=%.4f\n",
           (double)(t_idle_end - t_idle_start) / 1e6, n_infer_idle,
           (double)(t_infer_end - t_infer_start) / 1e6, n_infer_cont,
           (double)(t_sleep_end - t_sleep_start) / 1e6, n_infer_duty,
           100.0 * (double)total_active_us / (double)(t_sleep_end - t_sleep_start));
    printf("#B2 END\n");
    fflush(stdout);
  }
}

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1500));
  run();
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
