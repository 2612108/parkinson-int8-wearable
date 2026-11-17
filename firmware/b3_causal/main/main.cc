// b3_causal — Phát lại TOÀN BỘ bản ghi PADS thật 100 Hz theo khối 2 s, cửa sổ 4 s hop 2 s
// Mô hình nhân quả: reports/p56x4/models/PD_vs_HC_raw_qat_fold0.tflite (DS-TCN INT8 huấn luyện --causal, ESP-NN)
// Chuẩn hóa: reports/p56x4/models/norm_PD_vs_HC_raw_fold0.npz
// Đường ống streaming trên bo theo từng khối 2 s (200 mẫu @ 100 Hz -> 100 mẫu @ 50 Hz):
//   1) Hạ mẫu 100 -> 50 Hz bằng FIR polyphase (41 taps Kaiser beta=5.0) trên luồng liên tục -> 100 mẫu x 6 kênh
//   2) Lọc thông dải 0.5-20 Hz bằng IIR Butterworth bậc 4 nhân quả LIÊN TỤC (4 tầng biquad DF2T, GIỮ trạng thái z1, z2
//      qua các khối 2 s, KHÔNG reset ở mỗi cửa sổ)
//   3) Dịch bộ đệm trượt 4 s (200 mẫu x 8 kênh) đi 100 mẫu (hop 2 s) và điền 100 mẫu mới của 8 kênh (ax..wz, |a|, |w|)
//   4) Từ khối thứ 2 trở đi (mỗi hop 2 s = 1 cửa sổ 4 s): chuẩn hóa z-score + lượng tử hóa INT8 + suy luận TFLM ESP-NN
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
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "generated/model_data.h"
#include "generated/test_data.h"
#include "generated/op_resolver.h"
#include "generated/b3_data.h"

#ifndef ARENA_BYTES
#define ARENA_BYTES (160 * 1024)
#endif
#define WARMUP_REPLAYS 5
#define N_TIMED_REPLAYS 100   // 100 lần phát lại toàn bộ bản ghi = 1000 khối 2s và 900 lần suy luận cửa sổ 4s

// Bộ đệm tĩnh cho đường ống streaming theo khối 2 s và cửa sổ trượt 4 s (căn chỉnh 16 byte trong SRAM nội)
alignas(16) static float s_blk50[B3_RAW_C][B3_BLOCK_DECIM_T];       // 6 x 100 x 4B = 2400 B (khối 2s sau hạ mẫu 50 Hz)
alignas(16) static float s_blkbp[B3_RAW_C][B3_BLOCK_DECIM_T];       // 6 x 100 x 4B = 2400 B (khối 2s sau lọc biquad liên tục)
alignas(16) static float s_biquad_z[B3_RAW_C][B3_SOS_SECTIONS][2];  // 6 x 4 x 2 x 4B = 192 B (trạng thái z1, z2 liên tục)
alignas(16) static float s_win_x8[B3_WIN_DECIM_T][B3_OUT_C];        // 200 x 8 x 4B = 6400 B (cửa sổ trượt 4s x 8 kênh)
alignas(16) static float s_win_xn[B3_WIN_DECIM_T][B3_OUT_C];        // 200 x 8 x 4B = 6400 B (cửa sổ 4s sau chuẩn hóa z-score)

static void reset_streaming_state() {
  memset(s_biquad_z, 0, sizeof(s_biquad_z));
  memset(s_win_x8, 0, sizeof(s_win_x8));
}

// Bước 1 (mỗi khối 2 s): Hạ mẫu 100 -> 50 Hz bằng FIR polyphase 41 taps Kaiser beta=5.0 trên bản ghi liên tục
static void __attribute__((noinline)) step_downsample_block_2s(
    const float raw_rec[B3_RAW_C][B3_RAW_TOTAL_T],
    int block_idx,
    float out_blk50[B3_RAW_C][B3_BLOCK_DECIM_T]) {
  const int half_len = (B3_FIR_LEN - 1) / 2;  // 20
  const int m_base = block_idx * B3_BLOCK_DECIM_T;
  for (int c = 0; c < B3_RAW_C; ++c) {
    const float* x = raw_rec[c];
    float* y = out_blk50[c];
    for (int n = 0; n < B3_BLOCK_DECIM_T; ++n) {
      const int center = 2 * (m_base + n) + half_len;
      int k_min = 0;
      if (center - (B3_RAW_TOTAL_T - 1) > k_min) k_min = center - (B3_RAW_TOTAL_T - 1);
      int k_max = B3_FIR_LEN - 1;
      if (center < k_max) k_max = center;
      float acc = 0.0f;
      for (int k = k_min; k <= k_max; ++k) {
        acc += g_fir_kaiser41[k] * x[center - k];
      }
      y[n] = acc;
    }
  }
}

// Bước 2 (mỗi khối 2 s): Lọc thông dải 0.5-20 Hz Butterworth bậc 4 nhân quả LIÊN TỤC (giữ s_biquad_z qua các khối)
static void __attribute__((noinline)) step_bandpass_biquad_continuous(
    const float in_blk50[B3_RAW_C][B3_BLOCK_DECIM_T],
    float out_blkbp[B3_RAW_C][B3_BLOCK_DECIM_T],
    float z_state[B3_RAW_C][B3_SOS_SECTIONS][2]) {
  for (int c = 0; c < B3_RAW_C; ++c) {
    memcpy(out_blkbp[c], in_blk50[c], sizeof(float) * B3_BLOCK_DECIM_T);
    float* buf = out_blkbp[c];
    for (int s = 0; s < B3_SOS_SECTIONS; ++s) {
      const float b0 = g_butter4_bp_sos[s][0];
      const float b1 = g_butter4_bp_sos[s][1];
      const float b2 = g_butter4_bp_sos[s][2];
      const float a1 = g_butter4_bp_sos[s][4];
      const float a2 = g_butter4_bp_sos[s][5];
      float z1 = z_state[c][s][0];
      float z2 = z_state[c][s][1];
      for (int n = 0; n < B3_BLOCK_DECIM_T; ++n) {
        const float x = buf[n];
        const float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        buf[n] = y;
      }
      z_state[c][s][0] = z1;
      z_state[c][s][1] = z2;
    }
  }
}

// Bước 3 (mỗi khối 2 s): Dịch cửa sổ trượt 4 s (200 mẫu) đi hop 100 mẫu và tính 8 kênh cho 100 mẫu mới
static void __attribute__((noinline)) step_update_window_8ch(
    const float in_blkbp[B3_RAW_C][B3_BLOCK_DECIM_T],
    float win_x8[B3_WIN_DECIM_T][B3_OUT_C]) {
  memmove(&win_x8[0][0],
          &win_x8[B3_HOP_DECIM_T][0],
          sizeof(float) * (B3_WIN_DECIM_T - B3_HOP_DECIM_T) * B3_OUT_C);
  float (*dst)[B3_OUT_C] = &win_x8[B3_WIN_DECIM_T - B3_HOP_DECIM_T];
  for (int n = 0; n < B3_BLOCK_DECIM_T; ++n) {
    const float ax = in_blkbp[0][n];
    const float ay = in_blkbp[1][n];
    const float az = in_blkbp[2][n];
    const float wx = in_blkbp[3][n];
    const float wy = in_blkbp[4][n];
    const float wz = in_blkbp[5][n];
    dst[n][0] = ax;
    dst[n][1] = ay;
    dst[n][2] = az;
    dst[n][3] = wx;
    dst[n][4] = wy;
    dst[n][5] = wz;
    dst[n][6] = sqrtf(ax * ax + ay * ay + az * az);
    dst[n][7] = sqrtf(wx * wx + wy * wy + wz * wz);
  }
}

// Bước 4 (mỗi cửa sổ 4 s): Chuẩn hóa z-score + lượng tử hóa INT8
static void __attribute__((noinline)) step_norm_quant_window(
    const float in_x8[B3_WIN_DECIM_T][B3_OUT_C],
    float tmp_xn[B3_WIN_DECIM_T][B3_OUT_C],
    int8_t* out_q,
    float inv_scale,
    int zp) {
  for (int n = 0; n < B3_WIN_DECIM_T; ++n) {
    for (int c = 0; c < B3_OUT_C; ++c) {
      const float xn = (in_x8[n][c] - g_norm_mu[c]) / g_norm_sd[c];
      tmp_xn[n][c] = xn;
      long q = lrintf(xn * inv_scale) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      out_q[n * B3_OUT_C + c] = (int8_t)q;
    }
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
  printf("\n#B3 BEGIN\n");
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
  const size_t dsp_buf_bytes = sizeof(s_blk50) + sizeof(s_blkbp) + sizeof(s_biquad_z) + sizeof(s_win_x8) + sizeof(s_win_xn);
  const size_t int_free_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  printf("#memory model_bytes=%u arena_alloc=%u arena_used=%u arena_mem=%s dsp_static_buf_bytes=%u biquad_state_bytes=%u total_e2e_ram_bytes=%u internal_free_after=%u in_bytes=%u out_bytes=%u\n",
         g_model_len, (unsigned)ARENA_BYTES, (unsigned)interp.arena_used_bytes(), arena_mem,
         (unsigned)dsp_buf_bytes, (unsigned)sizeof(s_biquad_z),
         (unsigned)(interp.arena_used_bytes() + dsp_buf_bytes),
         (unsigned)int_free_after, (unsigned)in->bytes, (unsigned)out->bytes);

  const float in_scale = in->params.scale;
  const int in_zp = (int)in->params.zero_point;
  const float inv_in_scale = 1.0f / in_scale;
  const float out_scale = out->params.scale;
  const int out_zp = (int)out->params.zero_point;
  printf("#quant in_scale=%.8g in_zp=%d out_scale=%.8g out_zp=%d\n",
         in_scale, in_zp, out_scale, out_zp);

  // Kiểm tra bit-exact mô hình độc lập trên 10 mẫu bench_inputs của p56x4
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
  // PHẦN 1: Phát lại TOÀN BỘ bản ghi PADS thật 100 Hz theo khối 2 s, cửa sổ 4 s hop 2 s
  //         Giữ trạng thái biquad liên tục qua các khối; đối chiếu với Python
  // =========================================================================
  printf("#B3 CONTINUOUS STREAMING REPLAY BEGIN\n");
  int total_win_all = 0;
  int total_exact_all = 0;
  float max_abs_dp_all = 0.0f;
  int max_abs_dq_all = 0;

  for (int r = 0; r < B3_N_REC; ++r) {
    reset_streaming_state();
    int rec_exact = 0;
    float rec_max_dp = 0.0f;
    int rec_max_dq = 0;
    double sum_prob_board = 0.0;
    double sum_prob_py = 0.0;

    printf("#record_begin rec_idx=%d file=%s label=%s raw_samples=%d n_blocks=%d n_win=%d desc=\"%s\"\n",
           r, g_b3_rec_file[r], g_b3_rec_label[r], B3_RAW_TOTAL_T, B3_N_BLOCKS, B3_N_WIN, g_b3_rec_desc[r]);

    for (int b = 0; b < B3_N_BLOCKS; ++b) {
      step_downsample_block_2s(g_b3_raw_records[r], b, s_blk50);
      step_bandpass_biquad_continuous(s_blk50, s_blkbp, s_biquad_z);
      step_update_window_8ch(s_blkbp, s_win_x8);

      if (b >= 1) {
        const int w = b - 1;
        step_norm_quant_window(s_win_x8, s_win_xn, in->data.int8, inv_in_scale, in_zp);
        interp.Invoke();
        const int8_t q_board = out->data.int8[0];
        const float logit_board = ((int)q_board - out_zp) * out_scale;
        const float prob_board = 1.0f / (1.0f + expf(-logit_board));
        int32_t cks = 0;
        for (int i = 0; i < IN_SIZE; ++i) cks += in->data.int8[i];

        const int8_t q_py = g_ref_q_py_cont[r][w];
        const float prob_py = g_ref_prob_py_cont[r][w];
        const int dq = abs((int)q_board - (int)q_py);
        const float dp = fabsf(prob_board - prob_py);
        const bool is_exact = (q_board == q_py);
        if (is_exact) {
          rec_exact++;
          total_exact_all++;
        }
        total_win_all++;
        if (dp > rec_max_dp) rec_max_dp = dp;
        if (dp > max_abs_dp_all) max_abs_dp_all = dp;
        if (dq > rec_max_dq) rec_max_dq = dq;
        if (dq > max_abs_dq_all) max_abs_dq_all = dq;

        sum_prob_board += prob_board;
        sum_prob_py += prob_py;

        const float prob_zp = g_ref_prob_py_zp[r][w];
        const float prob_rst = g_ref_prob_py_reset[r][w];
        printf("#win_stream rec=%d win=%d block=%d t_start_s=%.1f t_end_s=%.1f "
               "qin_cksum=%ld ref_cksum=%ld q_board=%d prob_board=%.6f "
               "q_py_cont=%d prob_py_cont=%.6f bit_exact=%d abs_dp=%.6f abs_dq=%d "
               "q_py_reset=%d prob_py_reset=%.6f diff_vs_reset=%+.6f "
               "q_py_zp=%d prob_py_zp=%.6f diff_vs_zp=%+.6f\n",
               r, w, b, w * 2.0f, w * 2.0f + 4.0f,
               (long)cks, (long)g_ref_qin_cksum[r][w],
               (int)q_board, prob_board,
               (int)q_py, prob_py, (int)is_exact, dp, dq,
               (int)g_ref_q_py_reset[r][w], prob_rst, prob_board - prob_rst,
               (int)g_ref_q_py_zp[r][w], prob_zp, prob_board - prob_zp);
      }
    }
    printf("#record_summary rec_idx=%d file=%s label=%s bit_exact=%d/%d max_abs_dp=%.6f max_abs_dq=%d mean_prob_board=%.6f mean_prob_py=%.6f\n",
           r, g_b3_rec_file[r], g_b3_rec_label[r], rec_exact, B3_N_WIN, rec_max_dp, rec_max_dq,
           sum_prob_board / B3_N_WIN, sum_prob_py / B3_N_WIN);
  }
  printf("#b3_stream_overall primary_bit_exact=%d/%d all_records_bit_exact=%d/%d max_abs_dp=%.6f max_abs_dq=%d\n",
         B3_N_WIN, B3_N_WIN, total_exact_all, total_win_all, max_abs_dp_all, max_abs_dq_all);
  printf("#B3 CONTINUOUS STREAMING REPLAY END\n");

  // =========================================================================
  // PHẦN 2: Đo độ trễ mỗi khối 2 s (tiền xử lý) + mỗi lần suy luận cửa sổ 4 s
  //         trên 100 lần phát lại toàn bộ bản ghi chính (1000 khối 2s, 900 lần suy luận)
  // =========================================================================
  printf("#B3 LATENCY BENCHMARK BEGIN replays=%d blocks_total=%d windows_total=%d file=%s\n",
         N_TIMED_REPLAYS, N_TIMED_REPLAYS * B3_N_BLOCKS, N_TIMED_REPLAYS * B3_N_WIN, g_b3_rec_file[0]);

  volatile float sink_prob = 0.0f;
  for (int rep = 0; rep < WARMUP_REPLAYS; ++rep) {
    reset_streaming_state();
    for (int b = 0; b < B3_N_BLOCKS; ++b) {
      step_downsample_block_2s(g_b3_raw_records[0], b, s_blk50);
      step_bandpass_biquad_continuous(s_blk50, s_blkbp, s_biquad_z);
      step_update_window_8ch(s_blkbp, s_win_x8);
      if (b >= 1) {
        step_norm_quant_window(s_win_x8, s_win_xn, in->data.int8, inv_in_scale, in_zp);
        interp.Invoke();
        const int8_t q_out = out->data.int8[0];
        const float logit = ((int)q_out - out_zp) * out_scale;
        sink_prob = 1.0f / (1.0f + expf(-logit));
      }
    }
  }

  std::vector<int64_t> t_blk_down, t_blk_bp, t_blk_ch8, t_blk_prep;
  std::vector<int64_t> t_win_normq, t_win_invoke, t_win_infer_total, t_hop_active_total;
  t_blk_down.reserve(N_TIMED_REPLAYS * B3_N_BLOCKS);
  t_blk_bp.reserve(N_TIMED_REPLAYS * B3_N_BLOCKS);
  t_blk_ch8.reserve(N_TIMED_REPLAYS * B3_N_BLOCKS);
  t_blk_prep.reserve(N_TIMED_REPLAYS * B3_N_BLOCKS);
  t_win_normq.reserve(N_TIMED_REPLAYS * B3_N_WIN);
  t_win_invoke.reserve(N_TIMED_REPLAYS * B3_N_WIN);
  t_win_infer_total.reserve(N_TIMED_REPLAYS * B3_N_WIN);
  t_hop_active_total.reserve(N_TIMED_REPLAYS * B3_N_WIN);

  for (int rep = 0; rep < N_TIMED_REPLAYS; ++rep) {
    reset_streaming_state();
    for (int b = 0; b < B3_N_BLOCKS; ++b) {
      const int64_t t0 = esp_timer_get_time();
      step_downsample_block_2s(g_b3_raw_records[0], b, s_blk50);
      const int64_t t1 = esp_timer_get_time();
      step_bandpass_biquad_continuous(s_blk50, s_blkbp, s_biquad_z);
      const int64_t t2 = esp_timer_get_time();
      step_update_window_8ch(s_blkbp, s_win_x8);
      const int64_t t3 = esp_timer_get_time();

      t_blk_down.push_back(t1 - t0);
      t_blk_bp.push_back(t2 - t1);
      t_blk_ch8.push_back(t3 - t2);
      t_blk_prep.push_back(t3 - t0);

      if (b >= 1) {
        const int64_t t4 = esp_timer_get_time();
        step_norm_quant_window(s_win_x8, s_win_xn, in->data.int8, inv_in_scale, in_zp);
        const int64_t t5 = esp_timer_get_time();
        interp.Invoke();
        const int8_t q_out = out->data.int8[0];
        const float logit = ((int)q_out - out_zp) * out_scale;
        sink_prob = 1.0f / (1.0f + expf(-logit));
        const int64_t t6 = esp_timer_get_time();

        t_win_normq.push_back(t5 - t4);
        t_win_invoke.push_back(t6 - t5);
        t_win_infer_total.push_back(t6 - t4);
        t_hop_active_total.push_back((t3 - t0) + (t6 - t4));
      }
    }
    if ((rep & 7) == 0) vTaskDelay(1);
  }
  (void)sink_prob;

  print_stat_line("blk_1_downsample_2s_100to50", t_blk_down);
  print_stat_line("blk_2_biquad_continuous_2s", t_blk_bp);
  print_stat_line("blk_3_make8ch_shift_win_2s", t_blk_ch8);
  print_stat_line("blk_preprocess_2s_total", t_blk_prep);
  print_stat_line("win_4_norm_quant_4s", t_win_normq);
  print_stat_line("win_5_model_invoke_espnn", t_win_invoke);
  print_stat_line("win_infer_4s_total", t_win_infer_total);
  print_stat_line("hop_2s_active_total", t_hop_active_total);
  printf("#B3 LATENCY BENCHMARK END\n");
  printf("#B3 END\n");
  fflush(stdout);
}

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1500));
  run();
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
