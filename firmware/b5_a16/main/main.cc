// b5_a16 — đo mô hình 16x8 (int8 weights, int16 activations) trên ESP32-S3 với dữ liệu PADS phát lại.
// In CSV qua serial; tools/check_serial.py phân tích và so bit-exact với TFLite trên PC.
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_chip_info.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"  // kéo theo schema (vị trí header đổi giữa các bản TFLM)
#include "generated/model_data.h"
#include "generated/test_data.h"
#include "generated/op_resolver.h"

#ifndef ARENA_BYTES
#define ARENA_BYTES (256 * 1024)
#endif
#define WARMUP 3
#define N_TIMED 100

static void run() {
  printf("\n#B5 BEGIN\n");
  esp_chip_info_t ci; esp_chip_info(&ci);
  printf("#chip cores=%d rev=%d cpu_mhz=%d psram_bytes=%u internal_free=%u\n", ci.cores, ci.revision,
         CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#if CONFIG_NN_OPTIMIZED
  printf("#esp_nn=optimized\n");
#else
  printf("#esp_nn=ansi\n");
#endif

  const char* arena_mem = "internal";
  uint8_t* arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) { arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_SPIRAM); arena_mem = "psram"; }
  if (!arena) { printf("#ERR arena alloc\n"); return; }

  const tflite::Model* model = tflite::GetModel(g_model);
  static OpResolver resolver; register_ops(resolver);
  static tflite::MicroInterpreter interp(model, resolver, arena, ARENA_BYTES);
  if (interp.AllocateTensors() != kTfLiteOk) { printf("#ERR AllocateTensors (tăng ARENA_BYTES)\n"); return; }
  TfLiteTensor* in = interp.input(0); TfLiteTensor* out = interp.output(0);
  printf("#model_bytes=%u arena_bytes=%u arena_used=%u arena_mem=%s in_bytes=%u out_bytes=%u\n",
         g_model_len, ARENA_BYTES, (unsigned)interp.arena_used_bytes(), arena_mem,
         (unsigned)in->bytes, (unsigned)out->bytes);
  if (in->bytes != IN_BYTES || out->bytes != OUT_BYTES) { printf("#ERR size mismatch\n"); return; }
  printf("#quant in_scale=%.8g in_zp=%d out_scale=%.8g out_zp=%d\n", in->params.scale, (int)in->params.zero_point,
         out->params.scale, (int)out->params.zero_point);
  printf("#in_type=%d out_type=%d\n", (int)in->type, (int)out->type);

  // 1) bit-exact: chạy toàn bộ N_TEST mẫu, in đầu ra
  int mism = 0;
  printf("idx,match,out\n");
  for (int i = 0; i < N_TEST; ++i) {
    memcpy(in->data.raw, g_test_in[i], IN_BYTES);
    if (interp.Invoke() != kTfLiteOk) { printf("#ERR invoke %d\n", i); return; }
    bool ok = memcmp(out->data.raw, g_ref_out[i], OUT_BYTES) == 0; mism += !ok;
    vTaskDelay(1);
    printf("%d,%d,", i, ok);
    for (int k = 0; k < OUT_BYTES; ++k) printf("%02x%s", (unsigned)(uint8_t)out->data.raw[k], k + 1 < OUT_BYTES ? "" : "\n");
  }
  printf("#bitexact matched=%d/%d\n", N_TEST - mism, N_TEST);

  // 2) latency
  memcpy(in->data.raw, g_test_in[0], IN_BYTES);
  for (int i = 0; i < WARMUP; ++i) interp.Invoke();
  std::vector<int64_t> t(N_TIMED);
  for (int i = 0; i < N_TIMED; ++i) {
    memcpy(in->data.raw, g_test_in[i % N_TEST], IN_BYTES);
    int64_t t0 = esp_timer_get_time(); interp.Invoke(); t[i] = esp_timer_get_time() - t0;
    vTaskDelay(1);  // int16 kernels are slow: yield every run (outside the timed region)
  }
  std::sort(t.begin(), t.end());
  double s = 0; for (auto v : t) s += v;
  printf("#latency_us mean=%.1f p50=%lld p95=%lld min=%lld max=%lld n=%d\n", s / N_TIMED,
         t[N_TIMED / 2], t[(int)(N_TIMED * 0.95)], t.front(), t.back(), N_TIMED);
  printf("#B5 END\n");
}

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1500));
  run();
  while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
