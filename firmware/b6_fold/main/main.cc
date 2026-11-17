// b6_fold — replay a whole test fold over USB: host sends [A5 5A A5 5A][IN_BYTES int8], device replies "R <int8> <us>\n".
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "generated/model_data.h"
#include "generated/op_resolver.h"

#define ARENA_BYTES (160 * 1024)

static void say(const char* fmt, ...) {
  char b[96]; va_list ap; va_start(ap, fmt); int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
  usb_serial_jtag_write_bytes(b, n, pdMS_TO_TICKS(1000));
}

static int read_exact(uint8_t* buf, int n) {
  int got = 0;
  while (got < n) {
    int r = usb_serial_jtag_read_bytes(buf + got, n - got, pdMS_TO_TICKS(20000));
    if (r <= 0) return got;
    got += r;
  }
  return got;
}

extern "C" void app_main(void) {
  usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
  cfg.rx_buffer_size = 8192; cfg.tx_buffer_size = 1024;
  usb_serial_jtag_driver_install(&cfg);
  vTaskDelay(pdMS_TO_TICKS(500));
  uint8_t* arena = (uint8_t*)heap_caps_aligned_alloc(16, ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  const tflite::Model* model = tflite::GetModel(g_model);
  static OpResolver resolver; register_ops(resolver);
  static tflite::MicroInterpreter interp(model, resolver, arena, ARENA_BYTES);
  if (interp.AllocateTensors() != kTfLiteOk) { say("#ERR alloc\n"); return; }
  TfLiteTensor* in = interp.input(0); TfLiteTensor* out = interp.output(0);
  const int IN = in->bytes;
  for (int k = 0; k < 3; ++k) { say("#B6 READY in_bytes=%d arena_used=%u\n", IN, (unsigned)interp.arena_used_bytes()); vTaskDelay(pdMS_TO_TICKS(200)); }
  uint8_t m[4]; long n = 0;
  while (true) {
    // sync on magic
    if (read_exact(m, 1) != 1) { vTaskDelay(1); continue; }
    if (m[0] != 0xA5) continue;
    if (read_exact(m + 1, 3) != 3 || m[1] != 0x5A || m[2] != 0xA5 || m[3] != 0x5A) continue;
    if (read_exact((uint8_t*)in->data.int8, IN) != IN) { say("#ERR short\n"); continue; }
    int64_t t0 = esp_timer_get_time();
    if (interp.Invoke() != kTfLiteOk) { say("#ERR invoke\n"); continue; }
    int64_t dt = esp_timer_get_time() - t0;
    say("R %d %lld\n", (int)out->data.int8[0], (long long)dt);
    if ((++n & 31) == 0) vTaskDelay(1);
  }
}
