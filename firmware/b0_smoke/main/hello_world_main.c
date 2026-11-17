#include <stdio.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "esp_private/esp_clk.h"

void app_main(void)
{
    printf("=== B0 SMOKE TEST: ESP32-S3 (N16R8) ===\n");

    esp_chip_info_t chip_info;
    uint32_t flash_size = 0;
    esp_chip_info(&chip_info);

    printf("Chip: %s with %d CPU core(s), revision v%d.%d\n",
           CONFIG_IDF_TARGET, chip_info.cores,
           chip_info.revision / 100, chip_info.revision % 100);

    printf("CPU Frequency: %d MHz\n", esp_clk_cpu_freq() / 1000000);

    if (esp_flash_get_size(NULL, &flash_size) == ESP_OK) {
        printf("Flash size: %" PRIu32 " MB (%s)\n",
               flash_size / (1024 * 1024),
               (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    }

    size_t psram_size = esp_psram_get_size();
    printf("PSRAM size: %zu bytes (%.2f MB)\n",
           psram_size, (double)psram_size / (1024.0 * 1024.0));
    printf("Free heap: %" PRIu32 " bytes, Free PSRAM heap: %zu bytes\n",
           esp_get_free_heap_size(),
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    for (int i = 0; i < 10; i++) {
        printf("[B0_SMOKE] tick %d/10 - CPU %d MHz, PSRAM %zu KB\n",
               i + 1, esp_clk_cpu_freq() / 1000000, psram_size / 1024);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    printf("=== B0 SMOKE TEST DONE ===\n");
    fflush(stdout);
}
