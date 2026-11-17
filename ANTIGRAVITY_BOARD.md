# Nhiệm vụ cho Antigravity — Bo ESP32-S3 (Sparkle IoT XH-S3E, N16R8)

Bo: ESP32-S3, flash 16 MB, PSRAM 8 MB (octal), USB-C cắm vào máy này. Mục đích: đo thời gian suy luận và bộ nhớ của mô hình INT8 trên bo, chạy trên **dữ liệu PADS phát lại** (không đo người).
Thư mục: `~/Desktop/PARKINSON/firmware/`. Mọi kết quả ghi vào `~/Desktop/PARKINSON/reports/board/`.
Không sửa file ngoài `firmware/` và `reports/board/`.

## B0 — Dựng toolchain và kiểm tra bo (làm ngay, không cần mô hình)
1. Cài ESP-IDF **v5.3** (hoặc v5.x ổn định mới nhất) theo hướng dẫn chính thức cho Linux, target `esp32s3`:
   ```bash
   sudo apt-get install -y git wget flex bison gperf python3 python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
   mkdir -p ~/esp && cd ~/esp && git clone -b v5.3.2 --recursive https://github.com/espressif/esp-idf.git
   cd esp-idf && ./install.sh esp32s3 && . ./export.sh
   ```
2. Tìm cổng: `ls /dev/ttyACM* /dev/ttyUSB*`. Nếu bị từ chối quyền: `sudo usermod -aG dialout $USER` (cần đăng xuất/đăng nhập lại — báo anh Tuyến).
3. `esptool.py --chip esp32s3 -p <PORT> flash_id` → ghi kích thước flash, loại chip, MAC (bỏ MAC) vào `reports/board/B0_chip.txt`.
4. Tạo project kiểm tra `firmware/b0_smoke/` từ example `hello_world` của ESP-IDF, bật PSRAM octal trong menuconfig
   (`CONFIG_SPIRAM=y`, `CONFIG_SPIRAM_MODE_OCT=y`, flash size 16MB, CPU 240 MHz). Build, flash, monitor 10 s.
   Ghi log (có dòng báo PSRAM và tần số CPU) vào `reports/board/B0_smoke.log`.
5. Thêm component `espressif/esp-tflite-micro` (qua `idf.py add-dependency "espressif/esp-tflite-micro"`) và build example `hello_world` của nó (mô hình sine). In thời gian `Invoke()` trung bình 1000 lần bằng `esp_timer_get_time()`. Ghi vào `reports/board/B0_tflm.log`.
6. Ghi `reports/board/B0_STATUS.txt`: phiên bản ESP-IDF, phiên bản esp-tflite-micro, cổng, có ESP-NN (`CONFIG_NN_OPTIMIZED=y`) hay không, lỗi gặp phải.

## B1 — Đo mô hình của bài (làm sau phiên P5, khi có `reports/p5/models/*.tflite`)
Claude sẽ viết `firmware/b1_bench/` + hướng dẫn chi tiết. Khung dự kiến:
- `model_data.cc` sinh từ `.tflite` bằng `xxd -i`; dữ liệu đầu vào: 200 cửa sổ PADS đã lượng tử hóa sẵn (`test_inputs.h`).
- Đo: latency/Invoke (mean, p95, 1000 lần, sau 10 lần khởi động), tensor arena thực dùng (`arena_used_bytes()`), flash của app (`idf.py size`), có/không ESP-NN.
- So output INT8 từng mẫu với TFLite trên PC (`reports/p5/ref_outputs.npy`) — phải khớp bit (bit-exact). In kết quả CSV qua serial.
