# Nhiệm vụ cho Antigravity — B1: build + nạp + đo mô hình trên ESP32-S3

Điều kiện: B0 đã xong (ESP-IDF v5.3.x ở ~/esp/esp-idf, bo nhận cổng /dev/ttyACM* hoặc /dev/ttyUSB*).
Thư mục: `~/Desktop/PARKINSON`. Chỉ ghi trong `firmware/` và `reports/board/`.

## B1-dry (làm ngay với mô hình giả, để sẵn đường ống)
Mô hình giả `firmware/b1_bench/dummy/dummy_dstcn_int8.tflite` (DS-TCN INT8, vào 200×8, ra 2, 38.5 kB) — đã được Claude chuyển sẵn sang C trong `firmware/b1_bench/main/generated/`.
```bash
cd ~/Desktop/PARKINSON/firmware/b1_bench
. ~/esp/esp-idf/export.sh
idf.py set-target esp32s3          # lần đầu; idf_component.yml sẽ kéo espressif/esp-tflite-micro (+ esp-nn)
idf.py build 2>&1 | tail -40
idf.py size > ../../reports/board/B1_dummy_size.txt
PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1)
idf.py -p $PORT flash
cd ~/Desktop/PARKINSON && . .venv/bin/activate && pip -q install pyserial numpy
python firmware/tools/check_serial.py --port $PORT --ref firmware/b1_bench/main/generated/ref_outputs.npy --out reports/board/B1_dummy
```
Nếu `check_serial.py` không nhận log (USB-Serial-JTAG không reset qua DTR/RTS): chạy `idf.py -p $PORT monitor` khoảng 60 s, lưu log ra `reports/board/B1_dummy_monitor.log`, rồi `python firmware/tools/check_serial.py --log reports/board/B1_dummy_monitor.log --ref ... --out reports/board/B1_dummy`.

Sửa lỗi build được phép (ghi lại mọi thay đổi vào `reports/board/B1_STATUS.txt`):
- `#ERR AllocateTensors` → tăng `ARENA_BYTES` trong main.cc.
- Lỗi tên Kconfig `CONFIG_NN_OPTIMIZED` → kiểm tra `idf.py menuconfig` mục ESP-NN, chọn bản tối ưu cho esp32s3.
- Lỗi phiên bản esp-tflite-micro → đổi ràng buộc trong `main/idf_component.yml`.
Không sửa `generated/` bằng tay.

Chạy 2 cấu hình và lưu cả hai:
1. ESP-NN tối ưu (mặc định) → `reports/board/B1_dummy_nn.*`
2. ESP-NN ANSI C (`CONFIG_NN_ANSI_C=y`, tắt optimized) → `reports/board/B1_dummy_ansi.*`

Ghi `reports/board/B1_STATUS.txt`: phiên bản ESP-IDF/esp-tflite-micro/esp-nn, flash app (từ idf.py size), arena_used, latency mean/p95 cho 2 cấu hình, số mẫu bit-exact /50, lỗi + cách sửa.

## B1-real (khi Claude Code xong P5)
Với mỗi mô hình `reports/p5/models/<tên>.tflite` và `reports/p5/bench_inputs.npy`:
```bash
python firmware/tools/tflite_to_c.py --model reports/p5/models/<tên>.tflite --inputs reports/p5/bench_inputs.npy --n 50 --out firmware/b1_bench/main/generated
cd firmware/b1_bench && idf.py build && idf.py -p $PORT flash && cd ../..
python firmware/tools/check_serial.py --port $PORT --ref firmware/b1_bench/main/generated/ref_outputs.npy --out reports/board/B1_<tên>_nn
```
(Cần TensorFlow trong .venv cho `tflite_to_c.py`: `pip install tensorflow-cpu`.)
