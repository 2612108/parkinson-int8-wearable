# Hướng dẫn đo năng lượng trên bo ESP32-S3 (B2)

## 1. Kết nối phần cứng
1. Cắm **đồng hồ đo công suất/dòng USB (USB tester)** nối tiếp giữa cổng USB của máy tính và cáp USB-C cấp nguồn cho bo **ESP32-S3** (`/dev/ttyACM0`).
2. Chuyển màn hình đồng hồ USB sang chế độ hiển thị rõ **Điện áp $V$ (V)**, **Dòng điện $I$ (mA)** (và Công suất $P$ (mW/W) nếu có).

## 2. Lệnh chạy lại chế độ đo năng lượng (3 pha × 30 giây)
Firmware `b2_e2e` đã được nạp sẵn trên bo. Để khởi động lại bo và theo dõi trực tiếp thời điểm bắt đầu/kết thúc từng pha trên màn hình terminal (mà không cần build/nạp lại), chạy lệnh:

```bash
cd ~/Desktop/PARKINSON
bash firmware/tools/b2_run.sh --monitor
```
*(Hoặc nhấn nút **RST** trên bo ESP32-S3; sau khoảng ~13 giây đo độ trễ, bo sẽ tự động bước vào 3 pha năng lượng).*

## 3. Ghi số liệu / chụp ảnh đồng hồ USB ở 3 pha
Quan sát terminal, khi thấy dòng báo `PHASE ... START`, đợi 3–5 giây cho dòng điện ổn định rồi **chụp ảnh màn hình đồng hồ USB** (hoặc ghi lại $V$ và $I$ trung bình) cho từng pha (mỗi pha kéo dài **30 giây**):

| Pha | Dấu hiệu trên Terminal | Trạng thái hoạt động của bo | Số liệu cần ghi / chụp |
| :--- | :--- | :--- | :--- |
| **Pha 1: `IDLE`** | `PHASE IDLE START` $\rightarrow$ `PHASE IDLE END` | CPU 240 MHz ở trạng thái nghỉ nền (`vTaskDelay`), không suy luận | $V_{\text{idle}}$ (V), $I_{\text{idle}}$ (mA) |
| **Pha 2: `INFER`** | `PHASE INFER START` $\rightarrow$ `PHASE INFER END` | Chạy đường ống đầu-cuối (DSP + chuẩn hóa + suy luận ESP-NN) liên tục 100% thời gian | $V_{\text{infer}}$ (V), $I_{\text{infer}}$ (mA) |
| **Pha 3: `INFER_DUTY`** | `PHASE INFER_DUTY START` $\rightarrow$ `PHASE INFER_DUTY END` | Chu kỳ thực tế: mỗi **4 s** thức dậy chạy 1 cửa sổ (`14.64 ms`) rồi vào **light-sleep** phần thời gian còn lại (`99.60%`) | $V_{\text{duty}}$ (V), $I_{\text{duty}}$ (mA) |

## 4. Công thức tính năng lượng cho 1 lần suy luận đầu-cuối
Từ số đo ở Pha 1 (`IDLE`) và Pha 2 (`INFER`) cùng thời gian xử lý 1 cửa sổ 4 giây $t_{\text{infer}} = 14.641\text{ ms} = 0.014641\text{ s}$ (mục `e2e_total` trong `reports/board/B2_SUMMARY.txt`):

1. **Công suất nền và công suất khi suy luận (mW):**
   $$P_{\text{idle}} = V_{\text{idle}} \times I_{\text{idle}} \quad (\text{mW})$$
   $$P_{\text{infer}} = V_{\text{infer}} \times I_{\text{infer}} \quad (\text{mW})$$
   *(với $V$ tính bằng $\text{V}$, $I$ tính bằng $\text{mA}$)*.

2. **Năng lượng tiêu thụ tăng thêm của 1 lần suy luận đầu-cuối ($\text{mJ}$):**
   $$E = (P_{\text{infer}} - P_{\text{idle}}) \times t_{\text{infer}}$$
   *(Nếu tính năng lượng tổng gồm cả nền trong suốt $t_{\text{infer}}$: $E_{\text{total}} = P_{\text{infer}} \times t_{\text{infer}}$).*

3. **Công suất trung bình ở chế độ đeo thực tế (Pha 3 — chu kỳ 1 cửa sổ / 4 giây):**
   $$P_{\text{duty}} = V_{\text{duty}} \times I_{\text{duty}} \quad (\text{mW})$$
