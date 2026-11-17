# Nhiệm vụ cho Antigravity — Phiên P1 (Kaggle)

Thư mục làm việc: `~/Desktop/PARKINSON`. Không sửa nội dung notebook; chỉ chạy, theo dõi và kéo kết quả về.
Không in, không ghi token Kaggle vào bất kỳ file nào trong thư mục này.

## 1. Cài và kiểm tra Kaggle CLI
```bash
cd ~/Desktop/PARKINSON
python3 -m venv .venv && source .venv/bin/activate
pip install -U kaggle
ls -l ~/.kaggle/access_token          # phải có, quyền -rw-------
kaggle config view                     # lấy username
```
Nếu CLI báo không đọc được `access_token`: thử `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)` rồi chạy lại. Báo lỗi nguyên văn nếu vẫn hỏng.

## 2. Điền username và đẩy notebook
```bash
U=$(kaggle config view | awk -F': ' '/username/{print $2}')
sed -i "s/KAGGLE_USERNAME/$U/" kaggle/p1/kernel-metadata.json
kaggle kernels push -p kaggle/p1
```
Nếu báo cần xác minh số điện thoại để bật Internet: dừng lại, báo anh Tuyến.

## 3. Chờ chạy xong
```bash
watch -n 60 kaggle kernels status $U/parkinson-p1-audit     # chờ "complete" (khoảng 10–20 phút)
```
Nếu "error": chạy bước 4 để lấy log, rồi dừng.

## 4. Chỉ kéo báo cáo về (KHÔNG kéo data/raw ~1 GB)
```bash
mkdir -p reports/p1
kaggle kernels output $U/parkinson-p1-audit -p reports/p1 --file-pattern '^(reports/.*|.*\.log)$' \
  || echo "CLI không có --file-pattern"
```
Nếu CLI không hỗ trợ `--file-pattern`: KHÔNG tải toàn bộ. Ghi lại `kaggle --version` vào `reports/p1/LOI.txt` và dừng.

## 5. Bàn giao
Sau khi xong, trong `reports/p1/` phải có: `audit_pads.md`, `audit_monipar.md`, `env.md`, file `.log` của kernel.
Ghi thêm `reports/p1/STATUS.txt` gồm: username, trạng thái kernel, thời gian chạy, các lỗi gặp phải.
