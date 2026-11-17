"""Ghi cấu hình máy để Claude chọn batch size, framework, thời gian chạy."""
import importlib, os, platform, shutil, subprocess, pathlib

lines = ["# Môi trường", "", f"- OS: {platform.platform()}", f"- Python: {platform.python_version()}",
         f"- CPU cores: {os.cpu_count()}"]
try:
    mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
    lines.append(f"- RAM: {int(mem)/1e6:.1f} GB")
except Exception:
    pass
du = shutil.disk_usage(pathlib.Path(__file__).resolve().parents[1])
lines.append(f"- Ổ đĩa trống: {du.free/1e9:.0f} GB")
if shutil.which("nvidia-smi"):
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    lines.append(f"- GPU: {q}")
else:
    lines.append("- GPU: không thấy nvidia-smi")
for m in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "tensorflow", "torch"]:
    try:
        v = importlib.import_module(m).__version__
    except Exception:
        v = "chưa cài"
    lines.append(f"- {m}: {v}")
out = pathlib.Path(__file__).resolve().parents[1] / "reports" / "env.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
