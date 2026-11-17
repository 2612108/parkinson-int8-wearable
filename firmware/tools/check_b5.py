"""Capture b5_a16 serial output and summarise (bit-exact count reported by the device, latency, arena)."""
import argparse, json, pathlib, re, time
ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--timeout", type=int, default=240); a = ap.parse_args()
import serial
lines, t0 = [], time.time()
with serial.Serial(a.port, 115200, timeout=1) as s:
    s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
    while time.time() - t0 < a.timeout:
        ln = s.readline().decode(errors="ignore").rstrip()
        if ln:
            lines.append(ln)
            if ln.startswith("#B5 END") or ln.startswith("#ERR"): break
pathlib.Path(a.out + ".log").write_text("\n".join(lines) + "\n")
kv = {}
for ln in lines:
    if ln.startswith("#"):
        for k, v in re.findall(r"(\w+)=([\w./-]+)", ln): kv[k] = v
rows = [ln for ln in lines if re.match(r"^\d+,[01],", ln)]
res = dict(kv=kv, n_rows=len(rows), n_match_rows=sum(ln.split(",")[1] == "1" for ln in rows),
           errors=[ln for ln in lines if ln.startswith("#ERR")], completed=any(l.startswith("#B5 END") for l in lines))
pathlib.Path(a.out + ".json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
