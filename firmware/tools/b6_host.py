"""B6 host: replay all fold-0 test windows of a run on the ESP32-S3 and compare with the desktop interpreter.
python firmware/tools/b6_host.py --run reports/p56z0 --variant qat --port /dev/ttyACM0 --out reports/board/B6_qat
"""
import argparse, json, pathlib, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "firmware" / ".pydeps"))
import importlib.util
spec = importlib.util.spec_from_file_location("p56", ROOT / "scripts" / "p56x_quant_extra_r2.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
try:
    from ai_edge_litert import interpreter as tfl; Interp = tfl.Interpreter; REF = tfl.OpResolverType.BUILTIN_REF
except Exception:
    import tensorflow as tf; Interp = tf.lite.Interpreter; REF = tf.lite.experimental.OpResolverType.BUILTIN_REF

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--variant", default="qat"); ap.add_argument("--port", default="/dev/ttyACM0")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--causal", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
run = ROOT / a.run
S.CAUSAL = a.causal   # B6k: causal (device) preprocessing
mb = (run / "models" / f"PD_vs_HC_raw_{a.variant}_fold0.tflite").read_bytes()
nz = np.load(run / "models" / "norm_PD_vs_HC_raw_fold0.npz"); mu, sd = nz["mu"], nz["sd"]
base = S.find_pads(ROOT / "data/raw/pads"); P, recs = S.load_pads(base)
y_all = (P.label == "PD").astype(int).to_numpy(); sel = np.where(P.label.isin(["PD", "HC"]))[0]
tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(sel, y_all[sel])))
te_p = sel[te]
it = Interp(model_content=mb, experimental_op_resolver_type=REF); it.allocate_tensors()
ind, outd = it.get_input_details()[0], it.get_output_details()[0]; si, zi = ind["quantization"]; so, zo = outd["quantization"]
import serial
ser = serial.Serial(a.port, 115200, timeout=5)
ser.dtr = False; ser.rts = True; time.sleep(0.1); ser.rts = False
t0 = time.time(); ready = None
while time.time() - t0 < 30:
    ln = ser.readline().decode(errors="ignore").strip()
    if ln.startswith("#B6 READY"): ready = ln; break
if not ready: raise SystemExit("no READY from board")
time.sleep(0.8); ser.reset_input_buffer()
magic = bytes([0xA5, 0x5A, 0xA5, 0x5A])
rows = []; nwin = 0; exact = 0; maxd = 0; lat = []
for i in te_p:
    W = S.windows_for(recs[i]); X = (S.to_input(W, "raw") - mu) / sd
    q = np.clip(np.round(X / si + zi), -128, 127).astype(np.int8).reshape((len(X),) + tuple(ind["shape"][1:]))
    dev, ref = [], []
    for x in q:
        it.set_tensor(ind["index"], x[None]); it.invoke(); r = int(it.get_tensor(outd["index"]).ravel()[0])
        ser.write(magic + x.tobytes())
        ln = ser.readline().decode(errors="ignore").strip()
        while ln and not ln.startswith("R "):
            ln = ser.readline().decode(errors="ignore").strip()
        if not ln: raise SystemExit(f"timeout at participant {P.sid[i]}")
        _, v, us = ln.split(); v = int(v); lat.append(int(us))
        dev.append(v); ref.append(r); nwin += 1; exact += (v == r); maxd = max(maxd, abs(v - r))
    sig = lambda o: 1 / (1 + np.exp(-(np.array(o, float) - zo) * so))
    rows.append(dict(sid=P.sid[i], y=int(y_all[i]), n=len(q), p_dev=float(sig(dev).mean()), p_ref=float(sig(ref).mean())))
    if a.limit and len(rows) >= a.limit: break
yy = np.array([r["y"] for r in rows]); pdv = np.array([r["p_dev"] for r in rows]); prf = np.array([r["p_ref"] for r in rows])
res = dict(run=a.run, variant=a.variant, causal=a.causal, ready=ready, n_participants=len(rows), n_windows=nwin, n_bitexact=int(exact), max_abs_lsb_diff=int(maxd),
           auroc_device=float(roc_auc_score(yy, pdv)), auroc_desktop=float(roc_auc_score(yy, prf)),
           latency_us_mean=float(np.mean(lat)), latency_us_p95=float(np.percentile(lat, 95)), wall_s=round(time.time() - t0, 1))
pathlib.Path(a.out + ".json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
import pandas as pd; pd.DataFrame(rows).to_csv(a.out + "_participants.csv", index=False)
