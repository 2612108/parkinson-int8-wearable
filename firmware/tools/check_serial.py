"""Đọc log serial của b1_bench, so bit-exact với ref_outputs.npy, ghi tóm tắt.

Thu log:   python tools/check_serial.py --port /dev/ttyACM0 --ref <generated>/ref_outputs.npy --out reports/board/B1_<tag>
Có log sẵn: python tools/check_serial.py --log b1.log --ref ... --out ...
"""
import argparse, json, pathlib, re, time
import numpy as np


def capture(port, timeout=180):
    import serial  # pyserial
    lines, t0 = [], time.time()
    with serial.Serial(port, 115200, timeout=1) as s:
        s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False  # reset bo
        while time.time() - t0 < timeout:
            ln = s.readline().decode(errors="ignore").rstrip()
            if ln:
                lines.append(ln)
                if ln.startswith("#B1 END") or ln.startswith("#ERR"):
                    break
    return lines


def parse(lines, ref):
    kv, outs = {}, {}
    for ln in lines:
        if ln.startswith("#"):
            for k, v in re.findall(r"(\w+)=([\w./-]+)", ln):
                kv[k] = v
        m = re.match(r"^(\d+),([01]),(.*)$", ln)
        if m:
            outs[int(m.group(1))] = np.array([int(v) for v in m.group(3).split()], dtype=np.int8)
    n = len(ref)
    got = [i for i in range(n) if i in outs]
    exact = [i for i in got if np.array_equal(outs[i], ref[i])]
    maxdiff = max((int(np.abs(outs[i].astype(int) - ref[i].astype(int)).max()) for i in got), default=None)
    return dict(kv=kv, n_ref=n, n_received=len(got), n_bitexact=len(exact), max_abs_lsb_diff=maxdiff,
                errors=[ln for ln in lines if ln.startswith("#ERR")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--log"); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = capture(a.port, a.timeout) if a.port else pathlib.Path(a.log).read_text(errors="ignore").splitlines()
    pathlib.Path(str(out) + ".log").write_text("\n".join(lines) + "\n")
    res = parse(lines, np.load(a.ref))
    pathlib.Path(str(out) + ".json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
