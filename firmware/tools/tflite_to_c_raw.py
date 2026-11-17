"""Like tflite_to_c.py but for any input/output dtype (e.g. 16x8 model with float32 I/O).
Test inputs/outputs are stored as raw bytes; firmware compares output bytes with memcmp.
python tools/tflite_to_c_raw.py --model m.tflite --inputs x.npy --n 50 --out firmware/b5_a16/gen
"""
import argparse, json, pathlib, sys
import numpy as np
_pydeps = pathlib.Path(__file__).resolve().parents[1] / ".pydeps"
if _pydeps.is_dir(): sys.path.insert(0, str(_pydeps))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tflite_to_c import OPMAP, model_ops, c_array, _Interpreter, _BUILTIN_REF

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--inputs", required=True)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
mbytes = pathlib.Path(a.model).read_bytes()
it = _Interpreter(model_content=mbytes, experimental_op_resolver_type=_BUILTIN_REF); it.allocate_tensors()
ind, outd = it.get_input_details()[0], it.get_output_details()[0]
X = np.load(a.inputs)[: a.n].astype(np.float32)
if ind["dtype"] != np.float32:
    s, zp = ind["quantization"]; info_ = np.iinfo(ind["dtype"])
    X = np.clip(np.round(X / s + zp), info_.min, info_.max)
X = X.astype(ind["dtype"]).reshape((len(X),) + tuple(ind["shape"][1:]))
Y = []
for x in X:
    it.set_tensor(ind["index"], x[None]); it.invoke(); Y.append(it.get_tensor(outd["index"])[0].copy())
Y = np.stack(Y)
np.save(out / "ref_outputs.npy", Y)
ops = model_ops(a.model)
miss = [o for o in ops if o not in OPMAP]
if miss: raise SystemExit(f"op not in OPMAP: {miss}")
inb, outb = X[0].nbytes, Y[0].nbytes
(out / "model_data.h").write_text("#pragma once\nextern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n")
(out / "model_data.cc").write_text('#include "model_data.h"\n' + c_array("g_model", mbytes))
def rows(A):
    return ",\n".join("  {" + ",".join(str(b) for b in r.tobytes()) + "}" for r in A)
(out / "test_data.h").write_text("#pragma once\n#include <stdint.h>\n"
    f"#define N_TEST {len(X)}\n#define IN_BYTES {inb}\n#define OUT_BYTES {outb}\n"
    "extern const uint8_t g_test_in[N_TEST][IN_BYTES];\nextern const uint8_t g_ref_out[N_TEST][OUT_BYTES];\n")
(out / "test_data.cc").write_text('#include "test_data.h"\n'
    f"alignas(16) const uint8_t g_test_in[N_TEST][IN_BYTES] = {{\n{rows(X)}\n}};\n"
    f"alignas(16) const uint8_t g_ref_out[N_TEST][OUT_BYTES] = {{\n{rows(Y)}\n}};\n")
adds = "\n".join(f"  r.{OPMAP[o]}();" for o in ops)
(out / "op_resolver.h").write_text('#pragma once\n#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
    f"using OpResolver = tflite::MicroMutableOpResolver<{len(ops)}>;\ninline void register_ops(OpResolver& r) {{\n{adds}\n}}\n")
tens = it.get_tensor_details()
dtypes = sorted({str(t["dtype"].__name__) for t in tens})
info = dict(ops=ops, in_dtype=str(ind["dtype"].__name__), out_dtype=str(outd["dtype"].__name__),
            tensor_dtypes=dtypes, input_shape=[int(v) for v in ind["shape"]], model_bytes=len(mbytes), n_test=len(X),
            ref_out_first=Y[:3].ravel().tolist())
(out / "model_info.json").write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2))
