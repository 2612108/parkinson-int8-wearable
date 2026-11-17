"""Đóng gói một mô hình .tflite INT8 + dữ liệu vào thành mã C cho firmware b1_bench.

python tools/tflite_to_c.py --model m.tflite --inputs x.npy [--n 50] --out firmware/b1_bench/main/generated

x.npy: float32 (N, T, C) — sẽ lượng tử hóa bằng scale/zero-point của tensor vào; hoặc int8 sẵn.
Sinh ra:
  model_data.{h,cc}   mảng mô hình (căn 16 byte)
  test_data.{h,cc}    N mẫu vào INT8 + N đầu ra tham chiếu (TFLite trên PC)
  op_resolver.h       MicroMutableOpResolver chỉ gồm các op có trong mô hình
  ref_outputs.npy     đầu ra INT8 tham chiếu (để check_serial.py so bit-exact)
  model_info.json     op list, shape, scale/zp, kích thước
"""
import argparse, json, pathlib, sys
import numpy as np

_pydeps = pathlib.Path(__file__).resolve().parents[1] / ".pydeps"
if _pydeps.is_dir() and str(_pydeps) not in sys.path:
    sys.path.insert(0, str(_pydeps))
try:
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as sch
    _Interpreter = tf.lite.Interpreter
    _BUILTIN_REF = tf.lite.experimental.OpResolverType.BUILTIN_REF
except Exception:
    from ai_edge_litert import interpreter as _tfl
    from ai_edge_litert import schema_py_generated as sch
    _Interpreter = _tfl.Interpreter
    _BUILTIN_REF = _tfl.OpResolverType.BUILTIN_REF

# tên op TFLite -> phương thức MicroMutableOpResolver
OPMAP = {
    "CONV_2D": "AddConv2D", "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D", "FULLY_CONNECTED": "AddFullyConnected",
    "RESHAPE": "AddReshape", "EXPAND_DIMS": "AddExpandDims", "SQUEEZE": "AddSqueeze", "MEAN": "AddMean",
    "SUM": "AddSum", "SOFTMAX": "AddSoftmax", "ADD": "AddAdd", "MUL": "AddMul", "SUB": "AddSub",
    "QUANTIZE": "AddQuantize", "DEQUANTIZE": "AddDequantize", "PAD": "AddPad", "PADV2": "AddPadV2",
    "SPACE_TO_BATCH_ND": "AddSpaceToBatchNd", "BATCH_TO_SPACE_ND": "AddBatchToSpaceNd",
    "LOGISTIC": "AddLogistic", "TANH": "AddTanh", "RELU": "AddRelu", "RELU6": "AddRelu6",
    "MAX_POOL_2D": "AddMaxPool2D", "AVERAGE_POOL_2D": "AddAveragePool2D", "CONCATENATION": "AddConcatenation",
    "TRANSPOSE": "AddTranspose", "STRIDED_SLICE": "AddStridedSlice", "SHAPE": "AddShape", "PACK": "AddPack",
    "UNPACK": "AddUnpack", "GATHER": "AddGather", "REDUCE_MAX": "AddReduceMax", "BATCH_MATMUL": "AddBatchMatMul",
    "HARD_SWISH": "AddHardSwish", "LEAKY_RELU": "AddLeakyRelu", "SPLIT": "AddSplit", "SLICE": "AddSlice",
}


def model_ops(path):
    buf = open(path, "rb").read()
    m = sch.Model.GetRootAsModel(buf, 0)
    names = {v: k for k, v in vars(sch.BuiltinOperator).items() if not k.startswith("_")}
    ops = set()
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        ops.add(names[code])
    return sorted(ops)


def c_array(name, data: bytes, ctype="unsigned char", align=True):
    al = "alignas(16) " if align else ""
    rows = [", ".join(f"0x{b:02x}" for b in data[i:i + 16]) for i in range(0, len(data), 16)]
    return f"{al}const {ctype} {name}[] = {{\n  " + ",\n  ".join(rows) + f"\n}};\nconst unsigned int {name}_len = {len(data)};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--inputs", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mbytes = pathlib.Path(a.model).read_bytes()

    # kernel tham chiếu (BUILTIN_REF) gần TFLM nhất; mặc định (XNNPACK/optimized) có thể lệch vài LSB
    it = _Interpreter(model_content=mbytes,
                      experimental_op_resolver_type=_BUILTIN_REF)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    assert ind["dtype"] == np.int8 and outd["dtype"] == np.int8, "mô hình phải vào/ra INT8"
    X = np.load(a.inputs)[: a.n]
    if X.dtype != np.int8:
        s, zp = ind["quantization"]
        X = np.clip(np.round(X / s + zp), -128, 127).astype(np.int8)
    X = X.reshape((len(X),) + tuple(ind["shape"][1:]))
    Y = []
    for x in X:
        it.set_tensor(ind["index"], x[None]); it.invoke(); Y.append(it.get_tensor(outd["index"])[0].copy())
    Y = np.stack(Y).astype(np.int8)
    np.save(out / "ref_outputs.npy", Y)

    ops = model_ops(a.model)
    miss = [o for o in ops if o not in OPMAP]
    if miss:
        raise SystemExit(f"op chưa có trong OPMAP (bổ sung hoặc đổi kiến trúc): {miss}")
    in_sz, out_sz = int(np.prod(ind["shape"][1:])), int(np.prod(outd["shape"][1:]))

    (out / "model_data.h").write_text("#pragma once\nextern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n")
    (out / "model_data.cc").write_text('#include "model_data.h"\n' + c_array("g_model", mbytes))
    (out / "test_data.h").write_text(
        "#pragma once\n#include <stdint.h>\n"
        f"#define N_TEST {len(X)}\n#define IN_SIZE {in_sz}\n#define OUT_SIZE {out_sz}\n"
        "extern const int8_t g_test_in[N_TEST][IN_SIZE];\nextern const int8_t g_ref_out[N_TEST][OUT_SIZE];\n")
    def rows(A):
        return ",\n".join("  {" + ",".join(str(int(v)) for v in r.ravel()) + "}" for r in A)
    (out / "test_data.cc").write_text('#include "test_data.h"\n'
        f"const int8_t g_test_in[N_TEST][IN_SIZE] = {{\n{rows(X)}\n}};\n"
        f"const int8_t g_ref_out[N_TEST][OUT_SIZE] = {{\n{rows(Y)}\n}};\n")
    adds = "\n".join(f"  r.{OPMAP[o]}();" for o in ops)
    (out / "op_resolver.h").write_text(
        '#pragma once\n#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
        f"using OpResolver = tflite::MicroMutableOpResolver<{len(ops)}>;\n"
        f"inline void register_ops(OpResolver& r) {{\n{adds}\n}}\n")
    info = dict(ops=ops, input_shape=[int(v) for v in ind["shape"]], output_shape=[int(v) for v in outd["shape"]],
                input_quant=[float(v) for v in ind["quantization"]], output_quant=[float(v) for v in outd["quantization"]],
                model_bytes=len(mbytes), n_test=len(X))
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
