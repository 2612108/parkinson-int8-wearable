#pragma once
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
using OpResolver = tflite::MicroMutableOpResolver<5>;
inline void register_ops(OpResolver& r) {
  r.AddAdd();
  r.AddConv2D();
  r.AddDepthwiseConv2D();
  r.AddFullyConnected();
  r.AddMean();
}
