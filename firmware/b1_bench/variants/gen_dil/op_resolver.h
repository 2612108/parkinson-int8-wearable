#pragma once
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
using OpResolver = tflite::MicroMutableOpResolver<9>;
inline void register_ops(OpResolver& r) {
  r.AddAdd();
  r.AddBatchToSpaceNd();
  r.AddConv2D();
  r.AddDepthwiseConv2D();
  r.AddExpandDims();
  r.AddFullyConnected();
  r.AddMean();
  r.AddReshape();
  r.AddSpaceToBatchNd();
}
