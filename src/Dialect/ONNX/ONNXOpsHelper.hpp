//===------- ONNXOpsHelper.hpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

#include "onnx/onnx_pb.h"

using namespace mlir;

// Identity affine map:
// #map = affine_map<(d0)[] -> d0>
AffineMap getIdentityDimMap(Builder &builder);

// Pool/conv affine map:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) floordiv s2 + 1>
// In the case of `ceilMode = true`:
// #map0 = affine_map<(d0)[s0, s1, s2, s3]
//                    -> (d0 + s1 - (s0 - 1) * s3 - 1) ceildiv s2 + 1>
// where:
// - d0: input dim
// - s0: kernel
// - s1: pad
// - s2: stride
// - s3: dilation
AffineMap getConvDimMap(Builder &builder, bool ceilMode);

mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType);
