//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "ONNXOpsHelper.hpp"
#include "ONNXOps.hpp"

// Identity affine
using namespace mlir;
using namespace mlir::onnxmlir;
AffineMap getIdentityDimMap(Builder &builder) {
  return AffineMap::get(1, 0, {builder.getAffineDimExpr(0)});
}

// Pool/conv affine
// dim =
//   let numerator = (input + pad - (kernel - 1) * dilation - 1)
//   in let denominator = stride
//      in
//        if (ceilMode)
//          ceil(numerator / denominator) + 1
//        else
//          floor(numerator / denominator) + 1
AffineMap getConvDimMap(Builder &builder, bool ceilMode) {
  AffineExpr input = builder.getAffineDimExpr(0);
  AffineExpr kernel = builder.getAffineSymbolExpr(0);
  AffineExpr pad = builder.getAffineSymbolExpr(1);
  AffineExpr stride = builder.getAffineSymbolExpr(2);
  AffineExpr dilation = builder.getAffineSymbolExpr(3);

  AffineExpr dimExp;
  if (ceilMode)
    dimExp = (input + pad - (kernel - 1) * dilation - 1).ceilDiv(stride) + 1;
  else
    dimExp = (input + pad - (kernel - 1) * dilation - 1).floorDiv(stride) + 1;

  return AffineMap::get(1, 4, {dimExp});
}

// Convert type to MLIR type.
// A complete list of types can be found in:
// <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
// TODO: Update Int*/Uint* to emit signed/unsigned MLIR types
mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType) {
  switch (onnxType) {
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    return builder_.getF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
    return builder_.getF32Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    return builder_.getF64Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
    return builder_.getIntegerType(/*width=*/8);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
    return builder_.getIntegerType(/*width=*/8, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
    return builder_.getIntegerType(/*width=*/16);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
    return builder_.getIntegerType(/*width=*/16, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
    return builder_.getIntegerType(/*width=*/32);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
    return builder_.getIntegerType(/*width=*/32, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
    return builder_.getIntegerType(/*width=*/64);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
    return builder_.getIntegerType(/*width=*/64, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
    return builder_.getI1Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
    return StringType::get(builder_.getContext());

  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
  default:
    assert(false && "Unsupported data type encountered.");
    return nullptr;
  }
}
