/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- DialectBuilder.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains builder functions for ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

// Identity affine
namespace onnx_mlir {

//====-------------------------- ONNX Builder ---------------------------===//

Value OnnxBuilder::add(Value A, Value B) const {
  return b.create<ONNXAddOp>(loc, A, B);
}

Value OnnxBuilder::sub(Value A, Value B) const {
  return b.create<ONNXSubOp>(loc, A, B);
}

Value OnnxBuilder::mul(Value A, Value B) const {
  return b.create<ONNXMulOp>(loc, A, B);
}

Value OnnxBuilder::div(Value A, Value B) const {
  return b.create<ONNXDivOp>(loc, A, B);
}

Value OnnxBuilder::matmul(Type Y, Value A, Value B, bool useGemm) const {
  // Gemm only supports rank 2.
  bool canUseGemm = useGemm && A.getType().isa<ShapedType>() &&
                    A.getType().cast<ShapedType>().hasRank() &&
                    (A.getType().cast<ShapedType>().getRank() == 2) &&
                    B.getType().isa<ShapedType>() &&
                    B.getType().cast<ShapedType>().hasRank() &&
                    (B.getType().cast<ShapedType>().getRank() == 2);
  if (canUseGemm)
    return b.create<ONNXGemmOp>(loc, Y, A, B, b.createOrFold<ONNXNoneOp>(loc),
        /*alpha=*/b.getF32FloatAttr(1.0), /*beta=*/b.getF32FloatAttr(1.0),
        /*transA=*/
        IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 0, /*isSigned=*/true)),
        /*transB=*/
        IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 0, /*isSigned=*/true)));
  return b.create<ONNXMatMulOp>(loc, Y, A, B);
}

Value OnnxBuilder::reshape(Type outputType, Value input, Value shape) const {
  return b.create<ONNXReshapeOp>(loc, outputType, input, shape);
}

Value OnnxBuilder::transpose(
    Type outputType, Value input, ArrayAttr perm) const {
  return b.create<ONNXTransposeOp>(loc, outputType, input, perm);
}

Value OnnxBuilder::constant(Attribute denseAttr) const {
  return b.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

} // namespace onnx_mlir
