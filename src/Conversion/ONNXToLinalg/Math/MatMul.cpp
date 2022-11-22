/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

#define DEBUG_TYPE "matmul"

using namespace mlir;

namespace onnx_mlir {

struct ONNXMatMulOpLinalgLowering : public ConversionPattern {
  ONNXMatMulOpLinalgLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    return success();
  }
}; // namespace onnx_mlir

void populateLoweringONNXMatMulOpLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLinalgLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
