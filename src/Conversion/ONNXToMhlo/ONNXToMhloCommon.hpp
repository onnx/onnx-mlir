/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToMhloCommon.hpp - ONNX dialects to Mhlo lowering --------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// This is to get a mhlo operation of a given type for a specific operation.
//===----------------------------------------------------------------------===//
template <typename ONNXOp>
struct MhloDialectOp {
  using Op = void;
};

template <typename ONNXOp>
using MhloOp = typename MhloDialectOp<ONNXOp>::Op;

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to MHLO.
//===----------------------------------------------------------------------===//

// Get shaped constant zero for the given input mlir::Type. If the input type
// doesn't have static shape, then add dynamic broadcast.
mlir::Value getShapedZero(mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter, const mlir::ShapedType &inpType,
    mlir::Value &inp, const mlir::Type &resultType);

// Get shaped constant for the given input mlir::Type and float value. If the
// input type doesn't have static shape, then add dynamic broadcast.
template <typename T>
mlir::Value getShapedFloat(mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter, const mlir::ShapedType &inpType,
    const T &value, mlir::Value &inp, const mlir::Type &resultType) {
  mlir::Value broadcastedValue;
  if (inpType.hasStaticShape())
    broadcastedValue = rewriter.create<mlir::mhlo::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(inpType,
                 rewriter.getFloatAttr(inpType.getElementType(), value)));
  else {
    mlir::Type elemType = inpType.getElementType();
    mlir::Value floatValue = rewriter.create<mlir::mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, value));
    mlir::Value shape = rewriter.create<mlir::shape::ShapeOfOp>(loc, inp);
    broadcastedValue = rewriter.create<mlir::mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, floatValue, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedValue;
}

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
void populateLoweringONNXGemmOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
void populateLoweringONNXReductionOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXNormalizationOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
void populateLoweringONNXPoolingOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXConcatOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
void populateLoweringONNXConstantOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
void populateLoweringONNXReshapeOpToMhloPattern(
    mlir::RewritePatternSet &, mlir::MLIRContext *);
} // namespace onnx_mlir
