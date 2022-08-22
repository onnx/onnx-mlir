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

// Get shaped constant zero for the given input Type. If the input type doesn't
// have static shape, then add dynamic broadcast.
Value getShapedZero(Location loc, ConversionPatternRewriter &rewriter,
    const ShapedType &inpType, Value &inp, const Type &resultType);

// Get shaped constant for the given input Type and float value. If the input
// type doesn't have static shape, then add dynamic broadcast.
template <typename T>
Value getShapedFloat(Location loc, ConversionPatternRewriter &rewriter,
    const ShapedType &inpType, const T &value, Value &inp,
    const Type &resultType) {
  Value broadcastedValue;
  if (inpType.hasStaticShape())
    broadcastedValue = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(inpType,
                 rewriter.getFloatAttr(inpType.getElementType(), value)));
  else {
    Type elemType = inpType.getElementType();
    Value floatValue = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, value));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedValue = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, floatValue, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedValue;
}

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGemmOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReductionOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXNormalizationOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPoolingOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXConcatOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConstantOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReshapeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
} // namespace onnx_mlir
