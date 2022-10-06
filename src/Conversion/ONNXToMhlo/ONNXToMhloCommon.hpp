/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToMhloCommon.hpp - ONNX dialects to Mhlo lowering --------===//
//
// Copyright 2022
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

using namespace mlir;

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

// Get shaped constant zero for the given input Type. If the input type
// doesn't have static shape, then add dynamic broadcast.
Value getShapedZero(
    Location loc, ConversionPatternRewriter &rewriter, Value &inp);

// Get shaped constant for the given input Type and float value. If the
// input type doesn't have static shape, then add dynamic broadcast.
template <typename T>
Value getShapedFloat(Location loc, ConversionPatternRewriter &rewriter,
    const T &value, Value &inp) {
  Value broadcastedValue;
  ShapedType inpType = inp.getType().cast<ShapedType>();
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
        loc, inpType, floatValue, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedValue;
}

// Get shaped constant for the given input Type and int value. If the input
// type doesn't have static shape, then add dynamic broadcast.
template <typename T>
Value getShapedInt(Location loc, ConversionPatternRewriter &rewriter,
    const T &value, Value &inp) {
  Value broadcastedValue;
  ShapedType inpType = inp.getType().cast<ShapedType>();
  if (inpType.hasStaticShape())
    broadcastedValue = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(inpType,
                 rewriter.getIntegerAttr(inpType.getElementType(), value)));
  else {
    Type elemType = inpType.getElementType();
    Value intValue = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getIntegerAttr(elemType, value));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedValue = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inpType, intValue, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedValue;
}

llvm::SmallVector<Value, 4> getBroadcastedOperands(Operation *op,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank);

llvm::SmallVector<Value, 4> getBroadcastedOperands(
    llvm::SmallVector<Value, 4> &operands, Type outputType,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank);

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGemmOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXMatMulOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReductionOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXConvOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXNormalizationOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPoolingOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXArgMaxOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConcatOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConstantOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReshapeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXExpandOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXFlattenOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGatherOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXIdentityOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReshapeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXShapeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSliceOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSplitOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSqueezeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTileOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTransposeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXUnsqueezeOpToMhloPattern(
    RewritePatternSet &, MLIRContext *);
} // namespace onnx_mlir
