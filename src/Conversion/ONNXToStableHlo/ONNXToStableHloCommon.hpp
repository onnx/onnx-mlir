/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToStableHloCommon.hpp - ONNX dialects to StableHlo lowering
//--------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// This is to get a stablehlo operation of a given type for a specific
// operation.
//===----------------------------------------------------------------------===//
template <typename ONNXOp>
struct StableHloDialectOp {
  using Op = void;
};

template <typename ONNXOp>
using StableHloOp = typename StableHloDialectOp<ONNXOp>::Op;

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to StableHlo.
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
    broadcastedValue = rewriter.create<stablehlo::ConstantOp>(
        loc, DenseElementsAttr::get(inpType,
                 rewriter.getFloatAttr(inpType.getElementType(), value)));
  else {
    Type elemType = inpType.getElementType();
    Value floatValue = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, value));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedValue = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
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
    broadcastedValue = rewriter.create<stablehlo::ConstantOp>(
        loc, DenseElementsAttr::get(inpType,
                 rewriter.getIntegerAttr(inpType.getElementType(), value)));
  else {
    Type elemType = inpType.getElementType();
    Value intValue = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getIntegerAttr(elemType, value));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedValue = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, inpType, intValue, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedValue;
}

llvm::SmallVector<Value, 4> getBroadcastedOperands(Operation *op,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank);

llvm::SmallVector<Value, 4> getBroadcastedOperands(
    llvm::SmallVector<Value, 4> &operands, Type outputType,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank);

mlir::ElementsAttr getElementAttributeFromConstValue(mlir::Value value);

DenseIntElementsAttr GetI64ElementsAttr(
    ArrayRef<int64_t> values, Builder *builder);

//===----------------------------------------------------------------------===//
// Fold and emit support.
//===----------------------------------------------------------------------===//

/// Emit an ONNXSqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXSqueezeOpStableHlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXUnsqueezeOpStableHlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<mlir::Value> foldOrEmitONNXSplitOpStableHlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::ArrayRef<mlir::Type> resultTypes, mlir::Value input, int64_t axis);

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXTransposeOpStableHlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, mlir::ArrayAttr permAttr);

// `Math` directory methods:
void populateLoweringONNXClipOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXElementwiseOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGemmOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXMatMulOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReductionOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXConvOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConvTransposeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXNormalizationOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPoolingOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
// `RNN` directory methods:
void populateLoweringONNXLSTMOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *, bool);
// `Tensor` directory methods:
void populateLoweringONNXArgMaxOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConcatOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConstantOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXDepthToSpaceOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXExpandOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXFlattenOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGatherOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGatherElementsOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXIdentityOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXOneHotOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPadOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReshapeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXScatterNDOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXShapeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSliceOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSplitOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSqueezeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTileOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTransposeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXUnsqueezeOpToStableHloPattern(
    RewritePatternSet &, MLIRContext *);
} // namespace onnx_mlir
