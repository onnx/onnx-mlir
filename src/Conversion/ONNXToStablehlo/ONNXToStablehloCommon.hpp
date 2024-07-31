/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- ONNXToStablehloCommon.hpp - ONNX dialects to Stablehlo lowering--===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_STABLEHLO_H
#define ONNX_MLIR_ONNX_TO_STABLEHLO_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
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
struct StablehloDialectOp {
  using Op = void;
};

template <typename ONNXOp>
using StablehloOp = typename StablehloDialectOp<ONNXOp>::Op;

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to Stablehlo.
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
  ShapedType inpType = mlir::cast<ShapedType>(inp.getType());
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
        loc, inpType, floatValue, shape, rewriter.getDenseI64ArrayAttr({}));
  }
  return broadcastedValue;
}

// Get shaped constant for the given input Type and int value. If the input
// type doesn't have static shape, then add dynamic broadcast.
template <typename T>
Value getShapedInt(Location loc, ConversionPatternRewriter &rewriter,
    const T &value, Value &inp) {
  Value broadcastedValue;
  ShapedType inpType = mlir::cast<ShapedType>(inp.getType());
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
        loc, inpType, intValue, shape, rewriter.getDenseI64ArrayAttr({}));
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
mlir::Value foldOrEmitONNXSqueezeOpStablehlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXUnsqueezeOpStablehlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, int64_t axis);

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<mlir::Value> foldOrEmitONNXSplitOpStablehlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::ArrayRef<mlir::Type> resultTypes, mlir::Value input, int64_t axis);

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation, and
/// return a constant.
mlir::Value foldOrEmitONNXTransposeOpStablehlo(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Type resultType, mlir::Value input, mlir::ArrayAttr permAttr);

// `Math` directory methods:
void populateLoweringONNXClipOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXElementwiseOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGemmOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXMatMulOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReductionOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXConvOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConvTransposeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXNormalizationOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPoolingOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
// `RNN` directory methods:
void populateLoweringONNXLSTMOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *, bool);
// `Tensor` directory methods:
void populateLoweringONNXArgMaxOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConcatOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXConstantOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXDimOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXDepthToSpaceOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXExpandOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXFlattenOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGatherOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXGatherElementsOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXIdentityOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXOneHotOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXPadOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXReshapeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXScatterNDOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXShapeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSliceOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSplitOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSqueezeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTileOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXTransposeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXUnsqueezeOpToStablehloPattern(
    RewritePatternSet &, MLIRContext *);
void populateLoweringONNXSoftmaxOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx);
} // namespace onnx_mlir
#endif
