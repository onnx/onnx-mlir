/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DialectBuilder.hpp - Builder for ONNX dialects -----------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_DIALECT_BUILDER_H
#define ONNX_MLIR_ONNX_DIALECT_BUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

//====-------------------------- ONNX Builder ---------------------------===//

struct OnnxBuilder : DialectBuilder {
  OnnxBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  OnnxBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  OnnxBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~OnnxBuilder(){};

  // Create operation and infer shape.
  template <typename OnnxOpType, typename... Args>
  OnnxOpType createOpAndInferShapes(Args &&... args) const;

  template <typename OnnxOpType, typename... Args>
  OnnxOpType createTypedOpAndInferShapes(
      mlir::Type result_ty, Args &&... args) const;

  // ONNXAbsOp
  mlir::Value abs(mlir::Value input) const;

  // ONNXAddOp
  mlir::Value add(mlir::Value A, mlir::Value B) const;

  // ONNXCastOp
  mlir::Value cast(mlir::Type outputType, mlir::Value input,
      mlir::IntegerAttr saturate, mlir::TypeAttr to,
      bool inferShape = true) const;
  mlir::Value cast(
      mlir::Value input, mlir::IntegerAttr saturate, mlir::TypeAttr to) const;
  mlir::Value cast(mlir::Value input, mlir::TypeAttr to) const;
  mlir::Value cast(mlir::Value input, mlir::Type to) const;

  // ONNXCeilOp
  mlir::Value ceil(mlir::Value input) const;

  // ONNXClipOp
  mlir::Value clip(mlir::Value input, mlir::Value min, mlir::Value max,
      bool scalarType = false) const;

  // ONNXConcatOp
  mlir::Value concat(
      mlir::Type outputType, mlir::ValueRange inputs, int64_t axis) const;

  // ONNXConstantOp
  mlir::Value constant(mlir::Attribute denseAttr) const;
  mlir::Value constantInt64(const mlir::ArrayRef<int64_t> intVals) const;

  // ONNXConvOp
  mlir::Value conv(mlir::Type Y, mlir::Value X, mlir::Value W, mlir::Value B,
      llvm::StringRef autoPad, mlir::ArrayRef<int64_t> dilations, int64_t group,
      mlir::ArrayRef<int64_t> kernelShape, mlir::ArrayRef<int64_t> pads,
      mlir::ArrayRef<int64_t> strides) const;

  // ONNXDequantizeLinearOp
  mlir::Value dequantizeLinear(mlir::Type resType, mlir::Value X,
      mlir::Value scale, mlir::Value zeroPoint, int axis = 1) const;

  // ONNXDivOp
  mlir::Value div(mlir::Value A, mlir::Value B) const;

  // ONNXDimOp
  mlir::Value dim(mlir::Value input, int axis) const;

  // ONNXDimGroupOp
  void dimGroup(mlir::Value input, int axis, int groupID) const;

  // ONNXExpandOp
  mlir::Value expand(
      mlir::Type outputType, mlir::Value input, mlir::Value shape) const;

  // ONNXGeluOp
  mlir::Value gelu(mlir::Value input, mlir::StringAttr approximateAttr) const;

  // ONNXLayerNormalizationOp, version with one output only (Y).
  mlir::Value layerNorm(mlir::Type outputType, mlir::Value input,
      mlir::Value scale, mlir::Value bias, int64_t axis,
      mlir::FloatAttr epsilon) const;
  // In the case of GroupNormalization when stashType can be specified
  mlir::Value layerNorm(mlir::Type outputType, mlir::Value input,
      mlir::Value scale, mlir::Value bias, int64_t axis,
      mlir::FloatAttr epsilon, mlir::IntegerAttr stashType) const;

  // ONNXQLinearMatMulOp
  mlir::Value qlinearMatMul(mlir::Type outputType, mlir::Value a,
      mlir::Value aScale, mlir::Value aZeroPoint, mlir::Value b,
      mlir::Value bScale, mlir::Value bZeroPoint, mlir::Value yScale,
      mlir::Value yZeroPoint) const;

  // ONNXRMSLayerNormalizationOp, version with one output only (Y).
  mlir::Value RMSLayerNorm(mlir::Type outputType, mlir::Value input,
      mlir::Value scale, mlir::Value bias, int64_t axis,
      mlir::FloatAttr epsilon) const;

  // ONNXMatMulOp or ONNXGemmOp
  mlir::Value matmul(
      mlir::Type Y, mlir::Value A, mlir::Value B, bool useGemm = false) const;

  // ONNXMaxOp
  mlir::Value max(mlir::ValueRange inputs) const;

  // ONNXMinOp
  mlir::Value min(mlir::ValueRange inputs) const;

  // ONNXMulOp
  mlir::Value mul(mlir::Value A, mlir::Value B) const;
  mlir::Value mul(mlir::Type resultType, mlir::Value A, mlir::Value B) const;

  // ONNXNoneOp
  mlir::Value none() const;

  // ONNXPadOp
  mlir::Value pad(mlir::Value input, mlir::Value pads,
      mlir::Value constantValue, std::string mode = "constant") const;
  // Zero padding
  mlir::Value padZero(mlir::Value input, mlir::Value pads) const;

  // ONNXReduceMaxOp
  mlir::Value reduceMax(mlir::Type outputType, mlir::Value data,
      mlir::Value axes, bool keepDims = true,
      bool noop_with_empty_axes = false) const;

  // ONNXReduceMeanOp
  mlir::Value reduceMean(mlir::Type outputType, mlir::Value data,
      mlir::Value axes, bool keepDims = true,
      bool noop_with_empty_axes = false) const;

  // ONNXReduceMinOp
  mlir::Value reduceMin(mlir::Type outputType, mlir::Value data,
      mlir::Value axes, bool keepDims = true,
      bool noop_with_empty_axes = false) const;

  // ONNXReduceSumOp
  mlir::Value reduceSum(mlir::Type outputType, mlir::Value data,
      mlir::Value axes, bool keepDims = true,
      bool noop_with_empty_axes = false) const;

  // ONNXReshapeOp
  mlir::Value reshape(
      mlir::Type outputType, mlir::Value input, mlir::Value shape) const;
  mlir::Value reshape(mlir::Type outputType, mlir::Value input,
      mlir::Value shape, mlir::IntegerAttr allowZero) const;
  // Reshape input val to a N-dimensional shape; when collapseMostSignificant is
  // true, we collapse the most significant dimensions (and preserve the N-1
  // least significant dims); otherwise we collapse the least significant
  // dimensions (and preserve the N-1 most significant dims).
  mlir::Value reshapeToNDim(
      mlir::Value val, int64_t N, bool collapseMostSignificant) const;

  // ONNXReciprocalOp
  mlir::Value reciprocal(mlir::Value input) const;

  // ONNXReverseSequenceOp
  mlir::Value reverseSequence(mlir::Type outputType, mlir::Value input,
      mlir::Value sequenceLens, int64_t batchAxis, int64_t timeAxis) const;

  // ONNXRoundOp
  mlir::Value round(mlir::Value input, bool scalarType = false) const;

  // ONNXShapeOp (start is inclusive, default 0; end is exclusive, default
  // nullptr means all)
  mlir::Value shape(mlir::Value input) const; // Infer the type.
  mlir::Value shape(mlir::Type outputType, mlir::Value input) const;
  mlir::Value shape(
      mlir::Type outputType, mlir::Value input, int64_t start) const;
  mlir::Value shape(mlir::Type outputType, mlir::Value input, int64_t start,
      int64_t end) const;
  // Get the shape of an input and permute the positions of its shape dims. Perm
  // values are in the range [0, rank(input)). Say an 4D input with dims (d0,
  // d1, d2, d3). Call to "Shape(input, {0, 1, 3, 2})" will produce a tensor
  // with "[d0, d1, d3, d2]" values. Or call to "Shape(input, {0, 2, 3})" will
  // produce a shape of reduced dimensions (4D->3D) with dims "[d0, d2, d3]".
  mlir::Value shape(mlir::Value input, mlir::ArrayRef<int64_t> perm) const;

  // ONNXSliceOp
  mlir::Value slice(mlir::Type outputType, mlir::Value input,
      mlir::Value starts, mlir::Value ends, mlir::Value axes,
      mlir::Value steps) const;
  mlir::Value slice(mlir::Type outputType, mlir::Value input, int64_t start,
      int64_t end, int64_t step = 1) const; // 1D slice

  // ONNXSqrtOp
  mlir::Value sqrt(mlir::Value input) const;

  // ONNXSplitOp
  mlir::ValueRange split(mlir::TypeRange outputTypes, mlir::Value input,
      mlir::Value split, int64_t axis) const;

  // ONNXSqueezeOp
  mlir::Value squeeze(
      mlir::Type outputType, mlir::Value data, mlir::Value axes) const;

  // ONNXSubOp
  mlir::Value sub(mlir::Value A, mlir::Value B) const;

  // ONNXSumOp
  mlir::Value sum(mlir::Type outputType, mlir::ValueRange inputs) const;

  // UnrealizedConversionCastOp
  // Convert a Value to TensorType if it is of MemRefType.
  mlir::Value toTensor(mlir::Value input) const;
  // Convert a Type to TensorType if it is of MemRefType.
  mlir::TensorType toTensor(mlir::Type input) const;
  // Convert Type to TypeRange of TensorType if it is of MemRefType.
  mlir::TypeRange toTensors(mlir::TypeRange inputs) const;
  // Convert a Value to MemrefType if it is of TensorType.
  mlir::Value toMemref(mlir::Value input) const;

  // ONNXTransposeOp
  mlir::Value transpose(
      mlir::Type outputType, mlir::Value input, mlir::ArrayAttr perm) const;
  mlir::Value transposeInt64(
      mlir::Value input, mlir::ArrayRef<int64_t> intPerm) const;

  // ONNXUnsqueezeOp
  mlir::Value unsqueeze(
      mlir::Type outputType, mlir::Value data, mlir::Value axes) const;

  // ONNXWhereOp
  mlir::Value where(mlir::Type outputType, mlir::Value condition, mlir::Value X,
      mlir::Value Y) const;

  // =============================================================================
  // Fold and emit support.
  // =============================================================================

  // These utilities emit an ONNXOp and try to fold it if possible. If the input
  // is constant, do const propagation, and return a constant. The funcion needs
  // to have std::function<DenseElementsAttr(mlir::Value value)> as the last
  // argument. It is used to get the DenseElementsAttr from the value if the
  // value is a constant.

  using DenseElementsAttrGetter =
      std::function<mlir::DenseElementsAttr(mlir::Value)>;

  /// Emit an ONNXSqueezeOp. If the input is constant, do const propagation, and
  /// return a constant.
  mlir::Value foldOrEmitONNXSqueezeOp(mlir::ConversionPatternRewriter &rewriter,
      mlir::Location loc, mlir::Type resultType, mlir::Value input,
      int64_t axis, DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXSqueezeV11Op. If the input is constant, do const propagation,
  /// and return a constant.
  mlir::Value foldOrEmitONNXSqueezeV11Op(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Type resultType, mlir::Value input, int64_t axis,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXUnsqueezeOp. If the input is constant, do const propagation,
  /// and return a constant.
  mlir::Value foldOrEmitONNXUnsqueezeOp(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Type resultType, mlir::Value input, int64_t axis,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXUnsqueezeV11Op. If the input is constant, do const
  /// propagation, and return a constant.
  mlir::Value foldOrEmitONNXUnsqueezeV11Op(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Type resultType, mlir::Value input, int64_t axis,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
  /// return constants.
  /// Only support evenly splitting.
  std::vector<mlir::Value> foldOrEmitONNXSplitOp(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      llvm::ArrayRef<mlir::Type> resultTypes, mlir::Value input, int64_t axis,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXSplitV11Op. If the input is constant, do const propagation,
  /// and return constants. Only support evenly splitting.
  std::vector<mlir::Value> foldOrEmitONNXSplitV11Op(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      llvm::ArrayRef<mlir::Type> resultTypes, mlir::Value input, int64_t axis,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

  /// Emit an ONNXTransposeOp. If the input is constant, do const propagation,
  /// and return a constant.
  mlir::Value foldOrEmitONNXTransposeOp(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Type resultType, mlir::Value input, mlir::ArrayAttr permAttr,
      DenseElementsAttrGetter getDenseElementAttrFromConstValue);

private:
  mlir::IntegerAttr getSignedInt64Attr(int64_t n) const;
};

// Recursive class specialized for OnnxBuilder refereed to as onnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), onnx(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), onnx(db) {}
  OnnxBuilder onnx;
};

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// This class is not meant to work with the MultiDialectBuilder as it is not
// used for building, only for analysis. We force OpBuilder to be null as
// missing builder is used within IndexExpr as a sign that we are in shape
// inference analysis. Be mindful not to expect builder to then be passed to
// other builders.

struct IndexExprBuilderForAnalysis : IndexExprBuilder {
  IndexExprBuilderForAnalysis(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForAnalysis(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(loc) {} // Builder omitted during analysis.
  IndexExprBuilderForAnalysis(const DialectBuilder &db)
      : IndexExprBuilder(db.getLoc()) {} // Builder omitted during analysis.
  virtual ~IndexExprBuilderForAnalysis() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// Include inline code definitions.
#include "DialectBuilder.hpp.inc"

} // namespace onnx_mlir
#endif
