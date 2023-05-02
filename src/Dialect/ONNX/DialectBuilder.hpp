/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DialectBuilder.hpp - Builder for ONNX dialects -----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

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

  // ONNXAddOp
  mlir::Value add(mlir::Value A, mlir::Value B) const;

  // ONNXCastOp
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

  // ONNXDivOp
  mlir::Value div(mlir::Value A, mlir::Value B) const;

  // ONNXDimOp
  mlir::Value dim(mlir::Value input, int axis) const;

  // ONNXDimGroupOp
  void dimGroup(mlir::Value input, int axis, int groupID) const;

  // ONNXMatMulOp or ONNXGemmOp
  mlir::Value matmul(
      mlir::Type Y, mlir::Value A, mlir::Value B, bool useGemm = false) const;

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
  // Reshape input val to a N-dimensional shape; when collapseMostSignificant is
  // true, we collapse the most significant dimensions (and preserve the N-1
  // least significant dims); otherwise we collapse the least significant
  // dimensions (and preserve the N-1 most significant dims).
  mlir::Value reshapeToNDim(
      mlir::Value val, int64_t N, bool collapseMostSignificant) const;

  // ONNXReverseSequenceOp
  mlir::Value reverseSequence(mlir::Type outputType, mlir::Value input,
      mlir::Value sequenceLens, int64_t batchAxis, int64_t timeAxis) const;

  // ONNXRoundOp
  mlir::Value round(mlir::Value input, bool scalarType = false) const;

  // ONNXShapeOp
  mlir::Value shape(mlir::Type outputType, mlir::Value input) const;

  // ONNXSliceOp
  mlir::Value slice(mlir::Type outputType, mlir::Value input,
      mlir::Value starts, mlir::Value ends, mlir::Value axes,
      mlir::Value steps) const;
  mlir::Value slice(mlir::Type outputType, mlir::Value input, int64_t start,
      int64_t end, int64_t step = 1) const; // 1D slice

  // ONNXSplitOp
  mlir::ValueRange split(mlir::TypeRange outputTypes, mlir::Value input,
      mlir::Value split, int64_t axis) const;

  // ONNXSqueezeOp
  mlir::Value squeeze(
      mlir::Type outputType, mlir::Value data, mlir::Value axes) const;

  // ONNXSubOp
  mlir::Value sub(mlir::Value A, mlir::Value B) const;

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
