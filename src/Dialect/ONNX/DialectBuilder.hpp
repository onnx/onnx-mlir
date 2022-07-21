/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DialectBuilder.hpp - Builder for ONNX dialects -----------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"

namespace onnx_mlir {

//====-------------------------- ONNX Builder ---------------------------===//

struct OnnxBuilder : onnx_mlir::DialectBuilder {
  OnnxBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  OnnxBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  // ONNXAddOp
  mlir::Value add(mlir::Value A, mlir::Value B) const;

  // ONNXCastOp
  mlir::Value cast(mlir::Value input, mlir::TypeAttr to) const;

  // ONNXCeilOp
  mlir::Value ceil(mlir::Value input) const;

  // ONNXConstantOp
  mlir::Value constant(mlir::Attribute denseAttr) const;
  mlir::Value constantFromRawBuffer(mlir::Type resultType, char *buf) const;

  // ONNXDivOp
  mlir::Value div(mlir::Value A, mlir::Value B) const;

  // ONNXMatMulOp or ONNXGemmOp
  mlir::Value matmul(
      mlir::Type Y, mlir::Value A, mlir::Value B, bool useGemm = false) const;

  // ONNXMinOp
  mlir::Value min(mlir::ValueRange inputs) const;

  // ONNXMulOp
  mlir::Value mul(mlir::Value A, mlir::Value B) const;

  // ONNXReduceSumOp
  mlir::Value reduceSum(mlir::Type outputType, mlir::Value data,
      mlir::Value axes, bool keepdims = true,
      bool noop_with_empty_axes = false) const;

  // ONNXReshapeOp
  mlir::Value reshape(
      mlir::Type outputType, mlir::Value input, mlir::Value shape) const;

  // ONNXSqueezeOp
  mlir::Value squeeze(
      mlir::Type outputType, mlir::Value data, mlir::Value axes) const;

  // ONNXSubOp
  mlir::Value sub(mlir::Value A, mlir::Value B) const;

  // UnrealizedConversionCastOp
  // Convert a Value to TensorType if it is of MemRefType.
  mlir::Value toTensor(mlir::Value input) const;
  // Convert a Type to TensorType if it is of MemRefType.
  mlir::Type toTensor(mlir::Type input) const;
  // Convert a Value to MemrefType if it is of TensorType.
  mlir::Value toMemref(mlir::Value input) const;

  // ONNXTransposeOp
  mlir::Value transpose(
      mlir::Type outputType, mlir::Value input, mlir::ArrayAttr perm) const;

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

} // namespace onnx_mlir
