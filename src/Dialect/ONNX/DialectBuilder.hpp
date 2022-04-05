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
  OnnxBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  mlir::Value add(mlir::Value A, mlir::Value B) const;
  mlir::Value sub(mlir::Value A, mlir::Value B) const;
  mlir::Value mul(mlir::Value A, mlir::Value B) const;
  mlir::Value div(mlir::Value A, mlir::Value B) const;
  mlir::Value matmul(
      mlir::Type Y, mlir::Value A, mlir::Value B, bool useGemm = false) const;

  mlir::Value reshape(
      mlir::Type outputType, mlir::Value input, mlir::Value shape) const;
  mlir::Value transpose(
      mlir::Type outputType, mlir::Value input, mlir::ArrayAttr perm) const;

  mlir::Value constant(mlir::Attribute denseAttr) const;
};

// Recursive class specialized for OnnxBuilder refereed to as onnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), onnx(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), onnx(db) {}
  OnnxBuilder onnx;
};

} // namespace onnx_mlir
