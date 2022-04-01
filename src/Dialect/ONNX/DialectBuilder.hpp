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

namespace mlir {

//====-------------------------- ONNX Builder ---------------------------===//

struct OnnxBuilder : DialectBuilder {
  OnnxBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  OnnxBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value add(Value A, Value B) const;
  Value sub(Value A, Value B) const;
  Value mul(Value A, Value B) const;
  Value div(Value A, Value B) const;
  Value matmul(Type Y, Value A, Value B, bool useGemm = false) const;

  Value reshape(Type outputType, Value input, Value shape) const;
  Value transpose(Type outputType, Value input, ArrayAttr perm) const;

  Value constant(Attribute denseAttr) const;
};

// Recursive class specialized for OnnxBuilder refereed to as onnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), onnx(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), onnx(db) {}
  OnnxBuilder onnx;
};

} // namespace mlir
