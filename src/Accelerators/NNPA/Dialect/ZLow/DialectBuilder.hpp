/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - ZLow Dialect Builder -----------------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the ZLow Dialect Builder.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder for building
// =============================================================================

struct IndexExprBuilderForZLow : IndexExprBuilder {
  IndexExprBuilderForZLow(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForZLow(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForZLow(const DialectBuilder &db) : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForZLow() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// =============================================================================
// MultiDialectBuilder for ZLow
// =============================================================================

// Recursive class specialized for IndexExprBuilderForZLow referred to as
// zlowIE.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForZLow, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), zlowIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), zlowIE(db) {}
  IndexExprBuilderForZLow zlowIE;
};

} // namespace onnx_mlir
