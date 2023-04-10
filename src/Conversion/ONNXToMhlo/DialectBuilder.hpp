/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - Mhlo dialect builder --------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains dialect builder for Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "mhlo/IR/hlo_ops.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder for Shape lowering
// =============================================================================

struct IndexExprBuilderForMhlo : IndexExprBuilder {
  IndexExprBuilderForMhlo(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForMhlo(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForMhlo(const DialectBuilder &db) : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForMhlo() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForMhlo, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), mhloIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), mhloIE(db) {}
  IndexExprBuilderForMhlo mhloIE;
};

} // namespace onnx_mlir
