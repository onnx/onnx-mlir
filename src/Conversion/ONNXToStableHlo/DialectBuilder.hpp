/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - StableHlo dialect builder
//--------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains dialect builder for StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// stablehlo Builder
// =============================================================================

struct StablehloBuilder : DialectBuilder {
  StablehloBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  StablehloBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc), patternRewriter(&b) {}
  StablehloBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~StablehloBuilder() {}

  // ConstantOp
  mlir::Value constant(mlir::Type type, double val) const;
  mlir::Value constantI64(int64_t val) const;
  mlir::Value shaped_zero(mlir::Type type) const;
  // ReshapeOp
  mlir::Value reshape(mlir::Type resultType, mlir::Value operand) const;
  // SliceOp
  mlir::Value real_dynamic_slice(mlir::Type type, mlir::Value operand,
      mlir::Value startIndices, mlir::Value limitIndices,
      mlir::Value strides) const;
  mlir::Value dynamic_slice(mlir::Value operand,
      mlir::SmallVector<mlir::Value> startIndices,
      mlir::SmallVector<int64_t> sliceSizes) const;
  mlir::Value dynamic_slice(mlir::Value operand,
      mlir::SmallVector<mlir::Value> startIndices,
      mlir::DenseI64ArrayAttr sliceSizes) const;
  mlir::Value slice(mlir::Value operand,
      mlir::SmallVector<int64_t> startIndices,
      mlir::SmallVector<int64_t> limitIndices,
      mlir::SmallVector<int64_t> strides) const;
  mlir::Value slice(mlir::Value operand, mlir::DenseI64ArrayAttr startIndices,
      mlir::DenseI64ArrayAttr limitIndices,
      mlir::DenseI64ArrayAttr strides) const;

protected:
  // Private getters of builder (concise version).
  mlir::OpBuilder &rewriter() const {
    assert(patternRewriter && "rewriter is null");
    return *patternRewriter;
  }

private:
  mlir::OpBuilder *patternRewriter;
};

// =============================================================================
// IndexExpr Builder for Shape lowering
// =============================================================================

struct IndexExprBuilderForStableHlo : IndexExprBuilder {
  IndexExprBuilderForStableHlo(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForStableHlo(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForStableHlo(const DialectBuilder &db)
      : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForStableHlo() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// =============================================================================
// MultiDialectBuilder for Stablehlo
// =============================================================================

// Recursive class specialized for StablehloBuilder referred to as
// stablehlo.
template <class... Ts>
struct MultiDialectBuilder<StablehloBuilder, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), stablehlo(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), stablehlo(db) {}
  StablehloBuilder stablehlo;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForStableHlo, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), stableHloIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), stableHloIE(db) {}
  IndexExprBuilderForStableHlo stableHloIE;
};

} // namespace onnx_mlir
