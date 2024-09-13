/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- DialectBuilder.hpp - Stablehlo dialect builder -----------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file contains dialect builder for Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_STABLEHLO_H
#define ONNX_MLIR_DIALECT_BUILDER_STABLEHLO_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

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
  mlir::Value dynamic_reshape(
      mlir::Type type, mlir::Value input, mlir::Value shape) const;
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

//===----------------------------------------------------------------------===//
// Extends OnnxBuilder with member functions that might generate Stablehlo
// related dialect operations.
//===----------------------------------------------------------------------===//

struct OnnxToStablehloBuilder : public OnnxBuilder {
  OnnxToStablehloBuilder(mlir::Location loc) : OnnxBuilder(loc) {}
  OnnxToStablehloBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : OnnxBuilder(b, loc) {}
  OnnxToStablehloBuilder(const DialectBuilder &db) : OnnxBuilder(db) {}
  virtual ~OnnxToStablehloBuilder() {}

  // Generate an 'onnx.reshape' operation on the 'input' tensor, the new shape
  // is provided by 'shapeDims'.
  mlir::Value reshape(const mlir::Value input,
      const llvm::ArrayRef<DimIndexExpr> shapeDims) const;

  // Generate a 'onnx.Transpose' operation on the 'input' tensor given the
  // permutation array 'perm' and the operator output dimensions 'outputDims'.
  mlir::Value transpose(const mlir::Value input,
      const llvm::ArrayRef<int64_t> perm,
      const llvm::ArrayRef<DimIndexExpr> outputDims) const;
};

// =============================================================================
// IndexExpr Builder for Shape lowering
// =============================================================================

struct IndexExprBuilderForStablehlo : IndexExprBuilder {
  IndexExprBuilderForStablehlo(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForStablehlo(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForStablehlo(const DialectBuilder &db)
      : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForStablehlo() {}

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

// Recursive class specialized for OnnxToStablehloBuilder referred to as
// stablehloOnnx.
template <class... Ts>
struct MultiDialectBuilder<OnnxToStablehloBuilder, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), stablehloOnnx(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), stablehloOnnx(db) {}
  OnnxToStablehloBuilder stablehloOnnx;
};

// Recursive class specialized for IndexExprBuilderForStablehlo referred to as
// stableHloIE.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForStablehlo, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), stableHloIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), stableHloIE(db) {}
  IndexExprBuilderForStablehlo stableHloIE;
};

} // namespace onnx_mlir
#endif
