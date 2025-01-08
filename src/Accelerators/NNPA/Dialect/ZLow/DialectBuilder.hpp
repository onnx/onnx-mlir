/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - ZLow Dialect Builder -----------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the ZLow Dialect Builder.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_H
#define ONNX_MLIR_DIALECT_BUILDER_H

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
// ZLow Builder for building ZLow operations
// =============================================================================

struct ZLowBuilder : public DialectBuilder {
  ZLowBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  ZLowBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  ZLowBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~ZLowBuilder() {}

  void stick(mlir::Value x, mlir::Value out, mlir::StringAttr layout,
      mlir::IntegerAttr saturation) const;

  void quantizedStick(mlir::Value x, mlir::Value xRecScale, mlir::Value xOffset,
      mlir::Value out, mlir::StringAttr layout, mlir::StringAttr qType) const;

  void quantizedMatMul(mlir::Value x, mlir::Value xRecScale,
      mlir::Value xOffset, mlir::Value y, mlir::Value yRecScale,
      mlir::Value yOffset, mlir::Value b, mlir::Value bRecScale,
      mlir::Value bOffset, mlir::Value workArea, mlir::Value shape,
      mlir::Value out, mlir::Value outRecScale, mlir::Value outOffset,
      mlir::StringAttr xQType, mlir::StringAttr yQType, mlir::StringAttr bQType,
      mlir::StringAttr outQType, mlir::IntegerAttr isBcast,
      mlir::IntegerAttr isStacked, mlir::IntegerAttr preComputedBias,
      mlir::IntegerAttr disableClipping,
      mlir::IntegerAttr dequantizeOutput) const;
};

// =============================================================================
// MultiDialectBuilder for ZLow
// =============================================================================

// Recursive class specialized for ZLowBuilder referred to as krnl.
template <class... Ts>
struct MultiDialectBuilder<ZLowBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), zlow(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), zlow(db) {}
  ZLowBuilder zlow;
};

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
#endif
