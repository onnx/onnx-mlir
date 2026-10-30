/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------------ DialectBuilder.hpp - Linalg dialect builder ------------===//
//
// Copyright 2022-2026
//
// =============================================================================
//
// This file contains dialect builder for Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_LINALG_H
#define ONNX_MLIR_DIALECT_BUILDER_LINALG_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// IndexExpr Builder
// =============================================================================

struct IndexExprBuilderForTensor : public IndexExprBuilder {
  IndexExprBuilderForTensor(mlir::OpBuilder &builder, mlir::Location loc)
      : IndexExprBuilder(builder, loc) {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value array, uint64_t index) final;
  mlir::Value getShapeVal(mlir::Value tensor, uint64_t index) final;
};
} // namespace onnx_mlir

#endif // ONNX_MLIR_DIALECT_BUILDER_LINALG_H
