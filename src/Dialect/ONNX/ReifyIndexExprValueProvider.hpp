/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ReifyIndexExprValueProvider.hpp - shape IR for reifyResultShapes
//---===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Emits shape dialect operations when extracting runtime shape values during
// ReifyRankedShapedTypeOpInterface::reifyResultShapes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_REIFY_INDEX_EXPR_VALUE_PROVIDER_H
#define ONNX_MLIR_REIFY_INDEX_EXPR_VALUE_PROVIDER_H

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprValueProvider.hpp"

namespace onnx_mlir {

struct ReifyIndexExprValueProvider : IndexExprValueProvider {
  explicit ReifyIndexExprValueProvider(const DialectBuilder &db) : db(db) {}

  mlir::ElementsAttr getConst(mlir::Value value) override;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) override;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) override;

private:
  const DialectBuilder &db;
};

} // namespace onnx_mlir
#endif
