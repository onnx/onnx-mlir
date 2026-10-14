/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- IndexExprValueProvider.hpp - runtime value extraction -------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Pluggable source of compile-time constants and runtime index values for
// IndexExprBuilder. Implementations emit dialect-specific IR (e.g. shape::,
// krnl) or return nullptr during analysis.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INDEX_EXPR_VALUE_PROVIDER_H
#define ONNX_MLIR_INDEX_EXPR_VALUE_PROVIDER_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

namespace onnx_mlir {

struct IndexExprValueProvider {
  virtual ~IndexExprValueProvider() = default;

  // Return nullptr if value is not a constant.
  virtual mlir::ElementsAttr getConst(mlir::Value value) = 0;
  // Return nullptr if array element i cannot be obtained.
  virtual mlir::Value getVal(mlir::Value arrayVal, uint64_t i) = 0;
  // Return nullptr if shape dimension i cannot be obtained.
  virtual mlir::Value getShapeVal(
      mlir::Value tensorOrMemrefValue, uint64_t i) = 0;
};

} // namespace onnx_mlir
#endif
