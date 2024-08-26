/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXLayoutHelper.hpp - ONNX Layout Helper -----------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_LAYOUT_HELPER_H
#define ONNX_MLIR_ONNX_LAYOUT_HELPER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

/// Note: Keep these strings in sync with the one in Dialect/ONNX/ONNX.td.
const std::string LAYOUT_NCHW4C = "NCHW4C";
const std::string LAYOUT_KCMN4C4K = "KCMN4C4K";
const std::string LAYOUT_STANDARD = "STANDARD";

} // namespace onnx_mlir
#endif
