/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToAffineHelper.hpp ----------------------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Declare utility functions for the Krnl to Affine dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_TO_AFFINE_H
#define ONNX_MLIR_KRNL_TO_AFFINE_H

#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {
namespace krnl {

/// Compute the normalized trip count of a loop as:
///   trip = min(UB - GI, block);
IndexExpr trip(IndexExpr UB, IndexExpr block, IndexExpr GI);

} // namespace krnl
} // namespace onnx_mlir
#endif
