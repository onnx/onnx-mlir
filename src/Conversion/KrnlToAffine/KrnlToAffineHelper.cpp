
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToAffineHelper.cpp ----------------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Implement utility functions for the Krnl to Affine dialect conversion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "src/Conversion/KrnlToAffine/KrnlToAffineHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

IndexExpr trip(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Trip count in general: min(UB - GI, Block).
  UB.debugPrint("trip UB");
  block.debugPrint("trip block");
  GI.debugPrint("trip GI");
  // IndexExpr nTrip = IndexExpr::min(nUB - nGI, nBlock);
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full, so trip is always full.
    return block;
  }
  return IndexExpr::min(UB - GI, block);
}

} // namespace krnl
} // namespace onnx_mlir