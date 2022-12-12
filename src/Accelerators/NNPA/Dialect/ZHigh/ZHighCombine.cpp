/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ZHighCombine.cpp - ZHigh High Level Optimizer ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

bool oneIsOfNHWCLayout(Type t1, Type t2) {
  if (auto rtp1 = llvm::dyn_cast<RankedTensorType>(t1)) {
    if (onnx_mlir::zhigh::getZTensorLayout(rtp1) ==
        onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC)
      return true;
    // t1 is not of NHWC, check t2.
    if (auto rtp2 = llvm::dyn_cast<RankedTensorType>(t2)) {
      return (onnx_mlir::zhigh::getZTensorLayout(rtp2) ==
              onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC);
    }
    // t2 is unranked.
  }
  // t1 is unranked.
  // Unranked type is potentially of NHWC.
  return true;
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ONNXZHighCombine.inc"
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

/// Register optimization patterns as "canonicalization" patterns
/// on the ZHighStickOp.
void ZHighStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<StickUnstickSameLayoutRemovalPattern>(context);
  results.insert<StickUnstickDiffLayoutRemovalPattern>(context);
  results.insert<NoneTypeStickRemovalPattern>(context);
  results.insert<ReplaceONNXLeakyReluPattern>(context);
}

/// Register optimization patterns as "canonicalization" patterns
/// on the ZHighUnstickOp.
void ZHighUnstickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<UnstickStickRemovalPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
