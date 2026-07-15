/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- FusionOpBasePattern.hpp - Generic FusedOp pattern base ------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Non-accelerator infrastructure for building ONNXFusedOp out of a matched
// chain of ops.  Counterpart of FusedOpKindLowering (see
// src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp), which handles the reverse
// direction (lowering a FusedOp back down).
//
// FusedPatternForOpKind<AnchorOpType, FusionT>
//   Template rewrite pattern that matches on AnchorOpType, asks FusionT to
//   detect a fusible chain starting from it, and — if beneficial — replaces
//   the chain with an ONNXFusedOp via FusionT::fuse().  Register one
//   instantiation per (anchor op, fusion kind) pair.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FUSION_OP_BASE_PATTERN_H
#define ONNX_MLIR_FUSION_OP_BASE_PATTERN_H

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/Transforms/FusionOpHelper.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// FusedPatternForOpKind<AnchorOpType, FusionT>
//
// Usage:
//   using MyFusedPattern = FusedPatternForOpKind<MyAnchorOp, MyFusion>;
//   patterns.insert<MyFusedPattern>(context, dimAnalysis);
//===----------------------------------------------------------------------===//

template <typename AnchorOpType, typename FusionT>
class FusedPatternForOpKind : public mlir::OpRewritePattern<AnchorOpType> {
  DimAnalysis *dimAnalysis;

public:
  FusedPatternForOpKind(mlir::MLIRContext *context, DimAnalysis *dimAnalysis)
      : mlir::OpRewritePattern<AnchorOpType>(context, 1),
        dimAnalysis(dimAnalysis) {}

  mlir::LogicalResult matchAndRewrite(
      AnchorOpType anchorOp, mlir::PatternRewriter &rewriter) const override {
    FusionT fusion;
    if (!fusion.detectIfBeneficial(dimAnalysis, anchorOp))
      return mlir::failure();
    fusion.fuse(rewriter, anchorOp.getLoc());
    return mlir::success();
  }
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_FUSION_OP_BASE_PATTERN_H
