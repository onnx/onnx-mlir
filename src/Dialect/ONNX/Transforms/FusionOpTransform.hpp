/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- FusionOpTransform.hpp - Fuse ONNX op chains -----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Generic, non-accelerator-specific ONNXFusedOp formation. This is the
// shared home for every general (CPU) fusion kind's pattern registration --
// not named after any one pattern, the same way ONNXFusionOpHelper.{hpp,cpp}
// is the shared home for the FusionOpKindHelper subclasses themselves.
//
// populateONNXFusionOpPatterns() is called from two places:
//  - FusionOpTransform (this file's own pass), when no accelerator wants to
//    absorb these patterns into its own, later fusion pass -- see
//    src/Compiler/CompilerPasses.cpp's `targetCPU` gate.
//  - FusionOpStickUnstick (src/Accelerators/NNPA/Transform/ZHigh/
//    FusionOpStickUnstick.cpp), which merges these general patterns into its
//    own ZHigh-dialect-level RewritePatternSet on NNPA builds, so there is
//    still only one fusion pass invocation, at the point NNPA already forms
//    fused ops (late, after most optimizations, so they aren't disturbed).
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FUSION_OP_TRANSFORM_H
#define ONNX_MLIR_FUSION_OP_TRANSFORM_H

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

/// Registers every general (non-accelerator-specific) ONNXFusedOp pattern.
/// \p dimAnalysis must be non-null and already analyzed.
void populateONNXFusionOpPatterns(mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *context, DimAnalysis *dimAnalysis);

} // namespace onnx_mlir

#endif // ONNX_MLIR_FUSION_OP_TRANSFORM_H
