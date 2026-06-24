/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpFusionHelper.hpp - ZHigh Fusion Helper Functions ----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// Shared helpers for recognising and parameterising fused ZHigh operation
// patterns.  Functions here are used by both the pattern-creation pass
// (FusionOpStickUnstick) and the lowering pass (ZHighToZLow) so that the
// pattern-recognition logic lives in exactly one place.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H
#define ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Extended Layout Transform fusion
//
// Fused sequence:
//   ONNXLayoutTransformOp          (ZTensor -> CPU format)
//   ONNXReshapeOp      (optional)  (split one dim into two)
//   ONNXTransposeOp    (optional)  (permute; last dim must stay in place)
//   ONNXReshapeOp      (optional)  (merge two dims back into one)
//   ONNXLayoutTransformOp          (optional, CPU -> ZTensor)
//     OR ZHighDLF16ToF32Op         (optional, DLF16 -> F32 conversion)
//===----------------------------------------------------------------------===//

/// Parameters extracted from / matched against an extended layout transform
/// chain.  Shared between pattern creation and lowering.
struct ExtLayoutTransformFusionParams {
  int64_t reshapeSplitAxis = -1;
  int64_t reshapeSplitFactor = 1;
  std::optional<mlir::ArrayAttr> transposePattern;
  int64_t reshapeMergeAxis = -1;
  bool dlf16ToF32 = false;
  std::optional<mlir::StringAttr> finalLayout;
};

/// Everything the pattern-creation caller needs after a successful match.
struct ExtLayoutTransformChain {
  mlir::Value finalResult;                        ///< last value before yield
  llvm::SmallVector<mlir::Operation *> ops;       ///< chain ops, in order
  ExtLayoutTransformFusionParams params;
};

/// Locate and parameterize an extended layout transform chain.
///
/// Starting from \p startOp (an ONNXLayoutTransformOp), traverses the chain
/// via single-use edges, validates ZTensor / static-dim / layout / beneficial
/// constraints, and extracts the parameters.
///
/// Works identically in two contexts:
///  - Pre-fusion (FusedPatternsForExtendedLayoutTransform): \p startOp is the
///    anchor in the live IR graph.
///  - Post-fusion lowering (ZHighToZLow): \p startOp is the first op inside a
///    FusedOp body.  The same single-use traversal applies because each body op
///    feeds the next one exactly once; the YieldOp terminator is not an
///    ONNXLayoutTransformOp / ONNXReshapeOp / etc., so traversal stops
///    naturally at the last real op.
///
/// Returns failure() without side effects if no valid/beneficial chain is found.
///
/// \p dimAnalysis is optional.  When provided (pattern-creation pass), dim
/// comparisons inside reshape detection use DimAnalysis::sameDim for full
/// precision.  When null (lowering pass), shape-value comparison is used as a
/// conservative approximation.
mlir::FailureOr<ExtLayoutTransformChain>
locateExtLayoutTransformFusion(mlir::ONNXLayoutTransformOp startOp,
    const DimAnalysis *dimAnalysis = nullptr);

} // namespace zhigh
} // namespace onnx_mlir

#endif // ONNX_MLIR_ZHIGH_OP_FUSION_HELPER_H
