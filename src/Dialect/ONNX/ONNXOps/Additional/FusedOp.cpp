/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ FusedOp.cpp - ONNX FusedOp ------------------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect FusedOp operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/RegionKindInterface.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// RegionKindInterface
//===----------------------------------------------------------------------===//

// Declare the body as a Graph region so that the MLIR erase infrastructure
// (RewriterBase::eraseOp / eraseSingleOp) skips the SSA-dominance use-empty
// assertion when erasing the FusedOp and its body during lowering.  This
// matches the semantics of other ONNX subgraph ops (If, Loop, Scan) and is
// correct because the body is a linear chain with no control flow — declaring
// it Graph means "uses may exist at erase time; dropAllUses() will handle it."
mlir::RegionKind ONNXFusedOp::getRegionKind(unsigned index) {
  return mlir::RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXFusedOp::verify() {
  // Body must terminate with ONNXYieldOp.
  Block &body = getBody().front();
  if (body.empty())
    return emitOpError("body region must not be empty");
  auto yieldOp = dyn_cast<ONNXYieldOp>(body.getTerminator());
  if (!yieldOp)
    return emitOpError("body must terminate with onnx.Yield");

  // Yield operand count must match result count.
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("yield operand count (")
           << yieldOp.getNumOperands()
           << ") must match result count (" << getNumResults() << ")";

  // Block argument count must match inputs count (isolated region: every
  // external tensor is threaded through as a block argument; NoneType values
  // are recreated inside and are not block arguments).
  if (body.getNumArguments() != getInputs().size())
    return emitOpError("body block argument count (")
           << body.getNumArguments()
           << ") must match inputs count (" << getInputs().size() << ")";

  // Each block argument type must match the corresponding input type.
  for (auto [idx, argType, inputType] : llvm::enumerate(
           body.getArgumentTypes(), getInputs().getTypes()))
    if (argType != inputType)
      return emitOpError("body block argument ")
             << idx << " type " << argType
             << " does not match input type " << inputType;

  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

// Shape inference delegates entirely to the inner ops.  The ShapeInferencePass
// runs applyPatternsGreedily top-down, so inner ops are visited and their
// result types are set before this function is called via YieldShapesPattern.
// All we do here is copy the yield operand types to the FusedOp results.
LogicalResult ONNXFusedOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  doShapeInference(getBody());
  auto yieldOp =
      cast<ONNXYieldOp>(getBody().front().getTerminator());
  for (auto [i, operand] : llvm::enumerate(yieldOp.getOperands()))
    getResult(i).setType(operand.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Shape helper
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

// computeShape reads already-inferred result types and sets the output dims
// for each result.  By the time lowering calls this, shape inference has
// already run and all result types carry concrete static shapes.
template <>
LogicalResult ONNXFusedOpShapeHelper::computeShape() {
  ONNXFusedOp fusedOp = cast<ONNXFusedOp>(op);
  for (int64_t n = 0, numResults = fusedOp.getNumResults(); n < numResults;
       ++n) {
    if (failed(setOutputDimsFromTypeWithConstantShape(
            fusedOp.getResult(n).getType(), n)))
      return failure();
  }
  return success();
}

template struct ONNXNonSpecificOpShapeHelper<ONNXFusedOp>;

} // namespace onnx_mlir
