/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Loop.cpp - ONNX Operations -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Loop operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

/// Infer the output shape of the ONNXLoopOp.
LogicalResult ONNXLoopOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto builder = Builder(getContext());
  auto &loopBody = getRegion();
  assert(loopBody.getNumArguments() >= 2 &&
         "Loop body must take at least 2 inputs.");

  // We proceed to set types for loop body function inputs.
  // Set type for iteration number (trip count):
  loopBody.getArgument(0).setType(
      RankedTensorType::get({}, builder.getI64Type()));
  // Set type for termination condition:
  loopBody.getArgument(1).setType(
      RankedTensorType::get({}, builder.getI1Type()));

  // Set types for loop carried dependencies (i.e., set these loop carried
  // dependencies that appear in the body function input signature to have the
  // same type as their counterpart in LoopOp inputs).
  auto bodyInputs = loopBody.getArguments();
  auto bodyVRange = llvm::make_range(bodyInputs.begin() + 2, bodyInputs.end());
  for (auto opVToBodyVTy : llvm::zip(getVInitial(), bodyVRange)) {
    auto opVTy = std::get<0>(opVToBodyVTy).getType();
    std::get<1>(opVToBodyVTy).setType(opVTy);
  }

  // Now we have modified loop body function input signatures according to
  // the knowledge we have on the inputs we pass to this function. Dispatch
  // shape inference to obtain body function output types.
  doShapeInference(loopBody);

  // Output loop variables should have the same type as their input
  // counterparts.
  auto bodyResultTys = loopBody.back().getTerminator()->getOperandTypes();
  // Compute the type range corresponding to the final values of
  // loop-carried dependencies/scan outputs in the body function output
  // types.
  auto scanStartItr =
      std::next(bodyResultTys.begin(), 1 + getVInitial().size());
  auto bodyResVFinalTys =
      llvm::make_range(std::next(bodyResultTys.begin(), 1), scanStartItr);
  auto bodyResScanTys = llvm::make_range(scanStartItr, bodyResultTys.end());

  // Set shape for loop operation outputs corresponding to the final
  // values of loop-carried dependencies to be shape of their counterparts
  // in the body function output.
  for (auto vFinalValToTy : llvm::zip(v_final(), bodyResVFinalTys)) {
    std::get<0>(vFinalValToTy).setType(std::get<1>(vFinalValToTy));
  }

  // For scan outputs, we set their shape to be the shape of the return
  // values of the loop body function corresponding to scan outputs, but
  // with an extra leading dimension.
  for (auto vScanOutputValToTy : llvm::zip(scan_outputs(), bodyResScanTys)) {
    auto rankedScanTy =
        std::get<1>(vScanOutputValToTy).cast<RankedTensorType>();
    auto shape = rankedScanTy.getShape();
    SmallVector<int64_t, 4> unsqueezedShape(shape.begin(), shape.end());
    // Note that we may know the extent of the scan output leading
    // dimension, which is very likely just the trip count specified as an
    // input to Loop operation, but we need to eliminate the possibility of
    // early termination to be sure.
    unsqueezedShape.insert(unsqueezedShape.begin(), ShapedType::kDynamic);
    updateType(std::get<0>(vScanOutputValToTy), unsqueezedShape,
        rankedScanTy.getElementType(), /*encoding=*/nullptr,
        /*refineShape=*/false);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Helper function to obtain subset of op results corresponding to the final
// value of loop carried dependencies.
Operation::result_range ONNXLoopOp::v_final() {
  auto results = getResults();
  return llvm::make_range(
      results.begin(), results.begin() + getVInitial().size());
}

// Helper function to obtain subset of op results corresponding to the scan
// outputs.
Operation::result_range ONNXLoopOp::scan_outputs() {
  auto results = getResults();
  return llvm::make_range(
      results.begin() + getVInitial().size(), results.end());
}
