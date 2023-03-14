/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Scan.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Scan operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXScanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto &loopBody = getRegion();
  assert(!getScanInputAxes().has_value());

  // We proceed to set types for loop body function inputs.
  // Set types for loop carried dependencies (i.e., set these loop carried
  // dependencies that appear in the body function input signature to have
  // the same type as their counterpart in LoopOp inputs).
  auto bodyInputs = loopBody.getArguments();
  auto bodyVRange = llvm::make_range(bodyInputs.begin(), bodyInputs.end());
  for (auto opVToBodyVTy : llvm::zip(getVInitial(), bodyVRange)) {
    auto opVTy = std::get<0>(opVToBodyVTy).getType();
    std::get<1>(opVToBodyVTy).setType(opVTy);
  }

  auto bodyScanInputs = llvm::make_range(
      bodyInputs.begin() + getVInitial().size(), bodyInputs.end());
  for (auto vScanOutputValToTy : llvm::zip(scan_inputs(), bodyScanInputs)) {
    auto rankedScanTy =
        std::get<0>(vScanOutputValToTy).getType().cast<RankedTensorType>();
    auto shape = rankedScanTy.getShape();
    SmallVector<int64_t, 4> squeezedShape(shape.begin() + 1, shape.end());

    // Note that we may know the extent of the scan output leading
    // dimension, which is very likely just the trip count specified as an
    // input to Loop operation, but we need to eliminate the possibility of
    // early termination to be sure.
    updateType(std::get<1>(vScanOutputValToTy), squeezedShape,
        rankedScanTy.getElementType(), /*encoding=*/nullptr,
        /*refineShape=*/false);
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
  auto scanStartItr = std::next(bodyResultTys.begin(), getVInitial().size());
  auto bodyResVFinalTys = llvm::make_range(bodyResultTys.begin(), scanStartItr);
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
    auto scanExtent =
        scan_inputs().front().getType().cast<ShapedType>().getDimSize(0);
    unsqueezedShape.insert(unsqueezedShape.begin(), scanExtent);
    updateType(std::get<0>(vScanOutputValToTy), unsqueezedShape,
        rankedScanTy.getElementType(), /*encoding=*/nullptr,
        /*refineShape=*/false);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

Operation::operand_range ONNXScanOp::getVInitial() {
  auto numVInit = getInitialStateAndScanInputs().size() - getNumScanInputs();
  auto operands = getOperands();
  return llvm::make_range(operands.begin(), operands.begin() + numVInit);
}

Operation::operand_range ONNXScanOp::scan_inputs() {
  auto numVInit = getInitialStateAndScanInputs().size() - getNumScanInputs();
  auto operands = getOperands();
  return llvm::make_range(operands.begin() + numVInit, operands.end());
}

// Helper function to obtain subset of op results corresponding to the final
// value of loop carried dependencies.
Operation::result_range ONNXScanOp::v_final() {
  auto results = getResults();
  return llvm::make_range(
      results.begin(), results.begin() + getVInitial().size());
}

// Helper function to obtain subset of op results corresponding to the scan
// outputs.
Operation::result_range ONNXScanOp::scan_outputs() {
  auto results = getResults();
  return llvm::make_range(
      results.begin() + getVInitial().size(), results.end());
}
