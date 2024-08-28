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
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXLoopOp::resultTypeInference() {
  size_t numLoopCarriedDependencies = getVInitial().size();
  Operation *terminator = getRegion().back().getTerminator();
  assert(terminator->getNumOperands() == 1 + getVFinalAndScanOutputs().size() &&
         "LoopOp outputs count must match body results count");
  // Skip the termination condition.
  auto bodyOuputTys = llvm::drop_begin(terminator->getOperandTypes(), 1);
  std::vector<Type> resultTypes;
  for (auto [i, ty] : llvm::enumerate(bodyOuputTys)) {
    if (i < numLoopCarriedDependencies) { // loop carried dependency
      resultTypes.push_back(ty);
    } else { // scan output
      // Erase any rank and shape. Shape inference will add a leading dimension.
      Type elementType = mlir::cast<ShapedType>(ty).getElementType();
      resultTypes.push_back(UnrankedTensorType::get(elementType));
    }
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

/// Infer the output shape of the ONNXLoopOp.
LogicalResult ONNXLoopOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto builder = Builder(getContext());

  size_t numCarried = getVInitial().size();

  auto &loopBody = getRegion();
  // Body inputs: trip count, termination condition, loop carried dependencies.
  // TODO: Add verifier to check this.
  assert(loopBody.getNumArguments() == 2 + numCarried &&
         "LoopOp inputs count must match body operands count");

  // We proceed to set types for loop body function inputs.
  // Set type for iteration number (trip count):
  loopBody.getArgument(0).setType(
      RankedTensorType::get({}, builder.getI64Type()));
  // Set type for termination condition:
  loopBody.getArgument(1).setType(
      RankedTensorType::get({}, builder.getI1Type()));

  // Set body input types for loop carried dependencies to the types of
  // their LoopOp input counterpart.
  auto bodyCarriedInputs = llvm::drop_begin(loopBody.getArguments(), 2);
  for (auto [opInput, bodyInput] : llvm::zip(getVInitial(), bodyCarriedInputs))
    bodyInput.setType(opInput.getType());

  // Now we have modified loop body input types according to
  // the knowledge we have on the initial inputs. Dispatch
  // shape inference to obtain body output types.
  doShapeInference(loopBody);
  Operation *terminator = loopBody.back().getTerminator();
  assert(terminator->getNumOperands() == 1 + getVFinalAndScanOutputs().size() &&
         "LoopOp outputs count must match body results count");
  // Skip the termination condition.
  auto bodyOuputTys = llvm::drop_begin(terminator->getOperandTypes(), 1);

  // Set loop carried dependency LoopOp output types (and shapes) to the types
  // and inferred shapes of their counterparts in the body output.
  // zip() runs through the shortest range which is v_final().
  for (auto [val, ty] : llvm::zip(v_final(), bodyOuputTys))
    val.setType(ty);

  // For scan outputs, we set their shape to be the shape of the return
  // values of the loop body function corresponding to scan outputs, but
  // with an extra leading dimension.
  auto bodyScanOutputTys = llvm::drop_begin(bodyOuputTys, numCarried);
  for (auto [opScanOutput, ty] : llvm::zip(scan_outputs(), bodyScanOutputTys)) {
    // TODO: Handle SeqType, OptType.
    if (auto rankedTy = mlir::dyn_cast<RankedTensorType>(ty)) {
      SmallVector<int64_t, 4> unsqueezedShape(rankedTy.getShape());
      // Note that we may know the extent of the scan output leading
      // dimension, which is very likely just the trip count specified as an
      // input to Loop operation, but we need to eliminate the possibility of
      // early termination to be sure.
      unsqueezedShape.insert(unsqueezedShape.begin(), ShapedType::kDynamic);
      updateType(getOperation(), opScanOutput, unsqueezedShape,
          rankedTy.getElementType(), /*encoding=*/nullptr,
          /*refineShape=*/false);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Helper function to obtain subset of op results corresponding to the final
// value of loop carried dependencies.
Operation::result_range ONNXLoopOp::v_final() {
  size_t numCarried = getVInitial().size();
  auto outputs = getVFinalAndScanOutputs();
  return llvm::make_range(outputs.begin(), outputs.begin() + numCarried);
}

// Helper function to obtain subset of op results corresponding to the scan
// outputs.
Operation::result_range ONNXLoopOp::scan_outputs() {
  size_t numCarried = getVInitial().size();
  return llvm::drop_begin(getVFinalAndScanOutputs(), numCarried);
}
