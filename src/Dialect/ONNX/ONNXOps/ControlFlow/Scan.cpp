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

namespace {
template <typename Range>
Range take_begin(Range range, size_t N) {
  assert(range.size() >= N);
  return llvm::make_range(range.begin(), range.begin() + N);
}

size_t getNumStateVariables(ONNXScanOp *op) {
  return op->getInitialStateAndScanInputs().size() - op->getNumScanInputs();
}
} // namespace

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXScanOp::resultTypeInference() {
  unsigned numStateVariables = getNumStateVariables(this);
  Operation *terminator = getRegion().back().getTerminator();
  assert(terminator->getNumOperands() == getFinalStateAndScanOutputs().size() &&
         "ScanOp outputs count must match body results count");
  auto bodyOuputTys = terminator->getOperandTypes();
  std::vector<Type> resultTypes;
  for (auto [i, ty] : llvm::enumerate(bodyOuputTys)) {
    if (i < numStateVariables) { // state variable
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

LogicalResult ONNXScanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // TODO: Support scan_input/output_axes.
  assert(!getScanInputAxes() && "scan_input_axes are unsupported");
  assert(!getScanOutputAxes() && "scan_output_axes are unsupported");

  assert(!scan_inputs().empty() && "there must be 1 or more scan inputs");
  auto firstScanInputType =
      mlir::cast<ShapedType>(scan_inputs().front().getType());
  // Number of body iterations is the dim size of the scan input sequence axis,
  // which is also the dim size of the scan outputs concat axis.
  int64_t sequence_length = firstScanInputType.hasRank()
                                ? firstScanInputType.getDimSize(0)
                                : ShapedType::kDynamic;

  // ScanOp and body inputs/outputs are state variables followed by scan
  // inputs/outputs.
  unsigned numStateVariables = getNumStateVariables(this);

  auto &body = getRegion();
  assert(body.getNumArguments() == getInitialStateAndScanInputs().size() &&
         "ScanOp inputs count must match body operands count");
  auto bodyInputs = body.getArguments();

  // We proceed to set types for scan body state variables inputs to have
  // the same type as their counterpart in ScanOp inputs.
  // zip() runs through the shortest range which is getVInitial().
  for (auto [opStateVar, bodyStateVar] : llvm::zip(getVInitial(), bodyInputs))
    bodyStateVar.setType(opStateVar.getType());

  auto bodyScanInputs = llvm::drop_begin(bodyInputs, numStateVariables);
  for (auto [opScanInput, bodyScanInput] :
      llvm::zip(scan_inputs(), bodyScanInputs)) {
    if (auto rankedTy =
            mlir::dyn_cast<RankedTensorType>(opScanInput.getType())) {
      ArrayRef<int64_t> squeezedShape(rankedTy.getShape().drop_front(1));
      updateType(getOperation(), bodyScanInput, squeezedShape,
          rankedTy.getElementType(), /*encoding=*/nullptr,
          /*refineShape=*/false);
    }
  }

  // Now we have modified scan body input types according to
  // the knowledge we have on the inputs we pass to the body. Dispatch
  // shape inference to obtain body output types.
  doShapeInference(body);
  Operation *terminator = body.back().getTerminator();
  assert(terminator->getNumOperands() == getFinalStateAndScanOutputs().size() &&
         "ScanOp outputs count must match body results count");
  auto bodyOuputTys = terminator->getOperandTypes();

  // Set state variable ScanOp output types (and shapes) to the types
  // and inferred shapes of their counterparts in the body output.
  // zip() runs through the shortest range which is v_final().
  for (auto [val, ty] : llvm::zip(v_final(), bodyOuputTys))
    val.setType(ty);

  // For scan outputs, we set their shape to be the shape of the return
  // values of the body corresponding to scan outputs, but
  // with an extra leading dimension.
  auto bodyScanOutputTys = llvm::drop_begin(bodyOuputTys, numStateVariables);
  for (auto [opScanOutput, ty] : llvm::zip(scan_outputs(), bodyScanOutputTys)) {
    if (auto rankedTy = mlir::dyn_cast<RankedTensorType>(ty)) {
      SmallVector<int64_t, 4> unsqueezedShape(rankedTy.getShape());
      unsqueezedShape.insert(unsqueezedShape.begin(), sequence_length);
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

Operation::operand_range ONNXScanOp::getVInitial() {
  return take_begin(getInitialStateAndScanInputs(), getNumStateVariables(this));
}

Operation::operand_range ONNXScanOp::scan_inputs() {
  return llvm::drop_begin(
      getInitialStateAndScanInputs(), getNumStateVariables(this));
}

// Helper function to obtain subset of op results corresponding to the final
// value of state variables.
Operation::result_range ONNXScanOp::v_final() {
  return take_begin(getFinalStateAndScanOutputs(), getNumStateVariables(this));
}

// Helper function to obtain subset of op results corresponding to the scan
// outputs.
Operation::result_range ONNXScanOp::scan_outputs() {
  return llvm::drop_begin(
      getFinalStateAndScanOutputs(), getNumStateVariables(this));
}
