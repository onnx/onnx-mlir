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
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
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
  //
  // Special case for sequences: the initial value may have a statically known
  // length (e.g., SequenceEmpty → length 0), but the loop body can grow the
  // sequence on each iteration, so the length at any arbitrary iteration is
  // unknown.  Using the initial length would mislead SequenceInsert into
  // thinking the sequence is always empty, producing a wrong output length (1).
  // We therefore conservatively reset any known sequence length to kDynamic
  // for the block argument, so downstream shape inference stays sound.
  auto bodyCarriedInputs = llvm::drop_begin(loopBody.getArguments(), 2);
  for (auto [opInput, bodyInput] :
      llvm::zip(getVInitial(), bodyCarriedInputs)) {
    Type ty = opInput.getType();
    if (auto seqTy = mlir::dyn_cast<SeqType>(ty))
      if (seqTy.getLength() != ShapedType::kDynamic)
        ty = SeqType::get(seqTy.getElementType(), ShapedType::kDynamic);
    bodyInput.setType(ty);
  }

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
  //
  // Three cases in which we can determine a static leading dimension:
  //   M = 0  : body never executes → leading dim is 0, regardless of
  //             condition (ONNX spec: trip count takes priority).
  //   M > 0, NoneType condition : loop always runs exactly M iterations.
  //   M > 0, constant-true condition AND body always yields constant true:
  //             loop is also guaranteed to run M iterations.
  // In all other cases we fall back to kDynamic.
  int64_t leadingDim = ShapedType::kDynamic;
  Value tripCountVal = getM();
  Value condVal = getCond();
  if (auto tripAttr = getElementAttributeFromONNXValue(tripCountVal)) {
    auto staticCount = getScalarValue<int64_t>(
        tripAttr, mlir::cast<ShapedType>(tripCountVal.getType()));
    if (staticCount == 0) {
      // Body never executes regardless of condition: scan outputs are empty.
      leadingDim = 0;
    } else if (staticCount > 0) {
      bool condIsNone = mlir::isa<NoneType>(condVal.getType());
      // Check if the condition is statically always-true: both the initial
      // condition and the body's yielded condition must be constant true.
      bool condIsAlwaysTrue = false;
      if (!condIsNone) {
        // Use APInt iteration to safely read any integer width (including i1).
        auto readBool = [](ElementsAttr attr) -> bool {
          return (*attr.getValues<APInt>().begin()).getBoolValue();
        };
        if (auto condAttr = getElementAttributeFromONNXValue(condVal)) {
          bool initCondTrue = readBool(condAttr);
          // The body terminator's condition operand (operand 0) must also be
          // a constant true so we know there is no early-exit possibility.
          Value bodyYieldCond = terminator->getOperand(0);
          if (initCondTrue) {
            if (auto yieldCondAttr =
                    getElementAttributeFromONNXValue(bodyYieldCond)) {
              condIsAlwaysTrue = readBool(yieldCondAttr);
            }
          }
        }
      }
      if (condIsNone || condIsAlwaysTrue)
        leadingDim = staticCount;
    }
  }

  auto bodyScanOutputTys = llvm::drop_begin(bodyOuputTys, numCarried);
  for (auto [opScanOutput, ty] : llvm::zip(scan_outputs(), bodyScanOutputTys)) {
    // TODO: Handle SeqType, OptType.
    if (auto rankedTy = mlir::dyn_cast<RankedTensorType>(ty)) {
      SmallVector<int64_t, 4> unsqueezedShape(rankedTy.getShape());
      unsqueezedShape.insert(unsqueezedShape.begin(), leadingDim);
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
