/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- ONNXOpsHelper.hpp.inc - Helper functions for ONNX dialects -----===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

/// This function returns a location with the corresponding ONNX operator name
/// inside. This is useful when tracing what expanded MLIR instructions
/// correspond to what ONNX operator.
///
template <typename OP_TYPE>
mlir::Location ONNXLoc(mlir::Operation *op) {
  using namespace mlir;
  return NameLoc::get(
      StringAttr::get(op->getContext(), OP_TYPE::getOperationName()),
      op->getLoc());
}

inline bool isNoneValue(mlir::Value value) {
  return llvm::isa<mlir::NoneType>(value.getType());
}

/// Check the defining operation of a value.
template <typename OP>
bool definedBy(mlir::Value v) {
  using namespace mlir;
  return !v.isa<BlockArgument>() && isa<OP>(v.getDefiningOp());
}

// Support for recognizing patterns. Detects if the operation "op" has an input
// operand number "matchThisOperandIndex" that is defined by an operation of
// type "OP". If that is the case, "matchOperand" will be set to that operand,
// and "matchOp" will be set to that op. For unary operations; write matches
// only on success.
//
// This call is formatted so that it mimic the operations that we are trying to
// match. For example:
//
// %norm = "onnx.Div"(%d, %stdDev)
// %normScaled = "onnx.Mul"(%norm, %scale)
//
// We can have a match like this (version below is for binary op):
//
//  if (!operandOfOpDefinedBy<ONNXDivOp>(mulOp, norm, scale, divOp, 0))
//
// Namely, test if the mul op has input operands 0 that is defined by a divide
// op. If it does, then set norm, scale, and divOp to their appropriate values.

template <typename OP>
bool operandOfOpDefinedBy(mlir::Operation *op, mlir::Value &matchOperand,
    mlir::Operation *&matchOp, int64_t matchThisOperandIndex) {
  using namespace mlir;
  assert(matchThisOperandIndex >= 0 &&
         matchThisOperandIndex < op->getNumOperands() &&
         "bad match operand index");
  Value operand = op->getOperand(matchThisOperandIndex);
  // operand.dump();
  //  Check for a match with definition of operand.
  if (!operand.isa<BlockArgument>() && isa<OP>(operand.getDefiningOp())) {
    matchOperand = operand;
    matchOp = operand.getDefiningOp();
    return true;
  }
  return false;
}

// Similar as above, for binary operation; write matches only on success.
template <typename OP>
bool operandOfOpDefinedBy(mlir::Operation *op, mlir::Value &matchOperand0,
    mlir::Value &matchOperand1, mlir::Operation *&matchOp,
    int64_t matchThisOperandIndex) {
  using namespace mlir;
  Value dummy;
  if (operandOfOpDefinedBy<OP>(op, dummy, matchOp, matchThisOperandIndex)) {
    matchOperand0 = op->getOperand(0);
    matchOperand1 = op->getOperand(1);
    return true;
  }
  return false;
}
