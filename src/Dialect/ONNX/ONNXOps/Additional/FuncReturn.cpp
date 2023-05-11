/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- FuncReturn.cpp - ONNX Operations -------------------===//
//
// This file provides definition of ONNX dialect FuncReturn operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

namespace {
bool shapeIsSameOrMoreSpecific(ShapedType lhs, ShapedType rhs) {
  // Unranked is the least specific:
  if (!rhs.hasRank())
    return true;
  if (!lhs.hasRank())
    return false;

  // Otherwise the shapes are incompatible if ranks are different.
  if (lhs.getRank() != rhs.getRank())
    return false;

  for (auto [lhsDimSize, rhsDimSize] :
      llvm::zip(lhs.getShape(), rhs.getShape())) {
    // Dim size is less specific or incompatible unless they are
    // identical or rhs is dynamic ("?").
    if (!ShapedType::isDynamic(rhsDimSize) && lhsDimSize != rhsDimSize)
      return false;
  }
  return true;
}

bool typeIsSameOrMoreSpecific(Type lhs, Type rhs) {
  ShapedType lhsShaped = dyn_cast<ShapedType>(lhs);
  ShapedType rhsShaped = dyn_cast<ShapedType>(rhs);

  if (!lhsShaped && !rhsShaped) {
    // TODO: Check types are same or lhs is more specific than rhs
    //       when they are not shaped types.
    return true;
  }

  if (!lhsShaped || !rhsShaped) {
    // lhs and rhs must agree on whether they are ShapedType.
    return false;
  }

  return shapeIsSameOrMoreSpecific(lhsShaped, rhsShaped);
}
} // namespace

// Implementation is adapted from mlir/lib/Dialect/Func/IR/FuncOps.cpp
// relaxing the type check to allow more specific shapes.
LogicalResult ONNXFuncReturnOp::verify() {
  auto function = cast<func::FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (!typeIsSameOrMoreSpecific(getOperand(i).getType(), results[i]))
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") is incompatible with function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();

  return success();
}
