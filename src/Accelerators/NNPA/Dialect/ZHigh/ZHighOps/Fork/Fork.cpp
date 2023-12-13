/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Fork.cpp - ZHigh Operations -------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ZHigh dialect Fork operation.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ZHighForkOp::resultTypeInference() {
  Operation *terminator = getRegion().back().getTerminator();
  auto bodyOutputTys = terminator->getOperandTypes();

  //   // assert is checked in verify()
  //   assert(getNumResults() == thenResultTypes.size() &&
  //          getNumResults() == elseResultTypes.size() &&
  //          "if #results and branches #results differ");
  std::vector<Type> resultTypes;
  for (auto [i, ty] : llvm::enumerate(bodyOutputTys)) {
    resultTypes.push_back(ty);
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighForkOpShapeHelper::computeShape() {
  ZHighForkOp forkOp = llvm::dyn_cast<ZHighForkOp>(op);
  ZHighForkOp::Adaptor operandAdaptor(operands);
  // TODO: implement
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighForkOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  doShapeInference(getRegion());
  for (auto [i, ty] : llvm::enumerate(resultTypeInference()))
    getResult(i).setType(ty);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder: Referring to Async ExecuteOp
//===----------------------------------------------------------------------===//
void ZHighForkOp::build(OpBuilder &builder, OperationState &result,
    TypeRange resultTypes, ValueRange operands, BodyBuilderFn bodyBuilder) {

  result.addOperands(operands);
  result.addTypes(resultTypes);

  // Add a body region with block arguments
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value operand : operands) {
    bodyBlock.addArgument(operand.getType(), operand.getLoc());
  }

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (resultTypes.empty() && !bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<ONNXYieldOp>(result.location, ValueRange());
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

} // namespace zhigh
} // namespace onnx_mlir
