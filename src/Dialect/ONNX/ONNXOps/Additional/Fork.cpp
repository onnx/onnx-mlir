/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Fork.cpp - ONNX Operations -------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Fork operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

template <>
LogicalResult ONNXForkOpShapeHelper::computeShape() {
  ONNXForkOp forkOp = llvm::cast<ONNXForkOp>(op);
  (void)forkOp.inferShapes([](Region &region) {});
  Operation *yieldOp = forkOp.getBody().front().getTerminator();
  for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
    DimsExpr outputDims;
    Value returnVal = yieldOp->getOperands()[i];
    int64_t outRank = returnVal.getType().cast<ShapedType>().getRank();
    for (int64_t j = 0; j < outRank; ++j)
      outputDims.emplace_back(createIE->getShapeAsDim(returnVal, j));
    setOutputDims(outputDims, i);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type Inference
//===----------------------------------------------------------------------===//

std::vector<Type> ONNXForkOp::resultTypeInference() {
  Operation *terminator = getRegion().back().getTerminator();
  auto bodyOutputTys = terminator->getOperandTypes();
  std::vector<Type> resultTypes;
  for (auto [i, ty] : llvm::enumerate(bodyOutputTys)) {
    resultTypes.push_back(ty);
  }
  return resultTypes;
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXForkOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  doShapeInference(getRegion());
  for (auto [i, ty] : llvm::enumerate(resultTypeInference()))
    getResult(i).setType(ty);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder: Refer to Async ExecuteOp
//===----------------------------------------------------------------------===//
void ONNXForkOp::build(OpBuilder &builder, OperationState &result,
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
