/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Reduction.cpp - ONNX Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Reduce operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericReductionOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &axes, int noopWithEmptyAxes) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  Value data = operandAdaptor.data();
  int64_t rank = createIE->getShapedTypeRank(data);
  // Normalize the axes: at present, we only support compile time axes, but
  // with keep_dim on, it might not be too difficult to generate the code.
  SmallVector<int64_t, 4> uniqueAxes;
  if (axes.size() > 0) {
    for (uint64_t i = 0; i < axes.size(); ++i) {
      if (!axes[i].isLiteral())
        return op->emitError("expect compile time constant for reduction axes");
      int64_t axis = axes[i].getLiteral();
      if (axis < -rank || axis > rank - 1)
        return op->emitError("reduction axis is out of bound");
      axis = axis >= 0 ? axis : (rank + axis);
      if (std::find(uniqueAxes.begin(), uniqueAxes.end(), axis) ==
          uniqueAxes.end())
        uniqueAxes.emplace_back(axis);
    }
  } else if (!noopWithEmptyAxes) {
    // Mark all axes as target for reduction.
    for (int64_t axis = 0; axis < rank; ++axis)
      uniqueAxes.emplace_back(axis);
  }

  // Mark reduction axes.
  isReductionAxis.resize(rank);
  for (int64_t i = 0; i < rank; ++i)
    isReductionAxis[i] =
        std::find(uniqueAxes.begin(), uniqueAxes.end(), i) != uniqueAxes.end();

  // Generate the output dims.
  bool isKeepDims = (operandAdaptor.keepdims() == 1) ? true : false;
  DimsExpr outputDims;
  LiteralIndexExpr one(1);
  for (int64_t i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepDims)
        outputDims.emplace_back(one); // reduction dimension
    } else
      outputDims.emplace_back(createIE->getShapeAsDim(data, i));
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

// Default generic computeShape.
template <typename OP_TYPE>
LogicalResult ONNXGenericReductionOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr axes;
  createIE->getIntFromArrayAsLiterals(operandAdaptor.axesAttr(), axes);
  return customComputeShape(axes, /*noopWithEmptyAxes*/ false);
}

// ComputeShape that is specific to ReduceSumOp.
template <>
LogicalResult
ONNXGenericReductionOpShapeHelper<ONNXReduceSumOp>::computeShape() {
  ONNXReduceSumOp reduceOp = llvm::cast<ONNXReduceSumOp>(op);
  ONNXReduceSumOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr axes;
  if (isFromNone(operandAdaptor.axes())) {
    // Default will be used.
  } else if (getONNXConstantOp(operandAdaptor.axes())) {
    createIE->getIntFromArrayAsSymbols(operandAdaptor.axes(), axes);
  } else {
    // When the axis is dynamic, try to infer the rank of output tensor
    int64_t dataRank = createIE->getShapedTypeRank(operandAdaptor.data());
    int64_t axlesSize = createIE->getArraySize(operandAdaptor.axes());
    if (!operandAdaptor.keepdims() && axlesSize < 0 /*undef shape*/) {
      // Even though we did not compute the shape in ShapeHelper, return success
      // as we gen code for dyn sizes too. Ideally, we would have the code here
      // to compte the shape, it is currently residing in Krnl lowering.

      // In fact, this case here requires further handling of unranked tensors,
      // which I doubt we support. But there is currently a lit tests
      // (onnx/onnx_shape_inference.mlir's test_reduce_sum_5) expect the *xf32
      // sizes. So leave reporting success here, even though I really suspect we
      // should report failure. return op->emitError("does not support unranked
      // reduction output");
      return success();
    }
    // With keep dim, by def the output has the same rank as the input; without
    // keep dim, we remove 1 rank per value in the 1D axes tensor.
    int64_t outputRank =
        operandAdaptor.keepdims() ? dataRank : dataRank - axlesSize;
    assert(outputRank >= 0 && "expected to keep at least one dim");

    // Interesting idea below, namely to reuse info from output, or if not
    // there, from input putting question mark in there. Not sure if successful,
    // if it is, it should be generalized to all ops.

    if (reduceOp.getResult().getType().isa<RankedTensorType>()) {
      // Have already some shapes, keep them in ShapeHelper
      DimsExpr outputDims;
      createIE->getShapeAsDims(reduceOp.getResult(), outputDims);
      setOutputDims(outputDims);
      return success();
    }
    // Else set is as questionmarks. Output tensor should have the same rank as
    // the input. But size of dims is unknown.
    DimsExpr outputDims(outputRank, QuestionmarkIndexExpr());
    setOutputDims(outputDims);
    return success();
  }
  bool noopWithEmptyAxes = operandAdaptor.noop_with_empty_axes() != 0;
  return customComputeShape(axes, noopWithEmptyAxes);
}

} // namespace onnx_mlir

namespace {

// Method that does all the work for inference of traditional reductions, namely
// the ones that use the attributes for describing the axes.
template <class OP_TYPE>
static LogicalResult inferShapeForReductionOps_xxx(OP_TYPE &op) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (!hasShapeAndRank(operandAdaptor.data()))
    return success();

  ShapedType dataType =
      operandAdaptor.data().getType().template cast<ShapedType>();
  ONNXGenericReductionOpShapeHelper<OP_TYPE> shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(dataType.getElementType());
}

} // namespace

//===----------------------------------------------------------------------===//
// ReduceL1
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL1Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceL1Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceL2
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL2Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceL2Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceLogSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSumExp
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumExpOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceLogSumExpOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMax
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMean
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMeanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceMeanOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMin
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceProd
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceProdOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceProdOp>(*this);
}
//===----------------------------------------------------------------------===//
// ReduceSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(data()))
    return success();
  // Has an interesting axes but not yet shaped, wait for later.
  if (!isFromNone(axes()) && !hasShapeAndRank(axes()))
    return success();

  ShapedType dataType = data().getType().template cast<ShapedType>();
  ONNXReduceSumOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(dataType.getElementType());
}

//===----------------------------------------------------------------------===//
// ReduceSum legacy: ReduceSumV11
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceSumV11Op>(*this);
}
//===----------------------------------------------------------------------===//
// ReduceSumSquare
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumSquareOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_xxx<ONNXReduceSumSquareOp>(*this);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL1Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL2Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumExpOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMaxOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMeanOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMinOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceProdOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumV11Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumSquareOp>;

} // namespace onnx_mlir
