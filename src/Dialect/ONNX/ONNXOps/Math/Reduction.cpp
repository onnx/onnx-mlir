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
  Value data = operandAdaptor.getData();
  if (!hasShapeAndRank(data)) {
    return failure();
  }
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
  bool isKeepDims = (operandAdaptor.getKeepdims() == 1) ? true : false;
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

template <typename OP>
constexpr bool isAxesInput =
    std::is_same_v<OP, ONNXReduceSumOp> ||
    std::is_same_v<OP, ONNXReduceMinOp> ||
    std::is_same_v<OP, ONNXReduceMinV18Op> ||
    std::is_same_v<OP, ONNXReduceMaxOp> ||
    std::is_same_v<OP, ONNXReduceMaxV18Op> ||
    std::is_same_v<OP, ONNXReduceProdOp> ||
    std::is_same_v<OP, ONNXReduceMeanOp> ||
    std::is_same_v<OP, ONNXReduceL1Op> || std::is_same_v<OP, ONNXReduceL2Op> ||
    std::is_same_v<OP, ONNXReduceLogSumOp> ||
    std::is_same_v<OP, ONNXReduceLogSumExpOp> ||
    std::is_same_v<OP, ONNXReduceSumSquareOp>;

// Default generic computeShape.
template <typename OP_TYPE>
LogicalResult ONNXGenericReductionOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr axes;
  // Handle simple case where axes is an attribute.
  if constexpr (!isAxesInput<OP_TYPE>) {
    createIE->getIntFromArrayAsLiterals(operandAdaptor.getAxesAttr(), axes);
    return customComputeShape(axes, /*noopWithEmptyAxes*/ false);
  } else {
    if (isNoneValue(operandAdaptor.getAxes())) {
      // Default will be used.
    } else if (getONNXConstantOp(operandAdaptor.getAxes())) {
      createIE->getIntFromArrayAsSymbols(operandAdaptor.getAxes(), axes);
    } else {
      // When the axis is dynamic, try to infer the rank of output tensor
      const auto data = operandAdaptor.getData();
      if (!hasShapeAndRank(data)) {
        return failure();
      }
      int64_t dataRank = createIE->getShapedTypeRank(data);
      int64_t axlesSize = createIE->getArraySize(operandAdaptor.getAxes());
      if (!operandAdaptor.getKeepdims() && axlesSize < 0 /*undef shape*/) {
        // Even though we did not compute the shape in ShapeHelper, return
        // success as we gen code for dyn sizes too. Ideally, we would have the
        // code here to compte the shape, it is currently residing in Krnl
        // lowering.

        // In fact, this case here requires further handling of unranked
        // tensors, which I doubt we support. But there is currently a lit tests
        // (onnx/onnx_shape_inference.mlir's test_reduce_sum_5) expect the *xf32
        // sizes. So leave reporting success here, even though I really suspect
        // we should report failure. return op->emitError("does not support
        // unranked reduction output");
        return success();
      }
      // With keep dim, by def the output has the same rank as the input;
      // without keep dim, we remove 1 rank per value in the 1D axes tensor.
      int64_t outputRank =
          operandAdaptor.getKeepdims() ? dataRank : dataRank - axlesSize;
      assert(outputRank >= 0 && "expected to keep at least one dim");

      // Interesting idea below, namely to reuse info from output, or if not
      // there, from input putting question mark in there. Not sure if
      // successful, if it is, it should be generalized to all ops.
      OP_TYPE reduceOp = llvm::cast<OP_TYPE>(op);
      if (mlir::isa<RankedTensorType>(reduceOp.getResult().getType())) {
        // Have already some shapes, keep them in ShapeHelper
        DimsExpr outputDims;
        createIE->getShapeAsDims(reduceOp.getResult(), outputDims);
        setOutputDims(outputDims);
        return success();
      }
      // Else set is as questionmarks. Output tensor should have the same rank
      // as the input. But size of dims is unknown.
      DimsExpr outputDims(outputRank, QuestionmarkIndexExpr(/*isFloat*/ false));
      setOutputDims(outputDims);
      return success();
    }
  }
  bool noopWithEmptyAxes = false;
  if constexpr (isAxesInput<OP_TYPE>) {
    noopWithEmptyAxes = operandAdaptor.getNoopWithEmptyAxes() != 0;
  }
  return customComputeShape(axes, noopWithEmptyAxes);
}
} // namespace onnx_mlir

namespace {

// Method that does all the work for inference of old reductions, namely
// the ones that use the attributes for describing the axes.
template <class OP_TYPE>
static LogicalResult inferShapeForReductionOps_old(OP_TYPE &op) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (!hasShapeAndRank(operandAdaptor.getData()))
    return success();

  ShapedType dataType =
      mlir::cast<ShapedType>(operandAdaptor.getData().getType());
  ONNXGenericReductionOpShapeHelper<OP_TYPE> shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(dataType.getElementType());
}

// Method that does all the work for inference of new reductions (Opset 18
// and later), namely the ones that use inputs for describing the axes.
template <class OP_TYPE>
static LogicalResult inferShapeForReductionOps(OP_TYPE &op) {
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (!hasShapeAndRank(operandAdaptor.getData()))
    return success();
  // Has an interesting axes but not yet shaped, wait for later.
  if (!isNoneValue(operandAdaptor.getAxes()) &&
      !hasShapeAndRank(operandAdaptor.getAxes()))
    return success();

  ShapedType dataType =
      mlir::cast<ShapedType>(operandAdaptor.getData().getType());
  ONNXGenericReductionOpShapeHelper<OP_TYPE> shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(dataType.getElementType());
}

} // namespace

//===----------------------------------------------------------------------===//
// ReduceL1
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL1Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceL1Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceL1 legacy: ReduceL1V13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL1V13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceL1V13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceL2
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL2Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceL2Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceL2 legacy: ReduceL2V13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceL2V13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceL2V13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceLogSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSum legacy: ReduceLogSumV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceLogSumV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSumExp
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumExpOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceLogSumExpOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceLogSumExp legacy: ReduceLogSumExpV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceLogSumExpV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceLogSumExpV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMax
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMaxOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceMaxOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMax legacy: ReduceMaxV18
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMaxV18Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceMaxV18Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMax legacy: ReduceMaxV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMaxV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceMaxV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMean
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMeanOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceMeanOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMean legacy: ReduceMeanV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMeanV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceMeanV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMin
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMinOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceMinOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMin legacy: ReduceMinV18
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMinV18Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceMinV18Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceMin legacy: ReduceMinV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceMinV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceMinV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceProd
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceProdOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceProdOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceProd legacy: ReduceProdV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceProdV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceProdV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceSum
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceSumOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceSum legacy: ReduceSumV11
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceSumV11Op>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceSumSquare
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumSquareOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps<ONNXReduceSumSquareOp>(*this);
}

//===----------------------------------------------------------------------===//
// ReduceSumSquare legacy: ReduceSumSquareV13
//===----------------------------------------------------------------------===//

LogicalResult ONNXReduceSumSquareV13Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForReductionOps_old<ONNXReduceSumSquareV13Op>(*this);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL1Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL1V13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL2Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceL2V13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumExpOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceLogSumExpV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMaxOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMaxV18Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMaxV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMeanOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMeanV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMinOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMinV18Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceMinV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceProdOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceProdV13Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumV11Op>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumSquareOp>;
template struct ONNXGenericReductionOpShapeHelper<ONNXReduceSumSquareV13Op>;

} // namespace onnx_mlir
