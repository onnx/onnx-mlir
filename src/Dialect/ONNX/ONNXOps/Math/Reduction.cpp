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

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericReductionOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &axes, int noopWithEmptyAxes) {
  // DimsExpr axes; // hi alex
  // int noopWithEmptyAxes = 0;
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
#if 1
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr axes;
  createIE->getIntFromArrayAsLiterals(operandAdaptor.axesAttr(), axes);
  return customComputeShape(axes, /*noopWithEmptyAxes*/ false);
#else
  return success();
#endif
}

} // namespace onnx_mlir

namespace {

// Get reduction type from literal attributes.
static RankedTensorType getReductionOutputType(
    ShapedType operandTy, Optional<ArrayAttr> axesAttrs, uint64_t keepdims) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs != llvm::None)
    for (auto axisAttr : axesAttrs.value()) {
      int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
      axis = axis >= 0 ? axis : (rank + axis);
      assert(axis >= -rank && axis <= rank - 1);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }
  else
    for (decltype(rank) i = 0; i < rank; ++i)
      axes.emplace_back(i);

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i)
    isReductionAxis.emplace_back(
        (std::find(axes.begin(), axes.end(), i) != axes.end()) ? true : false);

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else
      dims.emplace_back(operandTy.getShape()[i]);
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

// Reduction with axes is from ConstantOp.
// Only ReduceSum call this function now.
static RankedTensorType getReductionOutputType(ShapedType operandTy,
    DenseElementsAttr axesAttrs, uint64_t keepdims,
    uint64_t noop_with_empty_axes) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  if (axesAttrs)
    for (auto element : axesAttrs.getValues<IntegerAttr>()) {
      int64_t axis = element.getInt();
      if (axis < -rank || axis > rank - 1)
        return RankedTensorType();

      axis = axis >= 0 ? axis : (rank + axis);
      if (std::find(axes.begin(), axes.end(), axis) == axes.end())
        axes.emplace_back(axis);
    }

  if (axes.size() == 0 && !noop_with_empty_axes)
    for (decltype(rank) i = 0; i < rank; ++i)
      axes.emplace_back(i);

  // Mark reduction axes.
  SmallVector<bool, 4> isReductionAxis;
  for (decltype(rank) i = 0; i < rank; ++i)
    isReductionAxis.emplace_back(
        (std::find(axes.begin(), axes.end(), i) != axes.end()) ? true : false);

  // KeepDims
  bool isKeepdims = (keepdims == 1) ? true : false;

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        dims.emplace_back(1); // reduction dimension
    } else
      dims.emplace_back(operandTy.getShape()[i]);
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

// Handle shape inference for reduction like operators.
template <class OP, class ADAPTOR>
static LogicalResult inferShapeForReductionOps(OP &op) {
  ADAPTOR operandAdaptor(op);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success(); // cannot infer when the operands shape is not yet known.

  auto operandTy = op.getOperand().getType().template cast<ShapedType>();
  auto resultTy = getReductionOutputType(operandTy, op.axes(), op.keepdims());

  updateType(op.getResult(), getShape(resultTy), resultTy.getElementType());
  return success();
}

template <class OP_TYPE>
static LogicalResult inferShapeForReductionOps_xxx(OP_TYPE &op) {
  #if 1
  typename OP_TYPE::Adaptor operandAdaptor(op);
  if (llvm::any_of(operandAdaptor.getOperands(),
          [](const Value &operand) { return !hasShapeAndRank(operand); }))
    return success(); // cannot infer when the operands shape is not yet known.

  ShapedType dataType =
      operandAdaptor.data().getType().template cast<ShapedType>();
  ONNXGenericReductionOpShapeHelper<OP_TYPE> shapeHelper(op.getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(dataType.getElementType());
  #else
  return success();  // hi alex
  #endif
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
  if (!data().getType().isa<RankedTensorType>())
    return success();

  auto operandTy = data().getType().cast<RankedTensorType>();
  /**
   *    In OpSet 13, axes of ReduceSum is an input, not an attribute.
   *    If the axes is not a constant, the output shape is unknown.
   *    So far, only constant input for axes is handled.
   *    Since other reduction ops still have axes as attributes,
   *    interface of getReductionOutputType is kept.
   *    An array attribute is generated from the constant input
   **/
  DenseElementsAttr constAxes;
  if (isFromNone(axes())) {
    // constAxes should just be NULL
    // Default value will be given in getReductionOutputType
  } else if (getONNXConstantOp(axes())) {
    constAxes = getONNXConstantOp(axes())
                    .valueAttr()
                    .dyn_cast_or_null<DenseElementsAttr>();
    if (!constAxes) {
      return emitError("ReduceSum: expect dense value for axes ");
    }
  } else {
    // When the axis is dynamic, try to infer the rank of output tensor

    // Can not infer when keepdims is false
    if (!keepdims())
      return success();

    if (getResult().getType().isa<RankedTensorType>())
      // Can not improve further
      return success();

    // Output tensor should have the same rank as the input
    // But size of dims is unknown
    auto outputNumDim = operandTy.getShape().size();
    SmallVector<int64_t, 4> dims(outputNumDim, -1);
    getResult().setType(
        RankedTensorType::get(dims, operandTy.getElementType()));
    return success();
  }

  RankedTensorType type = getReductionOutputType(
      operandTy, constAxes, keepdims(), noop_with_empty_axes());
  if (!type)
    return emitError("unknown shape");
  getResult().setType(type);
  return success();
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
