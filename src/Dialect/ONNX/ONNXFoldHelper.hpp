#pragma once

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

//===----------------------------------------------------------------------===//
// Instructions to add a constant operation.
//===----------------------------------------------------------------------===//
// There is currently support for adding constant propagation for unary and
// binary athythmetic ops (binary ops support broadcast). To add an operation,
// you simply have to add a templated method on how to compute the result in
// terms of one or two inputs. Values comes as Attribtues, and return is also an
// Attribute. In that function, presumably you will need different methods to
// handle int / float / strings... Note that these methods cannot fail. It is
// your responsablitity to tests for which data type are supported in the rules
// directly. Specific type restrictions can be added in the DRR files.

// The methods are:
//
// ComputeConstPropElementwiseBinary and ComputeConstPropElementwiseUnary
// and they need to be tempalted wtih an ONNX Operation (presuably).
//
// Then you need to add rules on how to transform the patterns; look into
// ConstProp.td for example.
//

template <typename OP>
Attribute ComputeConstPropElementwiseBinary(Builder &builder, Type elementType,
    Attribute lhsAttr, Attribute secondAttr) {
  llvm_unreachable("unkonwn operation");
}

// Recursively process one dimension in the rank of the two references. There
// can be one of 3 cases.
// 1) We have fully defined accesses for both operands, launch the computations.
// 2) One of the two has a higher number of unprocessed ranks, which is hte case
// when we have to broadcast the whole lower-dim reference with respect to the
// other. Iterate over each value of the higher ranked reference, keeping the
// reference of the lower ranked reference constant.
// 3) Both references have the same rank, we still do broadcast if one of the
// dimension size is equal to 1.

template <typename ElementwiseBinaryOp>
void RecurseConstPropElementwiseBinary(Builder &builder,
    std::vector<Attribute> &resVector, DenseElementsAttr lhsAttr,
    DenseElementsAttr rhsAttr, SmallVector<uint64_t, 4> &lhsIndices,
    SmallVector<uint64_t, 4> &rhsIndices, int lhsFreeRank, int rhsFreeRank) {
  if (lhsFreeRank == 0) {
    // Fully defined ranks.
    assert(
        rhsFreeRank == 0 && "expect both to recurse to zero at the same time");
    auto lhsElementAttr = lhsAttr.getValue(ArrayRef<uint64_t>(lhsIndices));
    auto rhsElementAttr = rhsAttr.getValue(ArrayRef<uint64_t>(rhsIndices));
    auto elementaryType = lhsAttr.getType().getElementType();
    auto res = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp>(
        builder, elementaryType, lhsElementAttr, rhsElementAttr);
    resVector.emplace_back(res);
  } else if (lhsFreeRank > rhsFreeRank) {
    // Initial broadcast from lhs.
    auto lhsShape = lhsAttr.getType().getShape();
    int lhsRank = lhsShape.size();
    int lhsIndex = lhsRank - lhsFreeRank;
    int lhsSize = lhsAttr.getType().getShape()[lhsIndex];
    for (int i = 0; i < lhsSize; ++i) {
      lhsIndices[lhsIndex] = i;
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(builder, resVector,
          lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank - 1,
          rhsFreeRank);
    }
  } else if (lhsFreeRank < rhsFreeRank) {
    // Initial broadcast from rhs.
    auto rhsShape = rhsAttr.getType().getShape();
    int rhsRank = rhsShape.size();
    int rhsIndex = rhsRank - rhsFreeRank;
    int rhsSize = rhsAttr.getType().getShape()[rhsIndex];
    for (int i = 0; i < rhsSize; ++i) {
      rhsIndices[rhsIndex] = i;
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(builder, resVector,
          lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank,
          rhsFreeRank - 1);
    }
  } else {
    // No initial broadcast, but if one element has size 1 and the other is
    // greater than one, then we also have broadcast.
    auto lhsShape = lhsAttr.getType().getShape();
    int lhsRank = lhsShape.size();
    int lhsIndex = lhsRank - lhsFreeRank;
    int lhsSize = lhsAttr.getType().getShape()[lhsIndex];
    auto rhsShape = rhsAttr.getType().getShape();
    int rhsRank = rhsShape.size();
    int rhsIndex = rhsRank - rhsFreeRank;
    int rhsSize = rhsAttr.getType().getShape()[rhsIndex];
    assert((lhsSize == 1 || rhsSize == 1 || lhsSize == rhsSize) &&
           "incompatible sizes");
    int size = std::max(lhsSize, rhsSize);
    lhsIndices[lhsIndex] = rhsIndices[rhsIndex] = 0;
    for (int i = 0; i < size; ++i) {
      if (lhsSize > 1)
        lhsIndices[lhsIndex] = i;
      if (rhsSize > 1)
        rhsIndices[rhsIndex] = i;
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(builder, resVector,
          lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank - 1,
          rhsFreeRank - 1);
    }
  }
}

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseBinaryOp>
DenseElementsAttr ConstPropElementwiseBinary(
    Builder &builder, Value resOperand, Attribute lhsAttr, Attribute rhsAttr) {
  DenseElementsAttr lhsDenseAttr =
      lhsAttr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr rhsDenseAttr =
      rhsAttr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert((lhsDenseAttr && lhsDenseAttr) && "expected dense attributes");
  assert(
      resOperand.getType().isa<RankedTensorType>() && "expected ranked tensor");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto lhsRank = lhsDenseAttr.getType().getShape().size();
  auto rhsRank = rhsDenseAttr.getType().getShape().size();
  SmallVector<uint64_t, 4> lhsIndices(lhsRank, 0);
  SmallVector<uint64_t, 4> rhsIndices(rhsRank, 0);
  std::vector<Attribute> resVector;
  RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(builder, resVector,
      lhsDenseAttr, rhsDenseAttr, lhsIndices, rhsIndices, lhsRank, rhsRank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}
template <>
Attribute ComputeConstPropElementwiseBinary<ONNXAddOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr);
template <>
Attribute ComputeConstPropElementwiseBinary<ONNXSubOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr);
template <>
Attribute ComputeConstPropElementwiseBinary<ONNXMulOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr);
template <>
Attribute ComputeConstPropElementwiseBinary<ONNXDivOp>(Builder &builder,
    Type elementType, Attribute lhsAttr, Attribute secondAttr);

template <typename OP>
Attribute ComputeConstPropElementwiseUnary(
    Builder &builder, Type elementType, Attribute attr) {
  llvm_unreachable("unkonwn operation");
}

template <typename ElementwiseUnaryOp>
static void RecurseConstPropElementwiseUnary(Builder &builder,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto elementAttr = attr.getValue(ArrayRef<uint64_t>(indices));
    auto elementaryType = attr.getType().getElementType();
    auto res = ComputeConstPropElementwiseUnary<ElementwiseUnaryOp>(
        builder, elementaryType, elementAttr);
    resVector.emplace_back(res);
  } else {
    // Recurse.
    auto shape = attr.getType().getShape();
    int rank = shape.size();
    int index = rank - freeRank;
    int size = attr.getType().getShape()[index];
    for (int i = 0; i < size; ++i) {
      indices[index] = i;
      RecurseConstPropElementwiseUnary<ElementwiseUnaryOp>(
          builder, resVector, attr, indices, freeRank - 1);
    }
  }
}

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseUnaryOp>
DenseElementsAttr ConstPropElementwiseUnary(
    Builder &builder, Value resOperand, Attribute attr) {
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  assert(
      resOperand.getType().isa<RankedTensorType>() && "expected ranked tensor");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto rank = denseAttr.getType().getShape().size();
  SmallVector<uint64_t, 4> indices(rank, 0);
  std::vector<Attribute> resVector;
  RecurseConstPropElementwiseUnary<ElementwiseUnaryOp>(
      builder, resVector, denseAttr, indices, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}
template <>
Attribute ComputeConstPropElementwiseUnary<ONNXNegOp>(
    Builder &builder, Type elementType, Attribute attr);
template <>
Attribute ComputeConstPropElementwiseUnary<ONNXSqrtOp>(
    Builder &builder, Type elementType, Attribute attr);

DenseElementsAttr ConstPropTranspose(
    Builder &builder, Value resOperand, Attribute attr, ArrayAttr permAttr);
DenseElementsAttr ConstPropUnsqueeze(
    Builder &builder, Value resOperand, Attribute attr);
DenseElementsAttr ConstPropSqueeze(
    Builder &builder, Value resOperand, Attribute attr);
DenseElementsAttr ConstPropConcat(
    Builder &builder, Value resOperand, ArrayRef<Attribute> attrs);
DenseElementsAttr ConstPropSlice(Builder &builder, Value resOperand,
    Attribute data, Attribute starts, Attribute ends, Attribute axes,
    Attribute steps);