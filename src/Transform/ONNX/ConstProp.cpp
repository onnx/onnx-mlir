/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the constpropd operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include <math.h>

using namespace mlir;

namespace {

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

/// A helper function to contruct a RankedTensorType from a ShapedType.
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation, also as an
// attribute.

template <typename OP>
Attribute ComputeConstPropElementwiseBinary(PatternRewriter &rewriter,
    Type elementType, Attribute lhsAttr, Attribute secondAttr) {
  llvm_unreachable("unkonwn operation");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXAddOp>(
    PatternRewriter &rewriter, Type elementType, Attribute lhsAttr,
    Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    double lhsVal = lhsAttr.cast<FloatAttr>().getValueAsDouble();
    double rhsVal = secondAttr.cast<FloatAttr>().getValueAsDouble();
    double res = lhsVal + rhsVal;
    // Could use the APFloat interface to emulate the results, are ok to simply
    // perform them in the highest possible precision.
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t lhsVal = lhsAttr.cast<IntegerAttr>().getInt();
    uint64_t rhsVal = secondAttr.cast<IntegerAttr>().getInt();
    uint64_t res = lhsVal + rhsVal;
    return rewriter.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for AddOp: unkonwn data type");
}

template <typename T>
T getAttributeValue(Attribute attr) {
  llvm_unreachable("unknown operation");
}

template <>
double getAttributeValue(Attribute attr) {
  return attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
float getAttributeValue(Attribute attr) {
  return (float)attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
int64_t getAttributeValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

template <>
int32_t getAttributeValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXSubOp>(
    PatternRewriter &rewriter, Type elementType, Attribute lhsAttr,
    Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    double lhsVal = lhsAttr.cast<FloatAttr>().getValueAsDouble();
    double rhsVal = secondAttr.cast<FloatAttr>().getValueAsDouble();
    double res = lhsVal - rhsVal;
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t lhsVal = lhsAttr.cast<IntegerAttr>().getInt();
    uint64_t rhsVal = secondAttr.cast<IntegerAttr>().getInt();
    uint64_t res = lhsVal - rhsVal;
    return rewriter.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SubOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXMulOp>(
    PatternRewriter &rewriter, Type elementType, Attribute lhsAttr,
    Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    double lhsVal = lhsAttr.cast<FloatAttr>().getValueAsDouble();
    double rhsVal = secondAttr.cast<FloatAttr>().getValueAsDouble();
    double res = lhsVal * rhsVal;
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t lhsVal = lhsAttr.cast<IntegerAttr>().getInt();
    uint64_t rhsVal = secondAttr.cast<IntegerAttr>().getInt();
    uint64_t res = lhsVal * rhsVal;
    return rewriter.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for MulOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseBinary<ONNXDivOp>(
    PatternRewriter &rewriter, Type elementType, Attribute lhsAttr,
    Attribute secondAttr) {
  if (elementType.isa<FloatType>()) {
    double lhsVal = lhsAttr.cast<FloatAttr>().getValueAsDouble();
    double rhsVal = secondAttr.cast<FloatAttr>().getValueAsDouble();
    assert(rhsVal != 0 && "division by a zero");
    double res = lhsVal / rhsVal;
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t lhsVal = lhsAttr.cast<IntegerAttr>().getInt();
    uint64_t rhsVal = secondAttr.cast<IntegerAttr>().getInt();
    assert(rhsVal != 0 && "division by a zero");
    uint64_t res = lhsVal / rhsVal;
    return rewriter.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for DivOp: unkonwn data type");
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
void RecurseConstPropElementwiseBinary(PatternRewriter &rewriter,
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
        rewriter, elementaryType, lhsElementAttr, rhsElementAttr);
    resVector.emplace_back(res);
  } else if (lhsFreeRank > rhsFreeRank) {
    // Initial broadcast from lhs.
    auto lhsShape = lhsAttr.getType().getShape();
    int lhsRank = lhsShape.size();
    int lhsIndex = lhsRank - lhsFreeRank;
    int lhsSize = lhsAttr.getType().getShape()[lhsIndex];
    for (int i = 0; i < lhsSize; ++i) {
      lhsIndices[lhsIndex] = i;
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(rewriter,
          resVector, lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank - 1,
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
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(rewriter,
          resVector, lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank,
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
      RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(rewriter,
          resVector, lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank - 1,
          rhsFreeRank - 1);
    }
  }
}

template <typename ElementwiseBinaryOp, typename T>
void FlatConstPropElementwiseBinary(PatternRewriter &rewriter,
    std::vector<T> &resVector, DenseElementsAttr lhsAttr,
    DenseElementsAttr rhsAttr, ArrayRef<int64_t> outputShape) {
  int outputRank = outputShape.size();
  // shape: [M, N, K]
  // strides:      [N * K, K, 1]
  //
  // Given a value x in range [0, M*N*K), convert it to an index of [m, m, k].
  // for(int i = 0; i < rank; ++i) {
  //   s = strides[i]
  //   if (x < s)
  //     indices[i] = 0
  //   else {
  //     indices[i] = x / s
  //     x = x % s
  //   }
  //
  // }

  //  Compute strides
  SmallVector<int64_t, 4> strides(outputRank, 0);
  int64_t elementCount = 1;
  for (int i = outputRank - 1; i >= 0; i--) {
    strides[i] = elementCount;
    elementCount *= outputShape[i];
  }

  resVector.reserve(elementCount);
  for (int64_t i = 0; i < elementCount; ++i) {
    SmallVector<int64_t, 4> outputIndices(outputRank, 0);
    int64_t x = i;
    for (int64_t j = 0; j < outputRank; ++j) {
      int64_t s = strides[j];
      if (x < s)
        outputIndices[j] = 0;
      else {
        outputIndices[j] = floor(x / s);
        x = x % s;
      }
    }

    auto lhsShape = lhsAttr.getType().getShape();
    int lhsRank = lhsShape.size();
    auto rhsShape = rhsAttr.getType().getShape();
    int rhsRank = rhsShape.size();

    // Compute indices to access inputs.
    SmallVector<uint64_t, 4> lhsIndices;
    SmallVector<uint64_t, 4> rhsIndices;
    for (int k = 0; k < outputRank; ++k) {
      // in the lhs index range.
      if (k >= outputRank - lhsRank) {
        int lhsIndex = k - outputRank + lhsRank;
        if (lhsShape[lhsIndex] == 1)
          // broadcast
          lhsIndices.emplace_back(0);
        else
          lhsIndices.emplace_back(outputIndices[k]);
      }
      // in the rhs index range.
      if (k >= outputRank - rhsRank) {
        int rhsIndex = k - outputRank + rhsRank;
        if (rhsShape[rhsIndex] == 1)
          // broadcast
          rhsIndices.emplace_back(0);
        else
          rhsIndices.emplace_back(outputIndices[k]);
      }
    }

    auto lhsElementAttr = lhsAttr.getValue(ArrayRef<uint64_t>(lhsIndices));
    auto rhsElementAttr = rhsAttr.getValue(ArrayRef<uint64_t>(rhsIndices));
    T lhsValue = getAttributeValue<T>(lhsElementAttr);
    T rhsValue = getAttributeValue<T>(rhsElementAttr);
    T res;
    if (std::is_same<ElementwiseBinaryOp, ONNXAddOp>::value)
      res = lhsValue + rhsValue;
    else if (std::is_same<ElementwiseBinaryOp, ONNXSubOp>::value)
      res = lhsValue - rhsValue;
    else if (std::is_same<ElementwiseBinaryOp, ONNXMulOp>::value)
      res = lhsValue * rhsValue;
    else if (std::is_same<ElementwiseBinaryOp, ONNXDivOp>::value)
      res = lhsValue / rhsValue;
    else
      llvm_unreachable("Unsupported operation");
    resVector.emplace_back(res);
  }
}

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseBinaryOp>
DenseElementsAttr ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value resOperand, Attribute lhsAttr, Attribute rhsAttr) {
  DenseElementsAttr lhsDenseAttr =
      lhsAttr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  DenseElementsAttr rhsDenseAttr =
      rhsAttr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert((lhsDenseAttr && lhsDenseAttr) && "expected dense attributes");
  assert(resOperand.getType().cast<ShapedType>().hasRank() &&
         "expected ranked tensor");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  auto outputShape = resOperand.getType().cast<ShapedType>().getShape();

  if (resType.getElementType().isa<FloatType>()) {
    FloatType floatTy = resType.getElementType().cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      std::vector<float> resVector;
      FlatConstPropElementwiseBinary<ElementwiseBinaryOp, float>(
          rewriter, resVector, lhsDenseAttr, rhsDenseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (floatTy.getWidth() == 64) {
      std::vector<double> resVector;
      FlatConstPropElementwiseBinary<ElementwiseBinaryOp, double>(
          rewriter, resVector, lhsDenseAttr, rhsDenseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  if (resType.getElementType().isa<IntegerType>()) {
    IntegerType intTy = resType.getElementType().cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      std::vector<int32_t> resVector;
      FlatConstPropElementwiseBinary<ElementwiseBinaryOp, int32_t>(
          rewriter, resVector, lhsDenseAttr, rhsDenseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (intTy.getWidth() == 64) {
      std::vector<int64_t> resVector;
      FlatConstPropElementwiseBinary<ElementwiseBinaryOp, int64_t>(
          rewriter, resVector, lhsDenseAttr, rhsDenseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  llvm_unreachable("Unknown data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP>
Attribute ComputeConstPropElementwiseUnary(
    PatternRewriter &rewriter, Type elementType, Attribute attr) {
  llvm_unreachable("unkonwn operation");
}

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXNegOp>(
    PatternRewriter &rewriter, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    double val = attr.cast<FloatAttr>().getValueAsDouble();
    double res = -val;
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t val = attr.cast<IntegerAttr>().getInt();
    uint64_t res = -val;
    return rewriter.getIntegerAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for NegOp: unkonwn data type");
}

template <>
Attribute ComputeConstPropElementwiseUnary<ONNXSqrtOp>(
    PatternRewriter &rewriter, Type elementType, Attribute attr) {
  if (elementType.isa<FloatType>()) {
    double val = attr.cast<FloatAttr>().getValueAsDouble();
    double res = sqrt(val);
    return rewriter.getFloatAttr(elementType, res);
  }
  llvm_unreachable("constant propagation for SqrtOp: unkonwn data type");
}

template <typename ElementwiseUnaryOp>
void RecurseConstPropElementwiseUnary(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto elementAttr = attr.getValue(ArrayRef<uint64_t>(indices));
    auto elementaryType = attr.getType().getElementType();
    auto res = ComputeConstPropElementwiseUnary<ElementwiseUnaryOp>(
        rewriter, elementaryType, elementAttr);
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
          rewriter, resVector, attr, indices, freeRank - 1);
    }
  }
}

template <typename ElementwiseUnaryOp, typename T>
void FlatConstPropElementwiseUnary(PatternRewriter &rewriter,
    std::vector<T> &resVector, DenseElementsAttr attr,
    ArrayRef<int64_t> outputShape) {
  int outputRank = outputShape.size();
  // shape: [M, N, K]
  // strides:      [N * K, K, 1]
  //
  // Given a value x in range [0, M*N*K), convert it to an index of [m, m, k].
  // for(int i = 0; i < rank; ++i) {
  //   s = strides[i]
  //   if (x < s)
  //     indices[i] = 0
  //   else {
  //     indices[i] = x / s
  //     x = x % s
  //   }
  //
  // }

  //  Compute strides
  SmallVector<int64_t, 4> strides(outputRank, 0);
  int64_t elementCount = 1;
  for (int i = outputRank - 1; i >= 0; i--) {
    strides[i] = elementCount;
    elementCount *= outputShape[i];
  }

  resVector.reserve(elementCount);
  for (int64_t i = 0; i < elementCount; ++i) {
    SmallVector<uint64_t, 4> outputIndices(outputRank, 0);
    int64_t x = i;
    for (int64_t j = 0; j < outputRank; ++j) {
      int64_t s = strides[j];
      if (x < s)
        outputIndices[j] = 0;
      else {
        outputIndices[j] = x / s;
        x = x % s;
      }
    }

    auto elementAttr = attr.getValue(ArrayRef<uint64_t>(outputIndices));
    auto elementType = attr.getType().getElementType();
    T value = getAttributeValue<T>(elementAttr);
    T res;
    if (std::is_same<ElementwiseUnaryOp, ONNXSqrtOp>::value)
      res = sqrt(value);
    else if (std::is_same<ElementwiseUnaryOp, ONNXNegOp>::value)
      res = -value;
    else
      llvm_unreachable("Unsupported operation");

    resVector.emplace_back(res);
  }
}

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseUnaryOp>
DenseElementsAttr ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value resOperand, Attribute attr) {
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  assert(resOperand.getType().cast<ShapedType>().hasRank() &&
         "expected ranked tensor");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  auto rank = denseAttr.getType().getShape().size();
  SmallVector<uint64_t, 4> indices(rank, 0);
  auto outputShape = resOperand.getType().cast<ShapedType>().getShape();

  if (resType.getElementType().isa<FloatType>()) {
    FloatType floatTy = resType.getElementType().cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      std::vector<float> resVector;
      FlatConstPropElementwiseUnary<ElementwiseUnaryOp, float>(
          rewriter, resVector, denseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (floatTy.getWidth() == 64) {
      std::vector<double> resVector;
      FlatConstPropElementwiseUnary<ElementwiseUnaryOp, double>(
          rewriter, resVector, denseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  if (resType.getElementType().isa<IntegerType>()) {
    IntegerType intTy = resType.getElementType().cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      std::vector<int32_t> resVector;
      FlatConstPropElementwiseUnary<ElementwiseUnaryOp, int32_t>(
          rewriter, resVector, denseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
    if (intTy.getWidth() == 64) {
      std::vector<int64_t> resVector;
      FlatConstPropElementwiseUnary<ElementwiseUnaryOp, int64_t>(
          rewriter, resVector, denseAttr, outputShape);
      return DenseElementsAttr::get(resType, llvm::makeArrayRef(resVector));
    }
  }

  llvm_unreachable("Unknown data type");
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

void RecurseConstPropTranspose(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, SmallVector<uint64_t, 4> &perm,
    int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    auto res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    auto shape = attr.getType().getShape();
    int rank = shape.size();
    int index = perm[rank - freeRank];
    int size = attr.getType().getShape()[index];
    for (int i = 0; i < size; ++i) {
      indices[index] = i;
      RecurseConstPropTranspose(
          rewriter, resVector, attr, indices, perm, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropTranspose(PatternRewriter &rewriter,
    Value resOperand, Attribute attr, ArrayAttr permAttr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  auto rank = denseAttr.getType().getShape().size();
  // Read permute vector.
  SmallVector<uint64_t, 4> perm;
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, 0);
  std::vector<Attribute> resVector;
  // Copy using permute order.
  RecurseConstPropTranspose(
      rewriter, resVector, denseAttr, indices, perm, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

DenseElementsAttr ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value resOperand, Attribute attr) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());

  // Unqueeze does not change the order of access, so just copy the whole data.
  return DenseElementsAttr::getFromRawBuffer(
      resType, denseAttr.getRawData(), denseAttr.isSplat());
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

void RecurseConstPropSplit(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, uint64_t splitAxis, uint64_t axisOffset,
    uint64_t axisSize, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    Attribute res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    ArrayRef<int64_t> shape = attr.getType().getShape();
    int rank = shape.size();
    int index = rank - freeRank;
    int start, size;
    if (index == splitAxis) {
      start = axisOffset;
      size = axisSize;
    } else {
      start = 0;
      size = attr.getType().getShape()[index];
    }
    for (int i = start; i < start + size; ++i) {
      indices[index] = i;
      RecurseConstPropSplit(rewriter, resVector, attr, indices, splitAxis,
          axisOffset, axisSize, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropSplit(PatternRewriter &rewriter, Value resOperand,
    Attribute attr, IntegerAttr axisAttr, ArrayAttr splitAttr,
    unsigned resIndex) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  unsigned rank = denseAttr.getType().getShape().size();
  // Read split axis.
  uint64_t splitAxis = axisAttr.getValue().getSExtValue();
  // Read split vector.
  SmallVector<uint64_t, 4> splits;
  assert(splitAttr && "split attribute expected to be defined here");
  for (Attribute splitVal : splitAttr.getValue())
    splits.emplace_back(splitVal.cast<IntegerAttr>().getInt());
  // Compute the range of elements of interest in the given axis.
  uint64_t axisOffset = 0, axisSize = splits[resIndex];
  for (int i = 0; i < resIndex; ++i)
    axisOffset += splits[i];
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, -1);
  std::vector<Attribute> resVector;
  // Copy.
  RecurseConstPropSplit(rewriter, resVector, denseAttr, indices, splitAxis,
      axisOffset, axisSize, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // A dense attribute that contains constant values of the split op's input.
    Attribute denseAttr;

    // Match
    ONNXSplitOp *splitOp = ::llvm::dyn_cast_or_null<::mlir::ONNXSplitOp>(&op);
    {
      Operation *producerOp = splitOp->input().getDefiningOp();
      ONNXConstantOp castedProducerOp =
          ::llvm::dyn_cast_or_null<::mlir::ONNXConstantOp>(producerOp);
      if (!castedProducerOp)
        return failure();
      // Check whether the constant op is using a dense value or not.
      Attribute sparseAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("sparse_value");
      if (sparseAttr)
        return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
          diag << "entities '' failed to satisfy constraint: Attribute "
                  "is null";
        });
      Attribute dataAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("value");
      denseAttr = dataAttr;
    }

    // Rewrite
    unsigned outputNum = splitOp->getNumResults();
    Value splitInput = splitOp->input();
    int64_t rank = splitInput.getType().cast<ShapedType>().getRank();
    IntegerAttr axisAttr = splitOp->axisAttr();
    ArrayAttr splitAttr = splitOp->splitAttr();
    if (!splitAttr) {
      // If split attribute is not specified, it is constructed from input.
      ArrayRef<int64_t> shape =
          splitInput.getType().cast<ShapedType>().getShape();
      uint64_t splitAxis = axisAttr.getValue().getSExtValue();
      assert(shape[splitAxis] % outputNum == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
      Attribute splitSize = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64), shape[splitAxis] / outputNum);
      SmallVector<Attribute, 4> splits(outputNum, splitSize);
      splitAttr = rewriter.getArrayAttr(splits);
    }

    SmallVector<::mlir::Value, 4> returnValues;
    for (int i = 0; i < outputNum; ++i) {
      Value splitOutput = splitOp->getResults()[i];
      Value constOp =
          rewriter.create<ONNXConstantOp>(loc, splitOutput.getType(),
              /*sparse_value=*/Attribute(),
              /*dense_value=*/
              ConstPropSplit(
                  rewriter, splitOutput, denseAttr, axisAttr, splitAttr, i));
      returnValues.emplace_back(constOp);
    }

    rewriter.replaceOp(op, returnValues);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ONNXConstProp.inc"

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void ConstPropONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  OwningRewritePatternList patterns;
  populateWithGenerated(context, patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());

  applyPatternsAndFoldGreedily(function, std::move(patterns));
} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
