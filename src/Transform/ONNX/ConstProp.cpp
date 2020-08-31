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
  assert(
      resOperand.getType().isa<RankedTensorType>() && "expected ranked tensor");
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
  auto lhsRank = lhsDenseAttr.getType().getShape().size();
  auto rhsRank = rhsDenseAttr.getType().getShape().size();
  SmallVector<uint64_t, 4> lhsIndices(lhsRank, 0);
  SmallVector<uint64_t, 4> rhsIndices(rhsRank, 0);
  std::vector<Attribute> resVector;
  RecurseConstPropElementwiseBinary<ElementwiseBinaryOp>(rewriter, resVector,
      lhsDenseAttr, rhsDenseAttr, lhsIndices, rhsIndices, lhsRank, rhsRank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
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

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseUnaryOp>
DenseElementsAttr ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value resOperand, Attribute attr) {
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
      rewriter, resVector, denseAttr, indices, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
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
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();
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
  ShapedType resType = resOperand.getType().cast<RankedTensorType>();

  // Unqueeze does not change the order of access, so just copy the whole data.
  std::vector<Attribute> resVector;
  for (auto value : denseAttr.getValues<Attribute>()) {
    resVector.emplace_back(value);
  }

  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

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
  populateWithGenerated(context, &patterns);

  applyPatternsAndFoldGreedily(function, patterns);
} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
