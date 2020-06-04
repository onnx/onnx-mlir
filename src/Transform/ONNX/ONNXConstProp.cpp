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

using namespace mlir;

namespace {

////////////////////////////////////////////////////////////////////////////////
// Code to perform constant propagation in presence of broadcast.
////////////////////////////////////////////////////////////////////////////////

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation, also as an
// attribute.

template <typename OP>
Attribute ComputeConstProppElementwiseBinary(PatternRewriter &rewriter,
    Type elementType, Attribute &lhsAttr, Attribute &secondAttr) {
  llvm_unreachable("unkonwn operation");
}

template <>
Attribute ComputeConstProppElementwiseBinary<ONNXAddOp>(
    PatternRewriter &rewriter, Type elementType, Attribute &lhsAttr,
    Attribute &secondAttr) {
  if (elementType.isa<FloatType>()) {
    double lhsVal = lhsAttr.cast<FloatAttr>().getValueAsDouble();
    double rhsVal = secondAttr.cast<FloatAttr>().getValueAsDouble();
    double res = lhsVal + rhsVal;
    // printf("  %f + %f -> %f\n", lhsVal, rhsVal, res);
    // Could use the APFloat interface to emulate the results, are ok to simply
    // perform them in the highest possible precision.
    return rewriter.getFloatAttr(elementType, res);
  }
  if (elementType.isa<IntegerType>()) {
    uint64_t lhsVal = lhsAttr.cast<IntegerAttr>().getInt();
    uint64_t rhsVal = secondAttr.cast<IntegerAttr>().getInt();
    uint64_t res = lhsVal + rhsVal;
    // printf("  %llu + %llu -> %llu\n", (unsigned long long) lhsVal, (unsigned
    // long long) rhsVal, (unsigned long long) res);
    return rewriter.getIntegerAttr(elementType, lhsVal + rhsVal);
  }
  llvm_unreachable("constant propagation for AddOp: unkonwn data type");
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
void RecurseConstProppElementwiseBinary(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr &lhsAttr,
    DenseElementsAttr &rhsAttr, SmallVector<uint64_t, 4> &lhsIndices,
    SmallVector<uint64_t, 4> &rhsIndices, int lhsFreeRank, int rhsFreeRank) {
  // printf("recurse with free %d/%d\n", lhsFreeRank, rhsFreeRank);
  if (lhsFreeRank == 0) {
    // Fully defined ranks.
    assert(
        rhsFreeRank == 0 && "expect both to recurse to zero at the same time");
    auto lhsElementAttr = lhsAttr.getValue(ArrayRef<uint64_t>(lhsIndices));
    auto rhsElementAttr = rhsAttr.getValue(ArrayRef<uint64_t>(rhsIndices));
    auto elementaryType = lhsAttr.getType().getElementType();
    auto res = ComputeConstProppElementwiseBinary<ElementwiseBinaryOp>(
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
      RecurseConstProppElementwiseBinary<ElementwiseBinaryOp>(rewriter,
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
      RecurseConstProppElementwiseBinary<ElementwiseBinaryOp>(rewriter,
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
      RecurseConstProppElementwiseBinary<ElementwiseBinaryOp>(rewriter,
          resVector, lhsAttr, rhsAttr, lhsIndices, rhsIndices, lhsFreeRank - 1,
          rhsFreeRank - 1);
    }
  }
}

// Process the constant operands, perform the operation with broadcast, and
// generate the new constant operation.
template <typename ElementwiseBinaryOp>
DenseElementsAttr ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value &resOperand, Attribute &lhsAttr, Attribute &rhsAttr) {
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
  RecurseConstProppElementwiseBinary<ElementwiseBinaryOp>(rewriter, resVector,
      lhsDenseAttr, rhsDenseAttr, lhsIndices, rhsIndices, lhsRank, rhsRank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

// Function called by the rules to generate the proper added constants
DenseElementsAttr ConstPropForAddOfTwoConst(
    PatternRewriter &rewriter, Value resOperand, Attribute &v1, Attribute &v2) {
  return ConstPropElementwiseBinary<mlir::ONNXAddOp>(
      rewriter, resOperand, v1, v2);
}

////////////////////////////////////////////////////////////////////////////////
// Pattern definition.
////////////////////////////////////////////////////////////////////////////////

#include "src/Transform/ONNX/ONNXConstProp.inc"

////////////////////////////////////////////////////////////////////////////////
// Code to manage the pass.
////////////////////////////////////////////////////////////////////////////////

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

static PassRegistration<ConstPropONNXToONNXPass> pass("constprop-onnx",
    "ConstProp ONNX operations into composition of other ONNX operations.");
