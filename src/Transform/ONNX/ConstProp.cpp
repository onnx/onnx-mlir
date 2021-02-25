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

/// A helper function to get a value of a given type from an attribute.
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

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as inupt
// the element type as well as the two element attributes for the
// operation, and return the result of the operation, also as an
// attribute.
//
template <typename OP, typename T>
struct ElementWiseBinaryOpImpl {
  static T impl(T lhs, T rhs) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAddOp, T> {
  static T impl(T lhs, T rhs) { return (lhs + rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSubOp, T> {
  static T impl(T lhs, T rhs) { return (lhs - rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMulOp, T> {
  static T impl(T lhs, T rhs) { return (lhs * rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXDivOp, T> {
  static T impl(T lhs, T rhs) { return (lhs / rhs); }
};

template <typename OP, typename T>
T ComputeConstPropElementwiseBinary(T lhs, T rhs) {
  return ElementWiseBinaryOpImpl<OP, T>::impl(lhs, rhs);
}

template <typename ElementwiseBinaryOp, typename T>
void FlatConstPropElementwiseBinary(PatternRewriter &rewriter,
    std::vector<T> &resVector, DenseElementsAttr lhsAttr,
    DenseElementsAttr rhsAttr, ArrayRef<int64_t> outputShape) {
  // The algorithm to compute the output in case of broadcasting is as follows:
  // For each value in [0, N), where N is the number of elements in the output:
  //   - compute the access indices for the output.
  //   - deduce access indices for the lhs and rhs from the output access
  //   indices, using broadcasting rules.
  //   - calculate element-wise binary result.
  //   - store the result

  // shape:   [M, N, K]
  // strides: [N * K, K, 1]
  //
  // Given a value x in range [0, M*N*K), convert it to an index of [m, m, k] as
  // follows:
  //
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
  
  int outputRank = outputShape.size();
  auto lhsShape = lhsAttr.getType().getShape();
  int lhsRank = lhsShape.size();
  auto rhsShape = rhsAttr.getType().getShape();
  int rhsRank = rhsShape.size();
  auto elementType = lhsAttr.getType().getElementType();

  //  Compute strides and the number of elements in the output.
  SmallVector<uint64_t, 4> strides(outputRank, 0);
  uint64_t elementCount = 1;
  for (int i = outputRank - 1; i >= 0; i--) {
    strides[i] = elementCount;
    elementCount *= outputShape[i];
  }

  resVector.reserve(elementCount);

  // Check broadcasting.
  bool broadcasting = false;
  if (lhsRank != rhsRank)
    broadcasting = true;
  else
    for (int i = 0; i < outputRank; ++i)
      if (lhsShape[i] != rhsShape[i]) {
        broadcasting = true;
        break;
      }

  // If not broadcasting, it is not necessary to compute access indices because
  // inputs and output have the same memory layout. So it is efficient to
  // traverse data in the increasing order.
  if (!broadcasting) {
    if (elementType.isa<FloatType>()) {
      auto lhsIt = lhsAttr.getValues<FloatAttr>().begin();
      auto rhsIt = rhsAttr.getValues<FloatAttr>().begin();
      for (int i = 0; i < elementCount; ++i) {
        // Get lhs and rhs elements.
        T lhsValue = (T)(*lhsIt++).cast<FloatAttr>().getValueAsDouble();
        T rhsValue = (T)(*rhsIt++).cast<FloatAttr>().getValueAsDouble();
        // Calculate element-wise binary result.
        T res = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
            lhsValue, rhsValue);
        resVector.emplace_back(res);
      }
    } else if (elementType.isa<IntegerType>()) {
      auto lhsIt = lhsAttr.getValues<IntegerAttr>().begin();
      auto rhsIt = rhsAttr.getValues<IntegerAttr>().begin();
      for (int i = 0; i < elementCount; ++i) {
        // Get lhs and rhs elements.
        T lhsValue = (T)(*lhsIt++).cast<IntegerAttr>().getInt();
        T rhsValue = (T)(*rhsIt++).cast<IntegerAttr>().getInt();
        // Calculate element-wise binary result.
        T res = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
            lhsValue, rhsValue);
        resVector.emplace_back(res);
      }
    } else
      llvm_unreachable("Unknown data type");
    return;
  }

  // If broadcasting, go through each element in [0, N), where N is the number
  // of elements in the output, and compute the access indices for the output.
  // Then, use the output access indices to deduce access indices for the
  // inputs, using broadcasting rules.
  for (int64_t i = 0; i < elementCount; ++i) {
    // Compute access indices for the output.
    SmallVector<uint64_t, 4> outputIndices(outputRank, 0);
    uint64_t x = i;
    for (int j = 0; j < outputRank; ++j) {
      uint64_t s = strides[j];
      if (x < s)
        outputIndices[j] = 0;
      else {
        outputIndices[j] = floor(x / s);
        x = x % s;
      }
    }

    // Compute indices to access inputs.
    SmallVector<uint64_t, 4> lhsIndices, rhsIndices;
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

    // Get lhs and rhs elements.
    Attribute lhsElementAttr = lhsAttr.getValue(ArrayRef<uint64_t>(lhsIndices));
    Attribute rhsElementAttr = rhsAttr.getValue(ArrayRef<uint64_t>(rhsIndices));

    // Calculate element-wise binary result.
    T lhsValue = getAttributeValue<T>(lhsElementAttr);
    T rhsValue = getAttributeValue<T>(rhsElementAttr);
    T res = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
        lhsValue, rhsValue);
    resVector.emplace_back(res);
  }
  return;
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

  // FloatType
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

  // IntegerType
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

template <typename OP, typename T>
struct ElementWiseUnaryOpImpl {
  static T impl(T val) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T> {
  static T impl(T val) { return (-val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, T> {
  static T impl(T val) { return sqrt(val); }
};

template <typename OP, typename T>
T ComputeConstPropElementwiseUnary(T val) {
  return ElementWiseUnaryOpImpl<OP, T>::impl(val);
}

template <typename ElementwiseUnaryOp, typename T>
void FlatConstPropElementwiseUnary(PatternRewriter &rewriter,
    std::vector<T> &resVector, DenseElementsAttr attr,
    ArrayRef<int64_t> outputShape) {
  int64_t elementCount = 1;
  for (int i = 0; i < outputShape.size(); ++i) {
    elementCount *= outputShape[i];
  }

  resVector.reserve(elementCount);

  auto elementType = attr.getType().getElementType();
  if (elementType.isa<FloatType>()) {
    auto it = attr.getValues<FloatAttr>().begin();
    for (int i = 0; i < elementCount; ++i) {
      T value = (T)(*it++).cast<FloatAttr>().getValueAsDouble();
      T res = ComputeConstPropElementwiseUnary<ElementwiseUnaryOp, T>(value);
      resVector.emplace_back(res);
    }
  } else if (elementType.isa<IntegerType>()) {
    auto it = attr.getValues<IntegerAttr>().begin();
    for (int i = 0; i < elementCount; ++i) {
      T value = (T)(*it++).cast<IntegerAttr>().getInt();
      T res = ComputeConstPropElementwiseUnary<ElementwiseUnaryOp, T>(value);
      resVector.emplace_back(res);
    }
  } else
    llvm_unreachable("Unknown data type");
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

  // FloatType
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

  // IntegerType
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
