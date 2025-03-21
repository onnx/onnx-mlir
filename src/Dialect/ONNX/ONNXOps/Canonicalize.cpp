/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
// When adding a canonicalizer for a new operation, please add that operation to
// the OpsWithCanonicalizer list in utils/gen_onnx_mlir.py
//
//===----------------------------------------------------------------------===//

#include <math.h>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "rewrite"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

// =============================================================================
// Helper functions for Rewrite.td and Rewrite.cpp files.
// =============================================================================

// If 'A' is NoneType, return -B. Otherwise return A-B.
Value subtractOrNeg(PatternRewriter &rewriter, Location loc, Value A, Value B) {
  if (mlir::isa<NoneType>(A.getType()))
    return rewriter.create<ONNXNegOp>(loc, B);
  return rewriter.create<ONNXSubOp>(loc, A, B);
}

// Create an ArrayAttr of IntegerAttr(s) of values in [N, M].
ArrayAttr createArrayAttrOfNToM(PatternRewriter &rewriter, int N, int M) {
  SmallVector<int64_t, 4> vals;
  for (int i = N; i <= M; ++i)
    vals.emplace_back(i);
  return rewriter.getI64ArrayAttr(vals);
}

// Create an DenseElementsAttr of i64 values in [N, M].
DenseElementsAttr createDenseElementsAttrOfNToM(
    PatternRewriter &rewriter, int64_t N, int64_t M) {
  SmallVector<int64_t, 4> vals;
  for (int i = N; i <= M; ++i)
    vals.emplace_back(i);
  return rewriter.getI64TensorAttr(vals);
}

// Get return type for a MatMulOp whose A's rank is N (>2) and B's rank is 2.
Type getReturnTypeForMatMulOpND2D(Value A, Value B) {
  ArrayRef<int64_t> aShape =
      mlir::cast<RankedTensorType>(A.getType()).getShape();
  ArrayRef<int64_t> bShape =
      mlir::cast<RankedTensorType>(B.getType()).getShape();
  SmallVector<int64_t> resShape(aShape.begin(), aShape.end() - 1);
  resShape.emplace_back(bShape[bShape.size() - 1]);
  return RankedTensorType::get(
      resShape, mlir::cast<ShapedType>(A.getType()).getElementType());
}

// Get the index of the axis value in the given permutation array.
IntegerAttr getIndexOfAxisInPerm(
    PatternRewriter &rewriter, ArrayAttr permAttr, IntegerAttr axis) {
  IntegerAttr result;
  for (uint64_t i = 0; i < permAttr.getValue().size(); ++i) {
    IntegerAttr attr = mlir::cast<IntegerAttr>(permAttr.getValue()[i]);
    assert(attr && "Element in ArrayAttr is not IntegerAttr");
    if (attr.getValue().getSExtValue() == axis.getValue().getSExtValue())
      return rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), i);
  }
  return result;
}

// Transpose a variadic input using a permutation array.
SmallVector<Value, 4> transposeVariadicInput(PatternRewriter &rewriter,
    Location loc, ValueRange inputs, ArrayAttr permAttr) {
  SmallVector<Value, 4> transposedInputs;
  for (Value inp : inputs) {
    ShapedType inpType = mlir::cast<ShapedType>(inp.getType());
    assert(inpType && "Type is not ShapedType");
    ONNXTransposeOp transposeOp = rewriter.create<ONNXTransposeOp>(
        loc, UnrankedTensorType::get(inpType.getElementType()), inp, permAttr);
    static_cast<void>(transposeOp.inferShapes([](Region &region) {}));
    transposedInputs.emplace_back(transposeOp.getResult());
  }
  return transposedInputs;
}

// Cast a variadic input using the given `saturate` and `to`.
SmallVector<Value, 4> castVariadicInput(PatternRewriter &rewriter, Location loc,
    ValueRange inputs, IntegerAttr saturate, TypeAttr to) {
  SmallVector<Value, 4> castInputs;
  for (Value inp : inputs) {
    ShapedType inpType = mlir::cast<ShapedType>(inp.getType());
    assert(inpType && "Type is not ShapedType");
    ONNXCastOp castOp = rewriter.create<ONNXCastOp>(loc,
        UnrankedTensorType::get(inpType.getElementType()), inp, saturate, to);
    static_cast<void>(castOp.inferShapes([](Region &region) {}));
    castInputs.emplace_back(castOp.getResult());
  }
  return castInputs;
}

// Check if all values are produced by ONNXTransposeOp.
bool areProducedByTransposeOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    if (mlir::isa<BlockArgument>(v))
      return false;
    return isa<ONNXTransposeOp>(v.getDefiningOp());
  });
}

// Create a DenseElementsAttr based on the shape of type.
DenseElementsAttr createDenseElementsAttrFromShape(PatternRewriter &rewriter,
    Value value, int64_t start = 0, std::optional<int64_t> end = std::nullopt) {

  auto inType = mlir::cast<ShapedType>(value.getType());
  assert(inType.hasRank() && "inType must be ranked");
  auto shape = inType.getShape();
  int64_t rank = inType.getRank();

  int64_t endValue = end.has_value() ? end.value() : rank;

  SmallVector<int64_t, 1> dims = {endValue - start};
  SmallVector<int64_t, 4> values(
      shape.begin() + start, shape.begin() + endValue);
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

// Create a DenseElementsAttr from Shape Op
DenseElementsAttr createDenseElementsAttrFromShapeOp(
    PatternRewriter &rewriter, Operation *op) {
  ONNXShapeOp shapeOp = llvm::cast<ONNXShapeOp>(op);
  int64_t start, end;
  ONNXShapeOpShapeHelper::getStartEndValues(shapeOp, start, end);
  return createDenseElementsAttrFromShape(
      rewriter, shapeOp.getData(), start, end);
}

/// Test if two axis arrays contain the same values or not.
/// If rank != 0 then negative axes are adjusted by adding rank.
/// No checking is done for invariants like out of range axes
/// or duplicate axes.
bool AreTheSameAxesArrayAttr(
    int64_t rank, ArrayAttr lhsAttr, ArrayAttr rhsAttr) {
  if (!lhsAttr || !rhsAttr)
    return false;

  auto asSet = [rank](ArrayRef<Attribute> array) {
    llvm::SmallSet<int64_t, 6> axes;
    for (auto attr : array) {
      int64_t axis = mlir::cast<IntegerAttr>(attr).getInt();
      axes.insert(axis < 0 ? axis + rank : axis);
    }
    return axes;
  };
  return asSet(lhsAttr.getValue()) == asSet(rhsAttr.getValue());
}

// Same as AreTheSameAxesArrayAttr but takes (result value of)
// ONNXConstantOp tensors as inputs.
// Returns false if any of the input Values are not constant results.
bool AreTheSameAxesConstant(int64_t rank, Value lhs, Value rhs) {
  assert(cast<ShapedType>(lhs.getType()).getElementType().isInteger(64));
  assert(cast<ShapedType>(rhs.getType()).getElementType().isInteger(64));
  auto lhsConstOp = mlir::dyn_cast_or_null<ONNXConstantOp>(lhs.getDefiningOp());
  auto rhsConstOp = mlir::dyn_cast_or_null<ONNXConstantOp>(rhs.getDefiningOp());
  return lhsConstOp && rhsConstOp &&
         AreTheSameAxesArrayAttr(rank,
             createArrayAttrFromConstantOp(lhsConstOp),
             createArrayAttrFromConstantOp(rhsConstOp));
}

/// Test if two values have the same static shape or not.
bool haveSameStaticShape(Value lhs, Value rhs) {
  if (!hasShapeAndRank(lhs) || !hasShapeAndRank(rhs))
    return false;
  Type lhsT = lhs.getType();
  Type rhsT = rhs.getType();
  return hasStaticShape(lhsT) && (getShape(lhsT) == getShape(rhsT));
}

/// Test if the input is a splat constant with a negative value or not.
bool isNegativeSplatConstant(Value val) {
  ElementsAttr valAttr = getElementAttributeFromONNXValue(val);
  if (!valAttr)
    return false;

  if (!valAttr.isSplat())
    return false;

  Type elemTy = mlir::cast<ShapedType>(val.getType()).getElementType();
  if (mlir::isa<FloatType>(elemTy)) {
    double v = valAttr.getSplatValue<double>();
    return (v < 0.0);
  } else if (mlir::isa<IntegerType>(elemTy)) {
    int64_t v = valAttr.getSplatValue<int64_t>();
    return (v < 0);
  }
  return false;
}

/// Test if the input is a constant with all negative small value or not.
// This function assumes input constant value(`val`) is dimension size. So, set
// 10 as the size of small constnt value.
bool isAllNegativeSmallIntegerConstant(Value val) {
  ElementsAttr valAttr = getElementAttributeFromONNXValue(val);
  if (!valAttr)
    return false;

  if (valAttr.size() > 10)
    return false;

  Type elemTy = mlir::cast<ShapedType>(val.getType()).getElementType();
  if (mlir::isa<IntegerType>(elemTy)) {
    for (auto v : valAttr.getValues<APInt>()) {
      if (v.getSExtValue() > 0)
        return false;
    }
  } else {
    return false;
  }
  return true;
}

/// Test if all values in the input ValueRange are dimension sizes.
bool areAllDimSizes(ValueRange vals) {
  return llvm::all_of(vals, [](Value val) {
    // Block arguments.
    if (mlir::isa<BlockArgument>(val))
      return false;
    // Defined by DimOp.
    if (val.getDefiningOp<ONNXDimOp>())
      return true;
    // Defined by ConstantOp.
    if (isDenseONNXConstant(val) && isScalarTensor(val)) {
      Type elemTy = mlir::cast<ShapedType>(val.getType()).getElementType();
      if (!mlir::isa<IntegerType>(elemTy))
        return false;
      ElementsAttr valAttr = getElementAttributeFromONNXValue(val);
      if (!valAttr)
        return false;
      int64_t v = (*valAttr.getValues<APInt>().begin()).getSExtValue();
      return (v > 0);
    }
    return false;
  });
}

// Match v = shape_transform(X*A + B).
// shape_transform is a sequence of operations like Reshape, Transpose,
// Squeeze, Unsqueeze, etc. that do not change the numerical values by data
// shape.
// A and B are constants.
bool matchShapeAddMatMul(Value v, Value &matA, Value &biasB,
    Operation *&matmulOrGemmOp, Operation *&addOp, bool &isGemm) {
  if (mlir::isa<BlockArgument>(v))
    return false;
  if (!hasOneUseExceptDimOp(v))
    return false;
  Value origV = v;
  // Match a sequence of shape operations. Each shape operation has only one
  // use.
  while (auto defOp = origV.getDefiningOp()) {
    if (!isa<ONNXReshapeOp, ONNXTransposeOp, ONNXSqueezeOp, ONNXUnsqueezeOp>(
            defOp))
      break;
    origV = defOp->getOperands()[0];
    if (!hasOneUseExceptDimOp(origV))
      break;
  }
  if (mlir::isa<BlockArgument>(origV) || !hasOneUseExceptDimOp(origV))
    return false;

  // Match Gemm
  auto onnxGemmOp = origV.getDefiningOp<ONNXGemmOp>();
  if (onnxGemmOp) {
    if (!isDenseONNXConstant(onnxGemmOp.getB()))
      return false;
    if (!isNoneValue(onnxGemmOp.getC()) &&
        !isDenseONNXConstant(onnxGemmOp.getC()))
      return false;
    matmulOrGemmOp = onnxGemmOp.getOperation();
    matA = onnxGemmOp.getB();
    biasB = onnxGemmOp.getC();
    isGemm = true;
    return true;
  }

  // Not Gemm, match Add.
  auto onnxAddOp = origV.getDefiningOp<ONNXAddOp>();
  if (!onnxAddOp)
    return false;
  Value lhsAdd = onnxAddOp.getA();
  Value rhsAdd = onnxAddOp.getB();

  // LHS of Add is the only one use of MatMul's result.
  if (!hasOneUseExceptDimOp(lhsAdd))
    return false;
  auto onnxMatMulOp = lhsAdd.getDefiningOp<ONNXMatMulOp>();
  if (!onnxMatMulOp)
    return false;
  Value rhsMatMul = onnxMatMulOp.getB();
  if (!isDenseONNXConstant(rhsMatMul))
    return false;

  // RHS of Add is a constant.
  if (!isDenseONNXConstant(rhsAdd))
    return false;

  // Passed all tests.
  matmulOrGemmOp = onnxMatMulOp.getOperation();
  addOp = onnxAddOp.getOperation();
  matA = rhsMatMul;
  biasB = rhsAdd;
  isGemm = false;

  return true;
}

} // namespace onnx_mlir

// =============================================================================
/// Include the patterns defined in the Declarative Rewrite framework.
// =============================================================================

#include "src/Dialect/ONNX/ONNXOps/ONNXCanonicalize.inc"

// =============================================================================
// Rewrite pattern for elementwise binary ops (not handled in Rewrite.td).
// =============================================================================

// Rewrites v1-v6 binary op with legacy axis and broadcast attributes set
// by unsqueezing the rhs shape as needed and removing the axis and broadcast
// attributes, provided that the operand shapes' ranks are known.
// The v1-v6 binary ops with axis and broadcast attributes are:
// Add, And, Div, Equal, Greater, Less, Or, Pow, Sub, Xor.
template <typename OP_TYPE>
class BinaryOpBroadcastAxisPattern : public OpRewritePattern<OP_TYPE> {
public:
  using OpRewritePattern<OP_TYPE>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OP_TYPE binaryOp, PatternRewriter &rewriter) const override {
    Operation *op = binaryOp.getOperation();

    IntegerAttr bcast = op->getAttrOfType<IntegerAttr>("broadcast");
    IntegerAttr axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!bcast || bcast.getValue().getSExtValue() != 1 || !axisAttr) {
      return failure(); // Pattern only applies when broadcast and axis are set.
    }
    int64_t axis = axisAttr.getValue().getSExtValue();

    assert(op->getNumOperands() == 2 && "op must be binary");
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    ShapedType lhsType = mlir::cast<ShapedType>(lhs.getType());
    ShapedType rhsType = mlir::cast<ShapedType>(rhs.getType());
    if (!lhsType.hasRank() || !rhsType.hasRank()) {
      return failure(); // Cannot apply pattern until ranks are known.
    }
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    if (axis > lhsRank) {
      return op->emitOpError("broadcast axis out of range: ")
             << "axis " << axis << ", lhs type " << lhsType;
    }
    if (rhsRank > lhsRank - axis) {
      return op->emitOpError("broadcast rhs shape too long: ")
             << "axis " << axis << ", lhs type " << lhsType << ", rhs type "
             << rhsType;
    }

    rewriter.modifyOpInPlace(op, [&] {
      if (rhsRank < lhsRank - axis) {
        OnnxBuilder createONNX(rewriter, op->getLoc());
        SmallVector<int64_t> axesArray;
        SmallVector<int64_t> unsqueezedShape(rhsType.getShape());
        for (int64_t x = rhsRank; x < lhsRank - axis; ++x) {
          axesArray.push_back(x);
          unsqueezedShape.push_back(1);
        }
        Value axes = createONNX.constantInt64(axesArray);
        auto unsqueezedType =
            RankedTensorType::get(unsqueezedShape, rhsType.getElementType());
        Value unsqueezed = createONNX.unsqueeze(unsqueezedType, rhs, axes);
        op->setOperand(1, unsqueezed);
      }
      Attribute removedAxisAttr = op->removeAttr("axis");
      assert(removedAxisAttr && "axis should be removed");
      Attribute removedBroadcastAttr = op->removeAttr("broadcast");
      assert(removedBroadcastAttr && "broadcast should be removed");
    });
    return success();
  }
};

// A pattern to turn
//   `BinaryOp(Constant_X, ExpandOp(Constant_Y))`
// into
//   `ExpandOp(BinaryOp(Constant_X, Constant_Y))`
// which put constants together so that BinaryOp can be folded. This pattern
// only handles the case where one of the operand is a scalar constant. For such
// a case, we can easily infer the shape operand for the resulting ExpandOp.

template <typename OP_TYPE>
class PropagateScalarConstantExpandPattern : public OpRewritePattern<OP_TYPE> {
public:
  using OpRewritePattern<OP_TYPE>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OP_TYPE binaryOp, PatternRewriter &rewriter) const override {
    Operation *op = binaryOp.getOperation();
    Location loc = binaryOp.getLoc();

    assert(op->getNumOperands() == 2 && "op must be binary");
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type outputType = op->getResult(0).getType();

    // Match
    //  - lhs is a scalar constant, and
    //  - rhs is ExpandOp whose input is a scalar constant, or vice versa.
    Value expandShape = nullptr;
    auto matchValue = [&expandShape](Value v) -> Value {
      Value res = v;
      if (auto expandOp =
              dyn_cast_if_present<ONNXExpandOp>(res.getDefiningOp())) {
        if (!expandShape) {
          res = expandOp.getInput();
          expandShape = expandOp.getShape();
        }
      }
      if (isDenseONNXConstant(res) && isScalarTensor(res))
        return res;
      return nullptr;
    };
    Value lhsConstant = matchValue(lhs);
    Value rhsConstant = matchValue(rhs);
    if (!expandShape || !lhsConstant || !rhsConstant)
      return failure();
    // Does not handle empty shape in ExpandOp, e.g. of type tensor<0xdtype>.
    if (!hasShapeAndRank(expandShape))
      return failure();
    ArrayRef<int64_t> dims = getShape(expandShape.getType());
    if ((dims.size() == 1) && (dims[0] == 0))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value res = create.onnx.expand(outputType,
        create.onnx.createOpAndInferShapes<OP_TYPE>(lhsConstant, rhsConstant),
        expandShape);

    rewriter.replaceOp(op, {res});
    return success();
  }
};

template <typename OP_TYPE>
class PropagateReshapeThroughBinaryOpPattern
    : public OpRewritePattern<OP_TYPE> {
public:
  using OpRewritePattern<OP_TYPE>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OP_TYPE binaryOp, PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    Operation *op = binaryOp.getOperation();

    assert(op->getNumOperands() == 2 && "op must be binary");
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type outputType = binaryOp.getResult().getType();

    Value reshapeInput;
    Value reshapeShape;
    IntegerAttr reshapeAZ;

    // Match
    // LHS is produced by a Reshape.
    Operation *reshapeGenericOp = lhs.getDefiningOp();
    if (!reshapeGenericOp)
      return failure();
    auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(reshapeGenericOp);
    if (!reshapeOp)
      return failure();
    // RHS is a scalar.
    if (!isScalarTensor(rhs))
      return failure();

    // Rewrite
    auto loc = rewriter.getFusedLoc({op->getLoc(), reshapeGenericOp->getLoc()});
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    reshapeInput = reshapeOp.getData();
    reshapeShape = reshapeOp.getShape();
    reshapeAZ = reshapeOp.getAllowzeroAttr();
    Value x = rewriter.create<OP_TYPE>(loc, reshapeInput, rhs);
    Value res = create.onnx.reshape(outputType, x, reshapeShape, reshapeAZ);

    rewriter.replaceOp(op, res);
    return success();
  };
};

// This rewriting is to optimize the scalar Div/Mul in self-attention layers.
// In particular, it rewrites the following pattern:
// ```
// shape_transform(X1 * A1 + B1) * shape_transform(X2 * A2 + B2) / k
// ```
//
// into
// ```
// shape_transform(X1 * A1 + B1) * shape_transform(X2 * A2/k + B2/k)
// ```
// if A2, B2 and k are constants,
//
// or into
// ```
// shape_transform(X1 * A1/k + B1/k) * shape_transform(X2 * A2 + B2)
// ```
// if A1, B1 and k are constants,
//
// where
// - * is matrix multiplication; + and / are element-wise addition and division
// - A1, A2, B1, B2, and k are constants so that A1/k, B1/k, A2/k and B2/k can
// be folded. k is a scalar constant so that it's broadcastable to all A1, A2,
// B1, B2.
// - shape_transform includes a sequence of operations that change the data
// shape of the input but not numerical values, for example: Reshape,
// Transpose, etc.
//
// This pattern supports both division and multiplication by k.
template <typename ONNXOp>
struct PropagateConstantScalingInAttentionLayerPattern
    : public OpRewritePattern<ONNXOp> {
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp omOp, PatternRewriter &rewriter) const final {
    Operation *genericOp = omOp.getOperation();
    Value lhsOMOp = omOp.getA();
    Value K = omOp.getB();

    // Match (lhs * rhs) / K.
    // The first operand of Div/Mul is produced by MatMulOp.
    auto onnxMatMulOp = lhsOMOp.getDefiningOp<ONNXMatMulOp>();
    if (!onnxMatMulOp)
      return rewriter.notifyMatchFailure(genericOp,
          "The first operand of Div/Mul is not produced by MatMulOp");
    Value lhs = onnxMatMulOp.getA();
    Value rhs = onnxMatMulOp.getB();
    // The second operand of Div/Mul is a scalar constant.
    if (!isScalarConstantTensor(K))
      return rewriter.notifyMatchFailure(
          genericOp, "The second operand of Div/Mul is not a scalar constant");

    // Match lhs = shape_transform(X1*A1 + B1)
    Value A, B;
    Operation *matmulOrGemmOp, *addOp;
    bool isGemm;
    bool matched =
        matchShapeAddMatMul(lhs, A, B, matmulOrGemmOp, addOp, isGemm);

    if (!matched) {
      // Match rhs = shape_transform(X2*A2 + B2)
      matched = matchShapeAddMatMul(rhs, A, B, matmulOrGemmOp, addOp, isGemm);
    }

    if (!matched)
      return rewriter.notifyMatchFailure(genericOp,
          "There is no constant tensor to replace the first operand "
          "of Div/Mul");

    // Rewrite.
    // Move K up before MatMul/Gemm to make sure it is in the dominant region.
    K.getDefiningOp()->moveBefore(matmulOrGemmOp);
    if (isGemm) {
      auto onnxGemmOp = cast<ONNXGemmOp>(matmulOrGemmOp);
      // Update in place B and C of Gemm.
      rewriter.modifyOpInPlace(onnxGemmOp, [&] {
        rewriter.setInsertionPoint(onnxGemmOp);
        onnxGemmOp.getBMutable().assign(rewriter.create<ONNXOp>(
            onnxGemmOp.getLoc(), onnxGemmOp.getB().getType(), A, K));
        if (!isNoneValue(onnxGemmOp.getC()))
          onnxGemmOp.getCMutable().assign(rewriter.create<ONNXOp>(
              onnxGemmOp.getLoc(), onnxGemmOp.getC().getType(), B, K));
      });
    } else {
      auto onnxSubMatOp = mlir::cast<ONNXMatMulOp>(matmulOrGemmOp);
      auto onnxAddOp = mlir::cast<ONNXAddOp>(addOp);
      // Update in place MatMul and Add.
      rewriter.modifyOpInPlace(onnxSubMatOp, [&] {
        rewriter.setInsertionPoint(onnxSubMatOp);
        onnxSubMatOp.getBMutable().assign(rewriter.create<ONNXOp>(
            onnxSubMatOp.getLoc(), onnxSubMatOp.getB().getType(), A, K));
      });
      rewriter.modifyOpInPlace(onnxAddOp, [&] {
        OnnxBuilder createONNX(rewriter, onnxAddOp.getLoc());
        rewriter.setInsertionPoint(onnxAddOp);
        onnxAddOp.getBMutable().assign(rewriter.create<ONNXOp>(
            onnxAddOp.getLoc(), onnxAddOp.getB().getType(), B, K));
      });
    }

    // Bypass Div/Mul.
    rewriter.replaceOp(genericOp, onnxMatMulOp.getY());
    return success();
  }
};

// =============================================================================
// Rewrite pattern for Resize (not handled in Rewrite.td).
// =============================================================================

// The yolo4 model uses a float tensor with shape [0] to represent that roi
// or scales is absent in accordance with the Resize v11 spec. This violates
// the spec from v13 onwards which says that empty string
// inputs represents absent arguments in the protobuf model representation.
// We work around this by interpreting a tensor with empty shape as an
// alternative way to express that an input is absent.
class EmptyTensorInputsResizePattern : public OpRewritePattern<ONNXResizeOp> {
public:
  using OpRewritePattern<ONNXResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXResizeOp onnxResizeOp, PatternRewriter &rewriter) const override {
    bool emptyRoi = isEmptyTensor(onnxResizeOp.getRoi());
    bool emptyScales = isEmptyTensor(onnxResizeOp.getScales());
    bool emptySizes = isEmptyTensor(onnxResizeOp.getSizes());
    if (emptyRoi || emptyScales || emptySizes) {
      rewriter.modifyOpInPlace(onnxResizeOp, [&] {
        OnnxBuilder createONNX(rewriter, onnxResizeOp.getLoc());
        if (emptyRoi)
          onnxResizeOp.getRoiMutable().assign(createONNX.none());
        if (emptyScales)
          onnxResizeOp.getScalesMutable().assign(createONNX.none());
        if (emptySizes)
          onnxResizeOp.getSizesMutable().assign(createONNX.none());
      });
      return success();
    } else {
      return failure(); // pattern didn't apply and onnxResizeOp is unchanged
    }
  }

private:
  bool isEmptyTensor(Value input) const {
    if (ShapedType shapedType = mlir::dyn_cast<ShapedType>(input.getType())) {
      return shapedType.hasStaticShape() && shapedType.getNumElements() == 0;
    } else {
      return false;
    }
  }
};

// =============================================================================
// Rewrite pattern for loop (not handled in Rewrite.td).
// =============================================================================

// In some ONNX models, the maximum trip count for LoopOp is set to a big value,
// e.g. LONG_MAX and termination depends on the break condition inside the loop.
// In the current lowering of LoopOp, the maximum trip count is used to allocate
// a buffer for all intermediate loop results. Since the actual number of loop
// iterations may be much smaller than the maximum trip count, it is redundant
// and error-prone to allocate a large buffer. For example, we may get segfault
// if the maximum trip count is out of range.
//
// This pattern tries to derive a new maximum trip count for LoopOp by analyzing
// the break condition. It only handles a special case where the loop is like a
// for-loop with step, e.g. `for (i = LB, i < UB, i = i + Step)`.
//
// For example, the following loop which mimics LoopOp:
// ```
// max_trip_count=9223372036854775807
// LB = -100
// UB = 100
// Step = 1
//
// i = 0
// k = LB
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
//    keepGoing = (k < UB)
// }
// ```
//
// will be rewritten into:
//
// ```
// max_trip_count=200
// LB = -100
// UB = 100
//
// i = 0
// k = LB
// keepGoing = true
// while (i < max_trip_count && keepGoing == true) {
//    k = k + STEP
// }
// ```
// where `max_trip_count` is replaced by an actual value derived from the loop.
//
class LoopOpRewriteMaxTripCountPattern : public OpRewritePattern<ONNXLoopOp> {
public:
  using OpRewritePattern<ONNXLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLoopOp onnxLoopOp, PatternRewriter &rewriter) const override {
    Location loc = onnxLoopOp.getLoc();
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // Match the following pattern:
    // ```
    // ubValue = ONNXConstantOp() {value = ...}
    // startValue = ONNXConstantOp() {value = ...}
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond_new = cond
    //     ONNXYieldOp (cond_new, ..., ubValue, ..., newCounterValue, ...)
    // ```
    bool matched;
    Value newMaxTripCountValue;
    std::tie(matched, newMaxTripCountValue) =
        matchOp(rewriter, loc, onnxLoopOp);
    if (!matched)
      return failure();

    // Rewrite
    loopOp->replaceUsesOfWith(maxTripCountValue, newMaxTripCountValue);
    // Modify the condition return
    Region &loopBody = onnxLoopOp.getBody();
    Operation *loopBodyTerminator = loopBody.front().getTerminator();
    loopBodyTerminator->setOperand(0, loopBody.front().getArgument(1));
    return success();
  }

private:
  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or not.
  bool isDefinedByIntegerConstantOp(Value v) const {
    if (mlir::isa<BlockArgument>(v))
      return false;
    if (mlir::isa<IntegerType>(
            mlir::cast<ShapedType>(v.getType()).getElementType()) &&
        isDenseONNXConstant(v))
      return true;
    return false;
  }

  // A helper function to check whether an block argument is invariant to
  // iterations or not. By the definition of LoopOp, input block arguments are
  // shifted by 1 to the left in YieldOp. If a block argument is unchanged when
  // being shifted in YieldOp, then it is invariant to iterations.
  bool isInvariantBlockArg(Value v, Operation *yieldOp) const {
    return mlir::isa<BlockArgument>(v) &&
           (v ==
               yieldOp
                   ->getOperands()[mlir::cast<BlockArgument>(v).getArgNumber() -
                                   1]);
  }

  // A helper function to check whether a value is defined by ONNXConstantOp in
  // the same block or an invariant block argument.
  bool isIntConstantOrInvariantBlockArg(Value v, Operation *yieldOp) const {
    return ((mlir::isa<BlockArgument>(v) && isInvariantBlockArg(v, yieldOp)) ||
            (!mlir::isa<BlockArgument>(v) && isDefinedByIntegerConstantOp(v)));
  }

  // A helper function to check whether an block argument is updated by a Value
  // inside the loop or not.
  bool isUpdatedArgByValue(Value v, Value newV, Operation *yieldOp) const {
    return mlir::isa<BlockArgument>(v) &&
           (newV ==
               yieldOp
                   ->getOperands()[mlir::cast<BlockArgument>(v).getArgNumber() -
                                   1]);
  }

  // A helper function to get the value that is fed to an operation's argument.
  Value getFedValue(Value arg, Operation *op) const {
    return op->getOperands()[mlir::cast<BlockArgument>(arg).getArgNumber()];
  }

  // A helper function to get an integer constant from a value.
  int64_t getOneIntegerConstant(Value v) const {
    return onnx_mlir::getScalarValue<int64_t>(
        v.getDefiningOp<ONNXConstantOp>());
  }

  // A helper function to match the pattern of the given operation. It also
  // returns a constant value for the max trip count during the matching, which
  // is to avoid recomputing values in the rewriting phase.
  //
  // Pattern:
  // ```
  // ubValue = ONNXConstantOp() {value = ...}
  // startValue = ONNXConstantOp() {value = ...}
  // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
  //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
  //     stepValue = ONNXConstantOp() {value = ...}
  //     newCounterValue = ONNXAddOp(counterValue, stepValue).
  //     cond = LessOp(newCounterValue, ubValue)
  //     ONNXYieldOp (cond, ..., ubValue, ..., newCounterValue, ...)
  // ```
  std::pair<bool, Value> matchOp(
      PatternRewriter &rewriter, Location loc, ONNXLoopOp onnxLoopOp) const {
    OnnxBuilder onnx(rewriter, loc);
    Operation *loopOp = onnxLoopOp.getOperation();
    Value maxTripCountValue = loopOp->getOperands()[0];

    // The maximum trip count is a constant.
    if (!isDefinedByIntegerConstantOp(maxTripCountValue))
      return std::make_pair(false, maxTripCountValue);

    // Get the loop region.
    Region &loopBody = onnxLoopOp.getBody();
    // Make sure the region has only one block.
    if (!loopBody.hasOneBlock())
      return std::make_pair(false, maxTripCountValue);

    // Get YieldOp of the body block.
    Block &bodyBlock = loopBody.front();
    Operation *yieldOp = bodyBlock.getTerminator();
    if (!isa<ONNXYieldOp>(yieldOp))
      return std::make_pair(false, maxTripCountValue);

    // Analyze the break condition of the loop body to see if we can derive a
    // new maximum trip count or not.

    // The break condition is the first argument of YieldOp.
    // `ONNXYieldOp (cond, ..., ubValue, ..., newCounterValue, ...)`
    Value breakCond = yieldOp->getOperands()[0];
    if (mlir::isa<BlockArgument>(breakCond))
      return std::make_pair(false, maxTripCountValue);
    Operation *breakCondOp = breakCond.getDefiningOp();

    // Only support LessOp as the op that defines the break condition at this
    // moment.
    // `cond = LessOp(newCounterValue, ubValue)`
    if (!isa<ONNXLessOp>(breakCondOp))
      return std::make_pair(false, maxTripCountValue);
    Value newCounterValue = breakCondOp->getOperands()[0];
    Value ubValue = breakCondOp->getOperands()[1];
    // Input type of Less must be integer.
    if (!mlir::isa<IntegerType>(
            mlir::cast<ShapedType>(newCounterValue.getType()).getElementType()))
      return std::make_pair(false, maxTripCountValue);

    // Compute a trip count from the break condition, given that the upper bound
    // is fixed and the lower bound is increased by a constant step at each
    // iteration. So, the trip count will be `(upper_bound - lower_bound)/step`.

    // Only support ONNXAddOp at this moment.
    if (mlir::isa<BlockArgument>(newCounterValue) ||
        !isa<ONNXAddOp>(newCounterValue.getDefiningOp()))
      return std::make_pair(false, maxTripCountValue);
    // ONNXLoop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     stepValue = ONNXConstantOp() {value = ...}
    //     newCounterValue = ONNXAddOp(counterValue, stepValue).
    //     cond = LessOp(newCounterValue, ubValue)
    //     ONNXYieldOp (cond, ..., ubValue, ..., newCounterValue, ...)
    Operation *addOp = mlir::cast<ONNXAddOp>(newCounterValue.getDefiningOp());
    Value counterValue = addOp->getOperands()[0];
    Value stepValue = addOp->getOperands()[1];
    // Counter is a block argument and updated at each iteration.
    if (!isUpdatedArgByValue(counterValue, newCounterValue, yieldOp))
      return std::make_pair(false, maxTripCountValue);
    // Step must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(stepValue, yieldOp))
      return std::make_pair(false, maxTripCountValue);

    // Check the lower bound of the break condition.
    // LowerBound is the initial value of the counter.
    Value lbValue = getFedValue(counterValue, loopOp);

    // Check the upper bound of the break condition.
    // UpperBound must be a constant inside the loop or an invariant argument.
    if (!isIntConstantOrInvariantBlockArg(ubValue, yieldOp))
      return std::make_pair(false, maxTripCountValue);

    // Get values for upper bound and step if they are invariant arguments.
    // Otherwise, clone them to location outside the loop.
    if (isInvariantBlockArg(ubValue, yieldOp))
      ubValue = getFedValue(ubValue, loopOp);
    else
      ubValue =
          mlir::cast<ONNXConstantOp>(rewriter.clone(*ubValue.getDefiningOp()))
              .getResult();
    if (isInvariantBlockArg(stepValue, yieldOp))
      stepValue = getFedValue(stepValue, loopOp);
    else
      stepValue =
          mlir::cast<ONNXConstantOp>(rewriter.clone(*stepValue.getDefiningOp()))
              .getResult();

    // Case 1: the upper bound, lower bound and step are constants.
    // - Compute the new max trip count at the compile time.
    if (isDefinedByIntegerConstantOp(lbValue) &&
        isDefinedByIntegerConstantOp(ubValue) &&
        isDefinedByIntegerConstantOp(stepValue)) {
      int64_t lowerBound = getOneIntegerConstant(lbValue);
      int64_t upperBound = getOneIntegerConstant(ubValue);
      int64_t step = getOneIntegerConstant(stepValue);
      if ((step <= 0) || (upperBound <= lowerBound))
        return std::make_pair(false, maxTripCountValue);
      int64_t derivedTripCount =
          ceil((1.0 * (upperBound - lowerBound)) / (1.0 * step));
      int64_t maxTripCount = getOneIntegerConstant(maxTripCountValue);

      // Check that the new trip count is smaller than the original trip count.
      if (maxTripCount <= derivedTripCount)
        return std::make_pair(false, maxTripCountValue);

      SmallVector<int64_t, 1> values(1, derivedTripCount);
      DenseElementsAttr valueAttr = DenseElementsAttr::get(
          RankedTensorType::get(
              {}, mlir::cast<ShapedType>(maxTripCountValue.getType())
                      .getElementType()),
          ArrayRef(values));
      return std::make_pair(true, onnx.constant(valueAttr));
    }

    // Case 2: Not all of the lower bound, upper bound and step are constants,
    // emit code to compute the new max trip count.
    // - new_max_trip_count =
    //      min(old_max_trip_count, ceil(upper_bound - lower_bound)/step)
    TypeAttr tripCountType = TypeAttr::get(
        mlir::cast<ShapedType>(maxTripCountValue.getType()).getElementType());

    // Cast the upper and lower bounds to the correct type.
    if (mlir::cast<ShapedType>(maxTripCountValue.getType()).getElementType() !=
        mlir::cast<ShapedType>(ubValue.getType()).getElementType())
      ubValue = onnx.cast(ubValue, tripCountType);
    if (mlir::cast<ShapedType>(maxTripCountValue.getType()).getElementType() !=
        mlir::cast<ShapedType>(lbValue.getType()).getElementType())
      lbValue = onnx.cast(lbValue, tripCountType);

    // Emit code to compute the max trip count.
    Value range = onnx.sub(ubValue, lbValue);
    Value rangeInFloat = onnx.cast(range, TypeAttr::get(rewriter.getF32Type()));
    Value stepInFloat =
        onnx.cast(stepValue, TypeAttr::get(rewriter.getF32Type()));
    Value tripCountInFloat = onnx.ceil(onnx.div(rangeInFloat, stepInFloat));
    Value newMaxTripCountValue = onnx.cast(tripCountInFloat, tripCountType);

    return std::make_pair(
        true, onnx.min(ValueRange({maxTripCountValue, newMaxTripCountValue})));
  }
};

// =============================================================================
// Rewrite pattern for RNNs
// =============================================================================

namespace {
// RNNOpRewriteLayoutPattern helper functions and classes.

template <typename ONNXOp>
void inferShapes(ONNXOp op) {
  if (failed(op.inferShapes([](Region &region) {})))
    llvm_unreachable("unexpected inferShapes failure");
}

// To transpose between [batch_size, seq_length/num_directions, size]
//                  and [seq_length/num_directions, batch_size, size].
ArrayAttr perm3RNN(Builder &b) { return b.getI64ArrayAttr({1, 0, 2}); }

// To transpose from [seq_length, num_directions, batch_size, hidden_size]
//                to [batch_size, seq_length, num_directions, hidden_size].
ArrayAttr perm4RNN(Builder &b) { return b.getI64ArrayAttr({2, 0, 1, 3}); }

class InputOutputTransposer {
public:
  InputOutputTransposer(OpBuilder &b, Location loc) : create(b, loc) {}

  void transposeInput(MutableOperandRange operand, ArrayAttr perm) {
    assert(operand.size() == 1 && "should be called with singleton range");
    Value input = operand[0].get();
    if (!mlir::isa<NoneType>(input.getType())) {
      Value transposed = transpose(input, perm);
      operand.assign(transposed);
    }
  }

  void transposeOutput(Value output, ArrayAttr perm) {
    if (!mlir::isa<NoneType>(output.getType())) {
      Value transposed = transpose(output, perm);
      output.replaceAllUsesExcept(transposed, transposed.getDefiningOp());
    }
  }

private:
  // Helper to create an ONNX transposition, using
  // ONNXTransposeOp::inferShapes() to infer the output shape.
  Value transpose(Value input, ArrayAttr perm) {
    Type elType = onnx_mlir::getElementType(input.getType());
    Type unrankedType = UnrankedTensorType::get({elType}); // placeholder
    Value transposed = create.transpose(unrankedType, input, perm);
    auto transposeOp = llvm::cast<ONNXTransposeOp>(transposed.getDefiningOp());
    inferShapes(transposeOp); // sets transposed's shape
    return transposed;
  }

  onnx_mlir::OnnxBuilder create;
};
} // namespace

// Rewrites layout=1 to layout=0 by transposing inputs and outputs.
template <typename ONNXOp>
class RNNOpRewriteLayoutPattern : public OpRewritePattern<ONNXOp> {
public:
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp onnxOp, PatternRewriter &rewriter) const override {
    if (onnxOp.getLayout() == 0) {
      return failure();
    }

    InputOutputTransposer transposer(rewriter, onnxOp.getLoc());
    ArrayAttr perm3 = perm3RNN(rewriter);

    // LSTM requires extra work for initial_c input and Y_c output.
    auto onnxLSTMOp = llvm::dyn_cast<ONNXLSTMOp>(*onnxOp);

    // Rewrite in-place because there are so many attributes, inputs, outputs.
    // Constructing a new op would be lengthy and hard to maintain.
    rewriter.modifyOpInPlace(onnxOp, [&]() {
      // Transpose the X and initial_h inputs by inserting an ONNXTransposeOp
      // before each and replacing the each input with the transpose output.
      rewriter.setInsertionPoint(onnxOp); // insert before (redundant)
      transposer.transposeInput(onnxOp.getXMutable(), perm3);
      transposer.transposeInput(onnxOp.getInitialHMutable(), perm3);
      if (onnxLSTMOp)
        transposer.transposeInput(onnxLSTMOp.getInitialCMutable(), perm3);
      // Set layout to zero.
      onnxOp->setAttr(onnxOp.getLayoutAttrName(),
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, /*isSigned=*/true), 0));
      // Update the output shape. Since the onnxOp is reused, it potentially had
      // some shape inference for its output. But since the input changed, we
      // don't want these now-erroneous output shapes to influence the output of
      // the revised op (as current output shape is used to potentially refine
      // existing shape inference). Long story short, we must reset the output
      // shapes. The call below does that. It is then safe to call shape
      // inference with the revised inputs.
      resetTypesShapeToQuestionmarks(onnxOp);
      inferShapes(onnxOp);
    });
    // Transpose the Y and Y_h outputs by inserting an ONNXTransposeOp
    // after each and replace all uses of each with the transpose output.
    ValueRange results = onnxOp.getResults();
    if (results.size() > 0) {
      rewriter.setInsertionPointAfter(onnxOp);
      transposer.transposeOutput(onnxOp.getY(), perm4RNN(rewriter));
      transposer.transposeOutput(onnxOp.getYH(), perm3);
      if (onnxLSTMOp)
        transposer.transposeOutput(onnxLSTMOp.getYC(), perm3);
    }

    return success();
  }
};

// Rewrites sequence_lens from tensor<bsxi32> to none when bs = 1. It works
// because by definition all batches (meaning one) has the same sequence length.
// This rewrite helps the compiler not need to handle sequence_lens.
template <typename ONNXOp>
class RNNOpRewriteSeqLenPattern : public OpRewritePattern<ONNXOp> {
public:
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp onnxOp, PatternRewriter &rewriter) const override {
    Operation *op = onnxOp.getOperation();
    Location loc = ONNXLoc<ONNXOp>(op);
    Value X = onnxOp.getX();
    Value initialH = onnxOp.getInitialH();
    Value seqLen = onnxOp.getSequenceLens();

    // sequence_lens is already none. Pattern does not match.
    if (isNoneValue(seqLen))
      return failure();

    // Check if batchsize is 1. Batchsize can be in:
    // - X: [seq_length, batch_size, input_size],
    // - intial_h: [num_directions, batch_size, hidden_size]
    // - sequence_lens: [batch_size], or
    bool oneInX = false, oneInSeqLen = false, oneInInitalH = false;
    if (isRankedShapedType(X.getType())) {
      ArrayRef<int64_t> shape = getShape(X.getType());
      oneInX = shape[1] == 1;
    }
    if (isRankedShapedType(seqLen.getType())) {
      ArrayRef<int64_t> shape = getShape(seqLen.getType());
      oneInSeqLen = (shape.size() == 1) && (shape[0] == 1);
    }
    if (!isNoneValue(initialH) && isRankedShapedType(initialH.getType())) {
      ArrayRef<int64_t> shape = getShape(initialH.getType());
      oneInInitalH = shape[1] == 1;
    }
    if (!oneInX && !oneInInitalH && !oneInSeqLen)
      return failure();

    // We know batchsize is 1. Rewrite now.
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    // Find the operand index of sequence_lens and update it with none.
    bool updated = false;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      if (op->getOperand(i) != seqLen)
        continue;
      op->setOperand(i, create.onnx.none());
      updated = true;
      break;
    }
    return updated ? success() : failure();
  }
};

// =============================================================================
// Rewrite pattern for Power
// =============================================================================

class PowToMulRewritePattern : public OpRewritePattern<ONNXPowOp> {
public:
  using OpRewritePattern<ONNXPowOp>::OpRewritePattern;

  PowToMulRewritePattern(MLIRContext *context, int64_t maxPower)
      : OpRewritePattern(context), maxPower(maxPower) {}

  LogicalResult matchAndRewrite(
      ONNXPowOp powOp, PatternRewriter &rewriter) const override {
    Operation *op = powOp.getOperation();
    Location loc = powOp.getLoc();
    int64_t exponent;
    // Test legality
    if (!CanExpandPowOpToMul(powOp, exponent))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value input = powOp.getX();

    Value result = nullptr;
    ShapedType resultType = mlir::cast<ShapedType>(powOp.getZ().getType());
    Type elementType = getElementType(resultType);
    if (exponent == 0) {
      Attribute one =
          isa<FloatType>(elementType)
              ? static_cast<Attribute>(rewriter.getFloatAttr(elementType, 1.0))
              : static_cast<Attribute>(rewriter.getIntegerAttr(elementType, 1));
      result = create.onnx.constant(DenseElementsAttr::get(resultType, one));
    } else {
      // calculate pow(input,exponent) with "exponentiation by squaring" method
      while (true) {
        if (exponent & 1)
          result = result ? create.onnx.mul(resultType, result, input) : input;
        exponent >>= 1;
        if (exponent == 0)
          break;
        input = create.onnx.mul(resultType, input, input);
      }
      assert(result && "should have a result here");
    }

    rewriter.replaceOp(op, {result});
    return success();
  };

private:
  // Check if a Pow can be simply rewritten as a sequence of multiply ops.
  bool CanExpandPowOpToMul(ONNXPowOp op, int64_t &powVal) const {
    return (hasIntegerPowerExponent(&op, powVal) && powVal >= 0 &&
            powVal <= maxPower);
  }
  // Data.
  int64_t maxPower;
};

// Rewrite a pattern like the following:
//
// %shape = onnx.Concat(%dim1, %dim2)
// %data = onnx.Expand(%input, %shape)
// %u = "onnx.Unsqueeze"(%data, %axes)
//
// into
//
// %new_shape = onnx.Concat(%dim1, %dim2, 1)
// %u = onnx.Expand(%input, %new_shape)
class ReplaceUnsqueezeOfExpandRewritePattern
    : public OpRewritePattern<ONNXUnsqueezeOp> {
public:
  using OpRewritePattern<ONNXUnsqueezeOp>::OpRewritePattern;

  ReplaceUnsqueezeOfExpandRewritePattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(
      ONNXUnsqueezeOp unsqueezeOp, PatternRewriter &rewriter) const override {
    Operation *op = unsqueezeOp.getOperation();
    Location loc = unsqueezeOp.getLoc();
    Value data = unsqueezeOp.getData();
    Value axes = unsqueezeOp.getAxes();

    // Match
    // 1. data is from ExpandOp, axes is from ConstantOp.
    if (!definedBy<ONNXExpandOp>(data) || !definedBy<ONNXConstantOp>(axes))
      return failure();
    auto expandOp = mlir::cast<ONNXExpandOp>(data.getDefiningOp());
    // 2. ExpandOp's input is a scalar tensor so that it's safe to use a new
    // shape that do not violate the broadcasting rule..
    if (!isScalarTensor(expandOp.getInput()))
      return failure();
    // 3. ExpandOp's shape is defined by dimensions.
    if (!areDims(expandOp.getShape()))
      return failure();

    // Rewrite
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    // Get the old shape.
    SmallVector<Value, 4> oldDims;
    getDims(expandOp.getShape(), oldDims);
    int64_t oldRank = oldDims.size();
    // Get unsqueeze axes.
    ElementsAttr axesAttrs = getElementAttributeFromONNXValue(axes);
    SmallVector<int64_t> axesI64(axesAttrs.getValues<int64_t>());
    for (unsigned int i = 0; i < axesI64.size(); ++i)
      if (axesI64[i] < 0)
        axesI64[i] += oldRank;

    // Construct a new shape.
    SmallVector<Value, 4> newDims;
    int64_t newRank = oldRank + axesI64.size();
    Value one = create.onnx.constantInt64(ArrayRef<int64_t>({1}));
    for (int64_t i = 0, j = 0; i < newRank || j < oldRank; ++i)
      if (std::find(axesI64.begin(), axesI64.end(), i) != axesI64.end())
        // found i in unsqueeze axes.
        newDims.emplace_back(one);
      else
        // original axes.
        newDims.emplace_back(oldDims[j++]);
    Value newShape = create.onnx.concat(
        RankedTensorType::get({newRank}, rewriter.getI64Type()), newDims, 0);

    Value res = create.onnx.expand(
        op->getResult(0).getType(), expandOp.getInput(), newShape);
    rewriter.replaceOp(op, {res});
    return success();
  };
};

/// The pattern is to replace two consecutive ReshapeOp with a single ReshapeOp.
/// It's not successful for arbitrary ReshapeOp, so let's consider necessary
/// condition for the replacement.
///
/// We would like to replace:
/// ```
// %0 = onnx.Reshape(%X, %shape1) {allowzero}
// %1 = onnx.Reshape(%0, %shape2) {allowzero}
// ```
// with
// ```
// %0 = onnx.Reshape(%X, %new_shape) {allowzero}
// ```
// where `%new_shape` is computed from `%shape1` and `%shape2` if possible.
//
// We only consider `allowzero=0` in this pattern.
//
// # Shape conditions
//
// According to ONNX specification for Reshape
// (https://onnx.ai/onnx/operators/onnx__Reshape.html#):
// - At most one dimension of the new shape can be -1. In this case, the value
// is inferred from the size of the tensor and the remaining dimensions
// - Dimension could also be 0. In this case,
//   - if allowzero = 0, the actual dimension value is unchanged;
//   - if allowzero = 1, the dimension will be set explicitly to zero.
// - If allowzero = 1, it is invalid for the specified shape to contain both a
// zero value and -1
//
// # Combining rules
//
// In this pattern, we use the following terms for values in a shape tensor:
// 0, -1, and L (a literal).
//
// These are the rules to combine two values:
//  (1st)  : (2nd)  => (result)
//   0     : 0      => 0
//   0     : L      => L
//   0     : -1     => -1
//
//  -1     : 0      => -1
//  -1     : L      => L
//  -1     : -1     => -1
//
//   L     : 0      => L
//   L     : L      => L
//   L     : -1     => -1
//
// To produce a new shape, we combine each value one by one from left to right.
//
// Example (allowzero = 0):
// Ex1. 1st: [0, -1, 0, 5], 2nd: [0, -1, 0] => [0, -1, 0]
// Ex2. 1st: [0, -1, 0, 5], 2nd: [5, -1, 0] => [5, -1, 0]
// Ex3. 1st: [0, -1, 0, 5], 2nd: [-1, 0, 0] => [-1, -1, 0]
// Ex4. 1st: [0, -1, 0, 5], 2nd: [0, 0, 5] => [0, -1, 5]
// Ex5. 1st: [0, -1, 5, 0], 2nd: [-1, 5, 0] => [-1, 5, 5]
//
// After combining two shapes, we check if the result shape is valid or not
// according to the shape conditions. If it is invalid, the two ReshapeOps are
// not combined. For example, the output shape in Ex3 is invalid because of two
// -1s.
//
class FuseTwoReshapesPattern : public OpRewritePattern<ONNXReshapeOp> {
public:
  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  FuseTwoReshapesPattern(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(
      ONNXReshapeOp secondReshapeOp, PatternRewriter &rewriter) const override {
    // Second Reshape.
    Operation *op = secondReshapeOp.getOperation();
    Value secondData = secondReshapeOp.getData();
    Value secondShape = secondReshapeOp.getShape();
    int64_t secondAllowZero = secondReshapeOp.getAllowzero();
    if (secondAllowZero != 0)
      return rewriter.notifyMatchFailure(op, "Does not support AllowZero != 0");

    // First Reshape.
    if (!definedBy<ONNXReshapeOp>(secondData))
      return rewriter.notifyMatchFailure(
          op, "The input data is not defined by a Reshape");
    auto firstReshapeOp = secondData.getDefiningOp<ONNXReshapeOp>();
    Value firstData = firstReshapeOp.getData();
    Value firstShape = firstReshapeOp.getShape();
    int64_t firstAllowZero = firstReshapeOp.getAllowzero();
    if (firstAllowZero != 0)
      return rewriter.notifyMatchFailure(op, "Does not support AllowZero != 0");

    Location loc = rewriter.getFusedLoc(
        {firstReshapeOp.getLoc(), secondReshapeOp.getLoc()});
    OnnxBuilder createONNX(rewriter, loc);

    // Try to compute a new shape tensor by fusing the two old shapes.
    SmallVector<Value, 4> firstDims, secondDims, fusedDims;
    if (!getValuesFromShape(createONNX, firstShape, firstDims) ||
        !getValuesFromShape(createONNX, secondShape, secondDims)) {
      // Not rewrite if we can not read dimension values (0, -1, L) from a shape
      // tensor.
      return rewriter.notifyMatchFailure(
          op, "Cannot read invididual dimensions");
    }

    // Iterate over the second shape that is similar to the output shape.
    int64_t s1 = firstDims.size();
    int64_t s2 = secondDims.size();
    uint64_t minusOnes = 0;
    for (int64_t i = 0; i < s2; ++i) {
      Value fusedD;
      if (i < s1) {
        // Fuse two dimensions.
        // These are the rules to combine two values:
        //  (1st)  : (2nd)  => (result)
        //   0     : 0      => 0
        //   0     : L      => L
        //   0     : -1     => -1
        //
        //  -1     : 0      => -1
        //  -1     : L      => L
        //  -1     : -1     => -1
        //
        //   L     : 0      => L
        //   L     : L      => L
        //   L     : -1     => -1
        Value d1 = firstDims[i];
        Value d2 = secondDims[i];
        fusedD = isZero(d2) ? d1 : d2;
      } else {
        // 2nd shape has more dims than the 1st shape. Get dims from the 2nd
        // shape as they are.
        fusedD = secondDims[i];
      }
      fusedDims.emplace_back(fusedD);
      if (isMinusOne(fusedD))
        minusOnes++;
    }
    if (minusOnes > 1) {
      // The fused shape is invalid because it has two -1s.
      return rewriter.notifyMatchFailure(op, "Failed to compute a fused shape");
    }

    // Rewrite phase.
    // Emit the fused shape using ONNXConstantOp or ONNXConcatOp.
    Value fusedShape;
    if (llvm::all_of(
            fusedDims, [](Value v) { return isScalarConstantTensor(v); })) {
      SmallVector<int64_t> dims;
      for (int64_t i = 0; i < s2; ++i)
        getI64ValuesFromONNXConstantOp(fusedDims[i], dims);
      fusedShape = createONNX.constantInt64(ArrayRef<int64_t>(dims));
    } else {
      fusedShape =
          createONNX.concat(RankedTensorType::get({s2}, rewriter.getI64Type()),
              fusedDims, /*axis=*/0);
    }
    // Emit a new Reshape.
    Value res = createONNX.reshape(secondReshapeOp.getResult().getType(),
        firstData, fusedShape, secondReshapeOp.getAllowzeroAttr());

    rewriter.replaceOp(op, res);
    return success();
  };

private:
  bool isZero(Value v) const {
    SmallVector<int64_t> dims;
    if (getI64ValuesFromONNXConstantOp(v, dims))
      return (dims[0] == 0);
    return false;
  }

  bool isMinusOne(Value v) const {
    SmallVector<int64_t> dims;
    if (getI64ValuesFromONNXConstantOp(v, dims))
      return (dims[0] == -1);
    return false;
  }

  bool isLiteral(Value v) const {
    SmallVector<int64_t> dims;
    if (getI64ValuesFromONNXConstantOp(v, dims))
      return (dims[0] > 0);
    if (definedBy<ONNXDimOp>(v)) {
      // Runtime dimension of a value is always literal.
      return true;
    }
    return false;
  }

  // Get invididual values from a shape tensor. Return true if succeeded.
  // Otherwise, return false.
  bool getValuesFromShape(OnnxBuilder &createONNX, Value shape,
      SmallVectorImpl<Value> &values) const {
    // Shape is defined by a Concat.
    if (areDimsFromConcat(shape)) {
      getDims(shape, values);
      return true;
    }

    // Shape is defined by a Constant.
    SmallVector<int64_t> dims;
    if (getI64ValuesFromONNXConstantOp(shape, dims)) {
      for (int64_t d : dims) {
        Value dim = createONNX.constantInt64({d});
        values.emplace_back(dim);
      }
      return true;
    }

    return false;
  }
};

// =============================================================================
// Rewrite pattern LayerNormalization
// =============================================================================

template <typename OP_TYPE>
struct PropagateBiasIntoLayerNormRewritePattern
    : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  PropagateBiasIntoLayerNormRewritePattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Value y, bias;
    Operation *yLayerNormOp;
    Operation *ywbAddOp = addOp.getOperation();
    Location loc = addOp.getLoc();
    // Match
    // %noBias = "onnx.NoValue"()
    // %y, %mean, %invStdDev = "onnx.LayerNormalization"(%x, %scale, %noBias)
    //     {axis = 2 : si64, epsilon = 9.994E-6 : f32, stash_type = 1 : si64}
    // %yBias = "onnx.Add"(%y, %bias)
    if (!onnx_mlir::operandOfOpDefinedBy<OP_TYPE>(
            yLayerNormOp, ywbAddOp, y, bias, 0) &&
        !onnx_mlir::operandOfOpDefinedBy<OP_TYPE>(
            yLayerNormOp, ywbAddOp, bias, y, 1))
      return reportFailure("missing y, layer norm op");
    // Study layer norm op; make sure its used only one and that bias is not
    // used.
    if (!yLayerNormOp->hasOneUse())
      return reportFailure("y/layer norm has too many uses");
    auto lnOp = mlir::cast<OP_TYPE>(yLayerNormOp);
    if (!onnx_mlir::isNoneValue(lnOp.getB()))
      return reportFailure("layer norm already has a bias");
    // We are fine.
    Value x = lnOp.getX();
    Value scale = lnOp.getScale();
    FloatAttr epsilon = lnOp.getEpsilonAttr();
    int64_t axis = lnOp.getAxis();
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm from add, axis : " << axis << "\n");

    // Replace
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Type xType = x.getType();
    Value res;
    if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value)
      res = create.onnx.layerNorm(xType, x, scale, bias, axis, epsilon);
    else if constexpr (std::is_same<OP_TYPE,
                           ONNXRMSLayerNormalizationOp>::value)
      res = create.onnx.RMSLayerNorm(xType, x, scale, bias, axis, epsilon);
    else
      llvm_unreachable("unsupported op");
    rewriter.replaceOp(addOp, res);
    return success();
  }

private:
  LogicalResult reportFailure(std::string msg) const {
    // Can disable line below if not needed.
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm failure:" << msg << "\n");
    return failure();
  }
};

// =============================================================================
// Rewrite pattern for Where
// =============================================================================

class RemoveWhereEqualPattern : public OpRewritePattern<ONNXWhereOp> {
public:
  using OpRewritePattern<ONNXWhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXWhereOp onnxWhereOp, PatternRewriter &rewriter) const override {
    Location loc = onnxWhereOp.getLoc();
    onnx_mlir::OnnxBuilder create(rewriter, loc);
    // Check operation pattern:
    // (ONNXWhereOp
    //     (ONNXEqualOp (ONNXConcatOp), (ONNXConstantOp)),
    //      (ONNXConstantOp),
    //      (ONNXConcatOp))
    // - The second input of EqualOp need to be all negative values.
    // - The output need to be integer type.
    // - Has shape and rank.
    // - DefiningOp of operands of ONNXConcatOp need to be DimOp or ConstantOp
    // with scalar tensor
    // - Operands in ONNXConcatOp need to be DimOp or ConstantOp

    // Check if the condition of WhereOp matches EqualOp, the X of it matches
    // ConstantOp, and the Y of it matches ConcatOp.
    Operation *equalOp, *constantOp, *concatOp;
    Value equalOpResVal, constantOpResVal, concatOpResVal;
    bool isEqualOp = operandOfOpDefinedBy<ONNXEqualOp>(
        equalOp, onnxWhereOp.getOperation(), equalOpResVal, 0);
    bool isConstantOp = operandOfOpDefinedBy<ONNXConstantOp>(
        constantOp, onnxWhereOp.getOperation(), constantOpResVal, 1);
    bool isConcatOp = operandOfOpDefinedBy<ONNXConcatOp>(
        concatOp, onnxWhereOp.getOperation(), concatOpResVal, 2);
    if (!isEqualOp || !isConstantOp || !isConcatOp)
      return failure();
    // Check if operands of the EqualOp are ConcatOp and ConstantOp.
    Value equalOpConstVal, equalOpConcatVal;
    bool isConcatAndConstOp =
        areDefinedBy<ONNXConcatOp, ONNXConstantOp>(equalOp->getOperand(0),
            equalOp->getOperand(1), equalOpConcatVal, equalOpConstVal);
    if (!isConcatAndConstOp)
      return failure();

    if (!hasShapeAndRank(equalOpConcatVal) ||
        !hasShapeAndRank(equalOpConstVal) || !hasShapeAndRank(concatOpResVal)) {
      return failure(); // Cannot apply pattern until ranks are known.
    }

    if (!isAllNegativeSmallIntegerConstant(equalOpConstVal))
      return failure();

    // Get attribute of constantOp, an operand of equal op (Negative values)
    SmallVector<int64_t> constAttrValues;
    if (!getI64ValuesFromONNXConstantOp(equalOpConstVal, constAttrValues))
      return failure();
    // Get attriubte of concatOp, an operand of equal op, and calculate the
    // result of the equalOp
    ValueRange concatOperands = concatOp->getOperands();
    llvm::SmallVector<bool, 1> equalOpResults;
    for (uint64_t i = 0; i < concatOperands.size(); ++i) {
      // Block arguments.
      if (mlir::isa<BlockArgument>(concatOperands[i]))
        return failure();
      if (concatOperands[i].getDefiningOp<ONNXDimOp>()) {
        // The value defined by DimOp is not negative value. So, results is
        // always false.
        equalOpResults.emplace_back(false);
      } else if (isDenseONNXConstant(concatOperands[i]) &&
                 isScalarTensor(concatOperands[i])) {
        // Compare the attributes to create results of the EqualOp.
        SmallVector<int64_t> concatAttrValues;
        if (!getI64ValuesFromONNXConstantOp(
                concatOperands[i], concatAttrValues))
          return failure();
        int64_t a = concatAttrValues.front();
        int64_t b = constAttrValues[i];
        equalOpResults.emplace_back(a == b);
      } else {
        return failure();
      }
    }
    // Create new concatOp by selecting X or Y of whereOp depending on the
    // result of equalOp.
    SmallVector<int64_t> valueX;
    if (!getI64ValuesFromONNXConstantOp(constantOpResVal, valueX))
      return failure();
    SmallVector<Value, 4> resVals;
    for (uint64_t i = 0; i < equalOpResults.size(); ++i) {
      if (equalOpResults[i]) {
        // ConstOp in X of WhereOp
        resVals.emplace_back(create.constantInt64({valueX[i]}));
      } else {
        // ConcatOp in Y of WhereOp
        resVals.emplace_back(concatOperands[i]);
      }
    }
    Value replacingValue = onnxWhereOp.getResult();
    ShapedType replacingType = mlir::cast<ShapedType>(replacingValue.getType());
    Value res = create.concat(replacingType, ValueRange(resVals), /*axis*/ 0);
    rewriter.replaceOp(onnxWhereOp, res);
    return success();
  }
};

// =============================================================================
/// Register optimization patterns as "canonicalization" patterns.
/// Add op to OpsWithCanonicalizer in gen_onnx_mlir.py to activate.
/// Please keep in alphabetical order.
// =============================================================================

/// on the ONNXBatchNormalizationInferenceModeOp.
void ONNXBatchNormalizationInferenceModeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FuseBatchNormInferenceModeConvPattern>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern1>(context);
  results.insert<RewriteBatchNormInferenceModeConvPattern2>(context);
}

/// on the ONNXAddOp.
void ONNXAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeAddPattern>(context);
  results.insert<MulAddToGemmOptPattern>(context);
  results.insert<FuseGemmFollowedByAddition>(context);
  results.insert<FuseAddConvPattern>(context);
  results.insert<FuseAddConvNullBiasPattern>(context);
  results.insert<BinaryOpBroadcastAxisPattern<ONNXAddOp>>(context);
  results.insert<PropagateScalarConstantExpandPattern<ONNXAddOp>>(context);
  results.insert<
      PropagateBiasIntoLayerNormRewritePattern<ONNXLayerNormalizationOp>>(
      context);
  results.insert<
      PropagateBiasIntoLayerNormRewritePattern<ONNXRMSLayerNormalizationOp>>(
      context);
  results.insert<PropagateReshapeThroughBinaryOpPattern<ONNXAddOp>>(context);
}

/// on the ONNXAndOp.
void ONNXAndOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXAndOp>>(context);
}

/// on the ONNXCastOp.
void ONNXCastOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<CastEliminationPattern>(context);
  result.insert<SwapCastConcatPattern>(context);
  result.insert<SwapCastSlicePattern>(context);
  // TODO: Reintroduce pattern for sound type combinations, see issue #2210.
  // result.insert<FuseCastCastPattern>(context);
}

/// on the ONNXConstantOp.
void ONNXConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {}

/// on the ONNXDepthToSpaceOp.
void ONNXDepthToSpaceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDepthToSpaceSpaceToDepthPattern>(context);
}

/// on the ONNXDivOp.
void ONNXDivOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXDivOp>>(context);
  result.insert<PropagateScalarConstantExpandPattern<ONNXDivOp>>(context);
  result.insert<PropagateReshapeThroughBinaryOpPattern<ONNXDivOp>>(context);
  result.insert<PropagateConstantScalingInAttentionLayerPattern<ONNXDivOp>>(
      context);
}

/// on the ONNXDropoutOp.
void ONNXDropoutOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<DropoutEliminationPattern>(context);
}

/// on the ONNXDimOp.
void ONNXDimOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DimOpToConstantPattern>(context);
}

/// on the ONNXEqualOp.
void ONNXEqualOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXEqualOp>>(context);
}

/// on the ONNXGlobalAveragePoolOp.
void ONNXGlobalAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalAveragePoolPattern>(context);
}

/// on the ONNXGlobalMaxPoolOp.
void ONNXGlobalMaxPoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<GlobalMaxPoolPattern>(context);
}

/// on the ONNXGreaterOp.
void ONNXGreaterOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXGreaterOp>>(context);
}

/// on the ONNXGRUOp.
void ONNXGRUOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXGRUOp>>(context);
  results.insert<RNNOpRewriteSeqLenPattern<ONNXGRUOp>>(context);
}

/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXLayoutTransformOp.
void ONNXLayoutTransformOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ONNXLayoutTransformEliminationPattern>(context);
  result.insert<ONNXLayoutTransformFusionPattern>(context);
}

/// on the ONNXLessOp.
void ONNXLessOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LessOpSameCastPattern>(context);
  results.insert<BinaryOpBroadcastAxisPattern<ONNXLessOp>>(context);
}

/// on the ONNXLoopOp.
void ONNXLoopOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LoopOpRewriteMaxTripCountPattern>(context);
}

/// on the ONNXLSTMOp.
void ONNXLSTMOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXLSTMOp>>(context);
  results.insert<RNNOpRewriteSeqLenPattern<ONNXLSTMOp>>(context);
}

/// on the ONNXMulOp.
void ONNXMulOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<NormalizeMulPattern>(context);
  results.insert<FuseMulConvNullBiasPattern>(context);
  results.insert<BinaryOpBroadcastAxisPattern<ONNXMulOp>>(context);
  results.insert<PropagateScalarConstantExpandPattern<ONNXMulOp>>(context);
  results.insert<PropagateReshapeThroughBinaryOpPattern<ONNXMulOp>>(context);
  results.insert<PropagateConstantScalingInAttentionLayerPattern<ONNXMulOp>>(
      context);
}

/// on the ONNXOrOp.
void ONNXOrOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXOrOp>>(context);
}

/// on the ONNXReshapeOp.
void ONNXReshapeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseTwoReshapesPattern>(context);
  result.insert<FuseTwoReshapesAllowZeroPattern>(context);
  result.insert<RemoveIdentityReshapePattern1>(context);
  result.insert<RemoveIdentityReshapePattern2>(context);
  result.insert<SwapReshapeMatMulPattern>(context);
}

/// on the ONNXResizeOp.
void ONNXResizeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<EmptyTensorInputsResizePattern>(context);
}

/// on the ONNXRNNOp.
void ONNXRNNOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RNNOpRewriteLayoutPattern<ONNXRNNOp>>(context);
  results.insert<RNNOpRewriteSeqLenPattern<ONNXRNNOp>>(context);
}

/// on the ONNXShapeOp.
void ONNXShapeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ShapeToConstantPattern>(context);
}

/// on the ONNXSubOp.
void ONNXSubOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXSubOp>>(context);
  result.insert<PropagateScalarConstantExpandPattern<ONNXSubOp>>(context);
  result.insert<PropagateReshapeThroughBinaryOpPattern<ONNXSubOp>>(context);
}

/// on ONNXShapeTransformOp
void ONNXShapeTransformOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ShapeTransformComposePattern>(context);
  results.insert<ShapeTransformIdentityPattern>(context);
}

/// on the ONNXSizeOp.
void ONNXSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SizeToConstantPattern>(context);
}

/// on the ONNXSoftmaxV11Op.
void ONNXSoftmaxV11Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SoftmaxV11ToLatestPattern>(context);
}

/// on the ONNXSpaceToDepthOp.
void ONNXSpaceToDepthOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveSpaceToDepthDepthToSpacePattern>(context);
}

/// on the ONNXSqueezeOp.
void ONNXSqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeUnsqueezePattern>(context);
  result.insert<RemoveSqueezeCastUnsqueezePattern>(context);
}

void ONNXSqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeV11UnsqueezeV11Pattern>(context);
  result.insert<RemoveSqueezeV11CastUnsqueezeV11Pattern>(context);
}

/// on the ONNXTileOp.
void ONNXTileOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveIdentityTilePattern>(context);
}

/// on the ONNXTransposeOp.
void ONNXTransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<FuseTransposePattern>(context);
  result.insert<FuseTransposeAndAtanPattern>(context);
  result.insert<FuseTransposeAndCastPattern>(context);
  result.insert<FuseTransposeAndCeilPattern>(context);
  result.insert<FuseTransposeAndCosPattern>(context);
  result.insert<FuseTransposeAndCoshPattern>(context);
  result.insert<FuseTransposeAndEluPattern>(context);
  result.insert<FuseTransposeAndErfPattern>(context);
  result.insert<FuseTransposeAndAcosPattern>(context);
  result.insert<FuseTransposeAndAcoshPattern>(context);
  result.insert<FuseTransposeAndAsinPattern>(context);
  result.insert<FuseTransposeAndAsinhPattern>(context);
  result.insert<FuseTransposeAndAtanhPattern>(context);
  result.insert<FuseTransposeAndExpPattern>(context);
  result.insert<FuseTransposeAndFloorPattern>(context);
  result.insert<FuseTransposeAndHardSigmoidPattern>(context);
  result.insert<FuseTransposeAndIsNaNPattern>(context);
  result.insert<FuseTransposeAndLeakyReluPattern>(context);
  result.insert<FuseTransposeAndLogPattern>(context);
  result.insert<FuseTransposeAndNegPattern>(context);
  result.insert<FuseTransposeAndNotPattern>(context);
  result.insert<FuseTransposeAndReciprocalPattern>(context);
  result.insert<FuseTransposeAndReluPattern>(context);
  result.insert<FuseTransposeAndRoundPattern>(context);
  result.insert<FuseTransposeAndSeluPattern>(context);
  result.insert<FuseTransposeAndSigmoidPattern>(context);
  result.insert<FuseTransposeAndSignPattern>(context);
  result.insert<FuseTransposeAndSinPattern>(context);
  result.insert<FuseTransposeAndSinhPattern>(context);
  result.insert<FuseTransposeAndSoftplusPattern>(context);
  result.insert<FuseTransposeAndSoftsignPattern>(context);
  result.insert<FuseTransposeAndSqrtPattern>(context);
  result.insert<FuseTransposeAndTanPattern>(context);
  result.insert<FuseTransposeAndTanhPattern>(context);
  result.insert<RemoveIdentityTransposePattern>(context);
  result.insert<SwapTransposeConcatPattern>(context);
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
  result.insert<RemoveUnsqueezeCastSqueezePattern>(context);
  result.insert<ReplaceUnsqueezeOfExpandRewritePattern>(context);
}

void ONNXUnsqueezeV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeV11SqueezeV11Pattern>(context);
  result.insert<RemoveUnsqueezeV11CastSqueezeV11Pattern>(context);
}

void ONNXPowOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  // Is 64 necessary? Maybe too high?
  result.insert<PowToMulRewritePattern>(context, 64);
  result.insert<BinaryOpBroadcastAxisPattern<ONNXPowOp>>(context);
}

/// on the ONNXXorOp.
void ONNXXorOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXXorOp>>(context);
}

// on the ONNXWhereOp.
void ONNXWhereOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<AlwaysFalseWherePattern>(context);
  result.insert<RemoveWhereEqualPattern>(context);
}

// on the ONNXDequantizeLinearOp.
void ONNXDequantizeLinearOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {}
