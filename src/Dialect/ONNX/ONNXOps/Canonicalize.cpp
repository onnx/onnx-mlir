/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRewrite.cpp - ONNX High Level Optimizer --------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
// Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
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

#include <functional>
#include <math.h>
#include <numeric>

#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "rewrite"

// Populated by configureBatchNormCanonicalization().
static bool disableBatchNormDecompose = false;

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

// Check if a value is a splat constant
static bool isSplatConstant(Value val) {
  auto constOp = val.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return false;

  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return false;

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr))
    return denseAttr.isSplat();

  return false;
}

// Create a reshaped constant for fusing into Conv weight multiplication.
//   1. For scalars: returns as-is.
//   2. For splats: creates scalar.
//   3. For per-output-channel: reshapes to [C_out, 1, 1, ...]
Value createReshapedConstantForWeightFusion(
    PatternRewriter &rewriter, Value constant, Value weight) {
  auto constantType = mlir::cast<ShapedType>(constant.getType());
  auto weightType = mlir::cast<ShapedType>(weight.getType());

  const int64_t numElements = constantType.getNumElements();
  const int64_t cOut = weightType.getShape()[0];
  const int64_t weightRank = weightType.getRank();

  // Case 1: Scalar (1 element) - return as-is
  if (numElements == 1) {
    return constant;
  }

  auto constOp = constant.getDefiningOp<ONNXConstantOp>();
  auto constOpLoc = constOp->getLoc();

  // Case 2: Splat constant - create a scalar constant with the splat value
  if (isSplatConstant(constant)) {
    auto denseAttr = mlir::cast<DenseElementsAttr>(constOp.getValueAttr());

    // Create a new scalar constant with the splat value
    auto elementType = constantType.getElementType();
    auto scalarType = RankedTensorType::get({}, elementType);
    auto splatValue = denseAttr.getSplatValue<Attribute>();
    auto scalarAttr = DenseElementsAttr::get(scalarType, splatValue);

    return rewriter.create<ONNXConstantOp>(constOpLoc, nullptr, scalarAttr);
  }

  // Case 3: Per-ouput-channel (C_out elements) - reshape to [C_out, 1, 1, ...]
  assert(cOut == numElements &&
         "For non-splat constants, numElements must equal C_out");

  SmallVector<int64_t> targetShape;
  targetShape.push_back(cOut);
  for (int i = 1; i < weightRank; ++i)
    targetShape.push_back(1);

  // Create shape constant
  Value shapeConst = rewriter.create<ONNXConstantOp>(
      constOpLoc, nullptr, rewriter.getI64TensorAttr(targetShape));

  // Create result type for reshape
  auto elementType = constantType.getElementType();
  auto resultType = RankedTensorType::get(targetShape, elementType);

  // Create and return the reshape op
  return rewriter.create<ONNXReshapeOp>(
      constOpLoc, resultType, constant, shapeConst);
}

// Check if constant to Mul has valid shape for folding into the weights of
// Conv. This is used for  FuseMulConvNullBiasPattern Valid cases:
//   1. Scalar (1 element)
//   2. Splat constant and only if Mul doesn't do broadcasting on spatial
//      dimensions (the last two dims)
//   3. Per-channel scaling: after right-aligning shapes for broadcasting,
//      the constant must be 1 on all dimensions except the channel dim (dim 1)
bool hasValidShapeForWeightFusion(
    Value constant, Value weight, Value mulResult) {
  auto constType = mlir::dyn_cast<ShapedType>(constant.getType());
  auto weightType = mlir::dyn_cast<ShapedType>(weight.getType());
  auto resultType = mlir::dyn_cast<ShapedType>(mulResult.getType());

  if (!constType || !weightType || !resultType)
    return false;
  if (!constType.hasRank() || !resultType.hasRank())
    return false;

  const int64_t numElements = constType.getNumElements();
  const int64_t cOut = weightType.getShape()[0];
  const int64_t resultRank = resultType.getRank();
  auto constShape = constType.getShape();
  const int64_t constRank = constType.getRank();

  // Case 1: Scalar (1 element)
  if (numElements == 1)
    return true;

  // Case 2: Splat constant (uniform value) - only if Mul doesn't change shape
  // If Mul does broadcasting (changes shape), don't fuse because we'd just
  // trade Mul for a Broadcast op, which is not a real optimization.
  if (isSplatConstant(constant)) {
    // Check if Mul changes shape by comparing Conv output with Mul result
    // We need the Conv output shape, which we can infer from the Mul inputs
    // For now, check if constant and result have compatible shapes
    // If they're the same shape, no broadcasting happens
    auto convOutput = mulResult.getDefiningOp()->getOperand(0);
    auto convType = mlir::dyn_cast<ShapedType>(convOutput.getType());
    if (!convType || !convType.hasRank())
      return false;

    // If Conv output shape != Mul result shape, Mul is doing broadcasting
    // In that case, don't fuse even for splats
    return convType.getShape() == resultType.getShape();
  }

  // Case 3: Per-channel scaling
  // After right-aligning shapes for broadcasting, the constant must be 1 on
  // all dimensions except the channel dimension (dim 1 in NCHW layout).
  // This ensures the constant only varies along the channel dimension.
  for (int64_t i = 0; i < resultRank; ++i) {
    // Right-align: find corresponding constant dimension
    int64_t constIdx = constRank - (resultRank - i);
    int64_t constDim = (constIdx >= 0) ? constShape[constIdx] : 1;

    if (i == 1) {
      // Channel dimension: can be 1 (broadcast) or C_out (per-channel)
      if (constDim != 1 && constDim != cOut)
        return false;
    } else {
      // Non-channel dimensions: MUST broadcast (must be 1)
      if (constDim != 1)
        return false;
    }
  }
  return true;
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

// Get return type for a MaxPoolOp assuming input is 4D NCHW.
Type getReturnTypeForMaxPool2D(Value input) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  return UnrankedTensorType::get(inputType.getElementType());
}

bool isNotConvProducer(mlir::Value val) {
  if (auto defOp = val.getDefiningOp()) {
    return !llvm::isa<mlir::ONNXConvOp>(defOp);
  }
  return true; // If no defining op, assume it's safe
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
    ONNXCastOp castOp = rewriter.create<ONNXCastOp>(
        loc, inpType.clone(to.getValue()), inp, saturate, to);
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

Value maxOrDefault(PatternRewriter &rewriter, Location loc, Value a, Value b) {
  // If A or B is NoneType, return the other value
  if (mlir::isa<NoneType>(a.getType()))
    return b;
  if (mlir::isa<NoneType>(b.getType()))
    return a;

  // Otherwise, return the max of A and B
  return rewriter.create<ONNXMaxOp>(loc, a.getType(), ValueRange{a, b});
}

Value minOrDefault(PatternRewriter &rewriter, Location loc, Value a, Value b) {
  // If A or B is NoneType, return the other value
  if (mlir::isa<NoneType>(a.getType()))
    return b;
  if (mlir::isa<NoneType>(b.getType()))
    return a;

  // Otherwise, return the min of A and B
  return rewriter.create<ONNXMinOp>(loc, a.getType(), ValueRange{a, b});
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

// Check if Reshape with allowzero == 1 can be replaced by
// another one with allowzero == 0. Conditions:
// - If no value in the 'shape' input is set to zero.
bool isConstantOpWithNoZeroElements(Value constVal) {
  if (!isDenseONNXConstant(constVal))
    return false;

  ONNXConstantOp constOp = constVal.getDefiningOp<ONNXConstantOp>();
  DenseElementsAttr intElemsAttr;
  if (auto elms =
          dyn_cast<mlir::DenseIntElementsAttr>(constOp.getValueAttr())) {
    intElemsAttr = elms;
  } else if (auto elms = dyn_cast<mlir::DisposableElementsAttr>(
                 constOp.getValueAttr())) {
    intElemsAttr = dyn_cast_or_null<mlir::DenseIntElementsAttr>(
        elms.toDenseElementsAttr());
  }
  if (!intElemsAttr)
    return false;

  auto isZero = [](int64_t val) { return val == 0; };

  return llvm::none_of(intElemsAttr.getValues<int64_t>(), isZero);
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

// This pattern bubbles up AddOp through transpose to keep the bias Add
// operation right after LN_type op. This will helps the other patterns fold the
// add into the operands of a Norm operator.
//
// From:
// Norm operator
//    |
// Transpose
//    |
//   Add
//
// To:
// Norm operator
//    |
//   Add
//    |
// Transpose
template <typename LN_TYPE>
class BubbleUpBiasForNormOpPattern : public OpRewritePattern<ONNXAddOp> {
public:
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &r) const override {
    if (!isConstLikeValue(addOp.getB()))
      return r.notifyMatchFailure(addOp, "not a constant rhs operand");

    auto transposeOp =
        llvm::dyn_cast_or_null<ONNXTransposeOp>(addOp.getA().getDefiningOp());
    if (!transposeOp)
      return r.notifyMatchFailure(addOp, "the producer is not a transpose");

    if (!transposeOp->hasOneUse())
      return r.notifyMatchFailure(
          addOp, "cannot bubble up because transpose has other user");

    auto layernormResult = transposeOp.getData();
    auto layerNorm =
        llvm::dyn_cast_or_null<LN_TYPE>(layernormResult.getDefiningOp());
    if (!layerNorm)
      return r.notifyMatchFailure(
          transposeOp, "the producer is not a layernorm");

    if (!isNoneValue(layerNorm.getB()))
      return r.notifyMatchFailure(layerNorm, "layernorm already has a bias");

    OnnxBuilder create(r, addOp.getLoc());

    auto perm = extractFromIntegerArrayAttr<int64_t>(transposeOp.getPermAttr());
    auto invertedPerm = invertPermutationVector(perm);
    auto cstReshaped = create.upRank(addOp.getB(), getRank(addOp.getType()));
    auto cstTranposed = create.transposeInt64(cstReshaped, invertedPerm);
    auto newAddOp = create.add(layernormResult, cstTranposed);
    auto transposedBack = create.transposeInt64(newAddOp, perm);

    r.replaceOp(addOp, transposedBack);

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
// Rewrite pattern for redundant resize (scale=1 or same input/output size)
// =============================================================================
//
// A resize with equal input and output dimensions is a noop.
// This assumes coordinate transformation mode is not "tf_crop_and_resize".
class RemoveRedundantResizePattern : public OpRewritePattern<ONNXResizeOp> {
public:
  using OpRewritePattern<ONNXResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXResizeOp onnxResizeOp, PatternRewriter &rewriter) const override {

    auto inputType =
        mlir::dyn_cast<RankedTensorType>(onnxResizeOp.getX().getType());
    auto outputType = mlir::dyn_cast<RankedTensorType>(onnxResizeOp.getType());

    if (!inputType || !outputType)
      return failure();

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return failure();

    if (inputType.getShape() != outputType.getShape())
      return failure();

    if (onnxResizeOp.getCoordinateTransformationMode() == "tf_crop_and_resize")
      return failure();

    rewriter.replaceOp(onnxResizeOp, onnxResizeOp.getX());

    return success();
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

    // Don't fuse if element types differ (e.g. quantized -> f32 boundary).
    auto firstDataElemType =
        mlir::cast<ShapedType>(firstData.getType()).getElementType();
    auto secondResultElemType =
        mlir::cast<ShapedType>(secondReshapeOp.getType()).getElementType();
    if (firstDataElemType != secondResultElemType)
      return rewriter.notifyMatchFailure(
          op, "Element types differ across reshape chain");

    Location loc = rewriter.getFusedLoc(
        {firstReshapeOp.getLoc(), secondReshapeOp.getLoc()});
    OnnxBuilder createONNX(rewriter, loc);

    auto eraseTriviallyDeadValues = [&](PatternRewriter &rewriter,
                                        SmallVector<Value, 4> &values) {
      for (auto val : values) {
        auto *op = val.getDefiningOp();
        if (!op || !isOpTriviallyDead(op))
          continue;
        rewriter.eraseOp(op);
      }
    };

    // Try to compute a new shape tensor by fusing the two old shapes.
    SmallVector<Value, 4> firstDims, secondDims, fusedDims;
    if (!getValuesFromShape(createONNX, firstShape, firstDims) ||
        !getValuesFromShape(createONNX, secondShape, secondDims)) {
      // New values may be created by getValuesFromShape. Erase newly-created
      // values before failing. This avoids that the PatternRewriter notify
      // changes and prevent convergence issue.
      eraseTriviallyDeadValues(rewriter, firstDims);
      eraseTriviallyDeadValues(rewriter, secondDims);

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
      // New values may be created by getValuesFromShape. Erase newly-created
      // values before failing. This avoids that the PatternRewriter notify
      // changes and prevent convergence issue.
      eraseTriviallyDeadValues(rewriter, firstDims);
      eraseTriviallyDeadValues(rewriter, secondDims);

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
// Rewrite pattern concat
// =============================================================================

struct RecomposeConcatPattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  // Helper function to check if an input is a mergeable Concat.
  static bool isMergeableConcat(Value input, int64_t axis) {
    ONNXConcatOp concatOp = input.getDefiningOp<ONNXConcatOp>();
    if (!concatOp)
      return false;
    return (concatOp.getAxis() == axis) && (concatOp.getResult().hasOneUse());
  }

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    ValueRange inputs = concatOp.getOperands();
    int64_t axis = concatOp.getAxis();

    // If there is only a single input, replace the concat with that input.
    if (inputs.size() == 1) {
      rewriter.replaceOp(concatOp, inputs[0]);
      return success();
    }

    SmallVector<Value, 16> newInputs;
    bool merged = false;
    SmallVector<Location> concatLocations;
    concatLocations.push_back(concatOp->getLoc());

    // Flatten nested concat nodes.
    for (Value input : inputs) {
      if (isMergeableConcat(input, axis)) {
        // Remove the nested concat and append its inputs.
        ONNXConcatOp innerConcat = cast<ONNXConcatOp>(input.getDefiningOp());
        newInputs.append(
            innerConcat.getOperands().begin(), innerConcat.getOperands().end());
        concatLocations.push_back(innerConcat->getLoc());
        merged = true;
      } else {
        // Push non-mergeable input.
        newInputs.push_back(input);
      }
    }

    if (merged) {
      // Create a new ONNXConcat op with the flattened inputs.
      auto newConcat =
          rewriter.create<ONNXConcatOp>(rewriter.getFusedLoc(concatLocations),
              concatOp.getResult().getType(), newInputs, axis);
      rewriter.replaceOp(concatOp, newConcat.getResult());
      return success();
    }

    return failure();
  }
};

// =============================================================================
// Rewrite pattern LayerNormalization
// =============================================================================
namespace {

// Checks if B is unidiretional broadcastable to A. Requires static shapes
[[nodiscard]] bool areUnidirectionalBroadcastCompatible(Type a, Type b) {
  auto aShaped = dyn_cast<ShapedType>(a);
  auto bShaped = dyn_cast<ShapedType>(b);
  if (!aShaped || !bShaped || !aShaped.hasStaticShape() ||
      !bShaped.hasStaticShape()) {
    return false;
  }
  SmallVector<int64_t> broadcastedShape;
  if (!OpTrait::util::getBroadcastedShape(
          aShaped.getShape(), bShaped.getShape(), broadcastedShape)) {
    return false;
  }
  // For unidirectional broadcasting, a and the resulting shape need to match
  return aShaped.getShape() == ArrayRef<int64_t>(broadcastedShape);
}

[[nodiscard]] bool isValueNoneOrConstZero(Value value) {
  if (!value) {
    return false;
  }
  if (isNoneValue(value)) {
    return true;
  }
  auto elementsAttr = getElementAttributeFromONNXValue(value);
  if (!elementsAttr) {
    return false;
  }
  if (!elementsAttr.isSplat()) {
    return false;
  }
  return elementsAttr.template getSplatValue<APFloat>().isZero();
}

template <typename LN_TYPE, typename MATCH_OP_TYPE,
    size_t OPERAND_TO_MODIFY_INDEX>
struct PropagateBiasOrScaleIntoLayerNormRewritePatternBase
    : public OpRewritePattern<MATCH_OP_TYPE> {
  using OpRewritePattern<MATCH_OP_TYPE>::OpRewritePattern;

  static_assert(std::is_same_v<MATCH_OP_TYPE, ONNXAddOp> ||
                    std::is_same_v<MATCH_OP_TYPE, ONNXMulOp>,
      "MATCH_OP_TYPE must be ONNXAddOp or ONNXMulOp");

  [[nodiscard]] virtual bool doExisitingScaleAndBiasAllowFusion(
      LN_TYPE lnOp) const = 0;

  FailureOr<SmallVector<int64_t>> verifyAndCalculateNewReshapeShapes(
      Operation *reshapeOp, MATCH_OP_TYPE matchOp, PatternRewriter &rewriter,
      Value scaleOrBias) const {
    // if we have a reshape, check that the add/mul is not changing the shape
    // by broadcasting
    auto reshapeResultType =
        dyn_cast<ShapedType>(reshapeOp->getResult(0).getType());
    auto addOrMulResultType =
        dyn_cast<ShapedType>(matchOp->getResult(0).getType());
    if (!reshapeResultType || !addOrMulResultType ||
        !reshapeResultType.hasStaticShape() ||
        !addOrMulResultType.hasStaticShape() ||
        reshapeResultType.getShape() != addOrMulResultType.getShape()) {
      return rewriter.notifyMatchFailure(
          matchOp, "incompatible shapes, add is broadcasting");
    }
    // Check that the bias/scale is only on a single dimension, that is not
    // affected by the reshape. The bias/scale could be multi-dimentional, but
    // this increases the complexity and was not seen in models
    auto scaleOrBiasType = dyn_cast<ShapedType>(scaleOrBias.getType());
    if (!scaleOrBiasType || !scaleOrBiasType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          matchOp, "bias/scale has not a static shape");
    }

    SmallVector<int64_t> biasOrScaleRankFixedShape;
    biasOrScaleRankFixedShape.append(
        addOrMulResultType.getRank() - scaleOrBiasType.getRank(), 1);
    biasOrScaleRankFixedShape.append(
        scaleOrBiasType.getShape().begin(), scaleOrBiasType.getShape().end());

    // biasOrScaleRankFixedShape should have exactly one dimension that is not
    // one
    std::optional<int64_t> afterReshapeComputationDim;
    for (auto [idx, dimSize] : enumerate(biasOrScaleRankFixedShape)) {
      if (dimSize != 1) {
        if (afterReshapeComputationDim) {
          return rewriter.notifyMatchFailure(
              matchOp, "scale/bias has more than one non-one dimension");
        }
        afterReshapeComputationDim = idx;
      }
    }
    if (!afterReshapeComputationDim) {
      return rewriter.notifyMatchFailure(
          matchOp, "scale/bias has no non-one dimension");
    }

    const auto shapeIncludingComputationDim =
        ArrayRef<int64_t>(reshapeResultType.getShape())
            .slice(0, *afterReshapeComputationDim + 1);
    const uint64_t computationRelevantSize =
        std::accumulate(shapeIncludingComputationDim.begin(),
            shapeIncludingComputationDim.end(), 1, std::multiplies<uint64_t>());

    // The bias/scale dim should be not affected by the reshape. We need to
    // map it back through it.
    size_t reshapeInComputationDim;
    auto reshapeInType =
        dyn_cast<ShapedType>(reshapeOp->getOperand(0).getType());
    if (!reshapeInType || !reshapeInType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          matchOp, "reshape input has not a static shape");
    }
    const auto reshapeInShape = reshapeInType.getShape();

    // trace the dim through the reshape
    uint64_t acc = 1;
    for (auto [idx, dimSize] : enumerate(reshapeInShape)) {
      acc *= dimSize;
      if (acc == computationRelevantSize) {
        if (dimSize != biasOrScaleRankFixedShape[*afterReshapeComputationDim]) {
          return rewriter.notifyMatchFailure(
              matchOp, "bias/scale shape is not compatible with reshape input");
        }
        reshapeInComputationDim = idx;
        break;
      }
      if (acc > computationRelevantSize) {
        return rewriter.notifyMatchFailure(
            matchOp, "bias/scale shape is not compatible with reshape input");
      }
    }
    SmallVector<int64_t> newScaleOrBiasShape;
    newScaleOrBiasShape.push_back(reshapeInShape[reshapeInComputationDim]);
    newScaleOrBiasShape.append(
        reshapeInShape.size() - reshapeInComputationDim - 1, 1);
    return newScaleOrBiasShape;
  }

  LogicalResult matchAndRewrite(
      MATCH_OP_TYPE matchOp, PatternRewriter &rewriter) const final {
    PatternRewriter::InsertionGuard guard(rewriter);
    using namespace onnx_mlir;
    Value y, scaleOrBias;
    Operation *yLayerNormOp = nullptr;
    Operation *reshapeOp = nullptr;
    SmallVector<int64_t> newScaleOrBiasShape; // only used if there is a reshape

    // Match
    // %noBias = "onnx.NoValue"()
    // %y, %mean, %invStdDev = "onnx.LayerNormalization"(%x, %scale, %noBias)
    //     {axis = 2 : si64, epsilon = 9.994E-6 : f32, stash_type = 1 : si64}
    // optional reshape between norm and add
    // %yBias = "onnx.Add/onnx.Mul"(%y, %scaleOrBias)

    if (onnx_mlir::operandOfOpDefinedBy<ONNXReshapeOp>(
            reshapeOp, matchOp, y, scaleOrBias, 0) ||
        onnx_mlir::operandOfOpDefinedBy<ONNXReshapeOp>(
            reshapeOp, matchOp, scaleOrBias, y, 1)) {
      yLayerNormOp = reshapeOp->getOperand(0).getDefiningOp<LN_TYPE>();
      if (!yLayerNormOp) {
        return rewriter.notifyMatchFailure(
            reshapeOp, "reshape op does not have a layer norm as input");
      }
      if (!reshapeOp->hasOneUse()) {
        return rewriter.notifyMatchFailure(
            reshapeOp, "reshape op does not have a single use");
      }
    } else {
      if (!onnx_mlir::operandOfOpDefinedBy<LN_TYPE>(
              yLayerNormOp, matchOp, y, scaleOrBias, 0) &&
          !onnx_mlir::operandOfOpDefinedBy<LN_TYPE>(
              yLayerNormOp, matchOp, scaleOrBias, y, 1))
        return rewriter.notifyMatchFailure(matchOp, "missing y, layer norm op");
    }

    // Study layer norm op; make sure its used only one and that bias is not
    // used.
    assert(yLayerNormOp && "yLayerNormOp should not be null");
    if (!yLayerNormOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          yLayerNormOp, "y/layer norm has too many uses");
    }
    auto lnOp = mlir::cast<LN_TYPE>(yLayerNormOp);
    if (!doExisitingScaleAndBiasAllowFusion(lnOp))
      return rewriter.notifyMatchFailure(
          lnOp, "existing scale and bias do not allow fusion");

    if (reshapeOp) {
      auto newShape = verifyAndCalculateNewReshapeShapes(
          reshapeOp, matchOp, rewriter, scaleOrBias);
      if (failed(newShape)) {
        return failure();
      }
      newScaleOrBiasShape = std::move(*newShape);
    }

    // Norms only support unidirectional broadcasting to x
    if (!reshapeOp && !areUnidirectionalBroadcastCompatible(
                          lnOp.getX().getType(), scaleOrBias.getType())) {
      return rewriter.notifyMatchFailure(matchOp,
          "layer norm and bias/scale are not unidirectional broadcast "
          "compatible");
    }

    rewriter.moveOpAfter(
        lnOp, matchOp); // Make sure we can use the const of the mul
    rewriter.setInsertionPoint(matchOp);
    if (reshapeOp) {
      onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
          rewriter, reshapeOp->getLoc());
      const auto newShapeConst = create.onnx.constantInt64(newScaleOrBiasShape);
      scaleOrBias = create.onnx.reshape(
          RankedTensorType::get(newScaleOrBiasShape,
              cast<ShapedType>(scaleOrBias.getType()).getElementType()),
          scaleOrBias, newShapeConst);
    }
    rewriter.modifyOpInPlace(lnOp, [&] {
      lnOp.setOperand(OPERAND_TO_MODIFY_INDEX, scaleOrBias);
      lnOp->setLoc(rewriter.getFusedLoc({lnOp.getLoc(), matchOp->getLoc()}));
    });
    if (reshapeOp) {
      rewriter.moveOpAfter(reshapeOp, lnOp);
      rewriter.replaceOp(matchOp, reshapeOp->getResult(0));
    } else {
      rewriter.replaceOp(matchOp, lnOp.getY());
    }
    return success();
  }
};

} // namespace

template <typename LN_TYPE>
struct PropagateScaleIntoLayerNormPattern
    : public PropagateBiasOrScaleIntoLayerNormRewritePatternBase<LN_TYPE,
          ONNXMulOp, /*scale*/ 1> {
  using PropagateBiasOrScaleIntoLayerNormRewritePatternBase<LN_TYPE, ONNXMulOp,
      /*scale*/ 1>::PropagateBiasOrScaleIntoLayerNormRewritePatternBase;

  bool doExisitingScaleAndBiasAllowFusion(LN_TYPE lnOp) const override {
    if (!isValueNoneOrConstZero(lnOp.getB())) {
      return false;
    }

    const auto elementsAttr = getElementAttributeFromONNXValue(lnOp.getScale());
    if (!elementsAttr) {
      return false;
    }
    if (!elementsAttr.isSplat()) {
      return false;
    }
    return elementsAttr.template getSplatValue<APFloat>().isExactlyValue(1.0);
  }
};

template <typename LN_TYPE>
struct PropagateBiasIntoLayerNormRewritePattern
    : public PropagateBiasOrScaleIntoLayerNormRewritePatternBase<LN_TYPE,
          ONNXAddOp, /*bias*/ 2> {
  using PropagateBiasOrScaleIntoLayerNormRewritePatternBase<LN_TYPE, ONNXAddOp,
      /*bias*/ 2>::PropagateBiasOrScaleIntoLayerNormRewritePatternBase;

  bool doExisitingScaleAndBiasAllowFusion(LN_TYPE lnOp) const override {
    return isValueNoneOrConstZero(lnOp.getB());
  }
};

// =============================================================================
// Rewrite pattern for Where
// =============================================================================

class NotWhereOptPattern : public OpRewritePattern<ONNXWhereOp> {
public:
  using OpRewritePattern<ONNXWhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXWhereOp onnxWhereOp, PatternRewriter &rewriter) const override {
    auto notOp = onnxWhereOp.getCondition().getDefiningOp<ONNXNotOp>();
    if (!notOp)
      return failure();
    rewriter.modifyOpInPlace(onnxWhereOp, [&]() {
      onnxWhereOp.getOperation()->setOperands(
          {notOp.getX(), onnxWhereOp.getY(), onnxWhereOp.getX()});
      onnxWhereOp->setLoc(
          rewriter.getFusedLoc({onnxWhereOp.getLoc(), notOp.getLoc()}));
    });
    return success();
  }
};

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
// Rewrite pattern for BatchNormalization
// =============================================================================
/// Decompose BatchNormV9 to BatchNorm
struct RemoveBatchNormV9Pattern
    : public OpRewritePattern<ONNXBatchNormalizationV9Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXBatchNormalizationV9Op batchNormOpV9,
      PatternRewriter &rewriter) const final {
    auto savedMeanRes = batchNormOpV9.getSavedMean();
    auto savedVarRes = batchNormOpV9.getSavedVar();
    if (!savedMeanRes.use_empty() || !savedVarRes.use_empty()) {
      return rewriter.notifyMatchFailure(batchNormOpV9.getLoc(),
          "saved_mean and saved_variance must have no use.");
    }
    auto batchNormOp = rewriter.create<ONNXBatchNormalizationOp>(
        batchNormOpV9.getLoc(),
        TypeRange{
            batchNormOpV9.getY().getType(),
            batchNormOpV9.getOutMean().getType(),
            batchNormOpV9.getOutVar().getType(),
        },
        batchNormOpV9.getX(), batchNormOpV9.getScale(), batchNormOpV9.getB(),
        batchNormOpV9.getMean(), batchNormOpV9.getVar(),
        batchNormOpV9.getEpsilon(), batchNormOpV9.getMomentum());
    rewriter.replaceOp(batchNormOpV9,
        {batchNormOp.getY(), batchNormOp.getRunningMean(),
            batchNormOp.getRunningVar(),
            rewriter.create<ONNXNoneOp>(batchNormOpV9.getLoc()),
            rewriter.create<ONNXNoneOp>(batchNormOpV9.getLoc())});
    return success();
  }
};

/// Decompose BatchNorm to BatchNormInferenceMode
struct RemoveBatchNormPattern
    : public OpRewritePattern<ONNXBatchNormalizationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXBatchNormalizationOp batchNormOp,
      PatternRewriter &rewriter) const final {

    auto meanRes = batchNormOp.getRunningMean();
    auto varianceRes = batchNormOp.getRunningVar();
    if (!meanRes.use_empty() || !varianceRes.use_empty()) {
      return rewriter.notifyMatchFailure(
          batchNormOp.getLoc(), "mean and variance must have no use.");
    }

    rewriter.replaceOp(batchNormOp,
        {rewriter.create<ONNXBatchNormalizationInferenceModeOp>(
             batchNormOp.getLoc(), batchNormOp.getY().getType(),
             batchNormOp.getX(), batchNormOp.getScale(), batchNormOp.getB(),
             batchNormOp.getInputMean(), batchNormOp.getInputVar(),
             batchNormOp.getEpsilon(), batchNormOp.getMomentum()),
            rewriter.create<ONNXNoneOp>(batchNormOp.getLoc()),
            rewriter.create<ONNXNoneOp>(batchNormOp.getLoc())});
    return success();
  }
};

// "Pulls" Relu-like operations up through a SplitOp
struct PullReluLikeOpsThroughSplitPattern
    : public OpRewritePattern<ONNXSplitOp> {
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const final {

    Operation *firstUser = nullptr;
    SmallVector<Operation *> reluLikeOps;
    Location newLoc = rewriter.getUnknownLoc();

    const auto areFilteredAttrsEqual = [](Operation *op1, Operation *op2) {
      DenseMap<StringRef, Attribute> filteredAttrs1;
      DenseMap<StringRef, Attribute> filteredAttrs2;
      for (const auto &attr : op1->getAttrs()) {
        if (attr.getName() != "onnx_node_name") {
          filteredAttrs1[attr.getName()] = attr.getValue();
        }
      }
      for (const auto &attr : op2->getAttrs()) {
        if (attr.getName() != "onnx_node_name") {
          filteredAttrs2[attr.getName()] = attr.getValue();
        }
      }
      return filteredAttrs1 == filteredAttrs2;
    };

    for (Operation *op : splitOp->getUsers()) {
      // TODO: This pattern could be more generic, for all unary, elementwise
      // ops. Having a trait for them would make this easier.
      if (!isa<ONNXReluOp, ONNXLeakyReluOp>(op)) {
        return rewriter.notifyMatchFailure(
            splitOp, "SplitOp must be used by a Relu-like op");
      }
      if (op->getOperand(0).getType() != op->getResult(0).getType()) {
        // This could happen if shape inference did not run
        return rewriter.notifyMatchFailure(
            splitOp, "Relu-like op must have same input and output type");
      }
      if (!firstUser) {
        firstUser = op;
      } else {
        if (firstUser->getName() != op->getName() ||
            !areFilteredAttrsEqual(firstUser, op)) {
          return rewriter.notifyMatchFailure(splitOp,
              "SplitOp must be used by Relu-like ops of the same type "
              "and attributes");
        }
      }
      reluLikeOps.push_back(op);
      newLoc = rewriter.getFusedLoc({newLoc, op->getLoc()});
    }
    rewriter.setInsertionPoint(splitOp);
    auto *newRelu = rewriter.clone(*reluLikeOps.front());
    rewriter.modifyOpInPlace(newRelu, [&]() {
      newRelu->setOperand(0, splitOp.getOperand(0));
      newRelu->getResult(0).setType(splitOp.getOperand(0).getType());
      newRelu->setLoc(newLoc);
    });
    rewriter.modifyOpInPlace(
        splitOp, [&]() { splitOp->setOperand(0, newRelu->getResult(0)); });
    for (Operation *op : reluLikeOps) {
      rewriter.replaceOp(op, op->getOperands());
    }
    return success();
  }
};

struct SoftmaxNegativeAxisPattern : public OpRewritePattern<ONNXSoftmaxOp> {
  using OpRewritePattern<ONNXSoftmaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXSoftmaxOp softmaxOp, PatternRewriter &rewriter) const final {

    auto inputType = dyn_cast<RankedTensorType>(softmaxOp.getInput().getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          softmaxOp, "Input is not a ranked tensor");

    const int64_t axis = softmaxOp.getAxis();
    const int64_t rank = inputType.getRank();

    if (axis >= 0)
      return failure(); // nothing to do.
    assert(-rank <= axis && "axis is out of range");
    rewriter.modifyOpInPlace(
        softmaxOp, [&]() { softmaxOp.setAxis(rank + axis); });
    return success();
  }
};

// Rewrite ONNXSoftmaxV11Op to ONNXSoftmaxOp (V13).
//
// V11 computes softmax over the flattened suffix [axis..rank-1].
// V13 computes softmax along a single axis.
//
// When axis is already the last dim the ops are equivalent.
// Otherwise we flatten the trailing dims, apply V13 softmax on that single
// flattened dim, then reshape back.
struct SoftmaxV11ToLatestPattern : public OpRewritePattern<ONNXSoftmaxV11Op> {
  using OpRewritePattern<ONNXSoftmaxV11Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSoftmaxV11Op op, PatternRewriter &rewriter) const final {
    Value input = op.getInput();
    int64_t axis = op.getAxis();
    Type resultType = op.getResult().getType();

    // axis == -1 always refers to the last dim, even for unranked tensors.
    if (axis == -1) {
      rewriter.replaceOpWithNewOp<ONNXSoftmaxOp>(op, resultType, input, axis);
      return success();
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "requires ranked input");

    int64_t rank = inputType.getRank();
    if (axis < 0)
      axis += rank;

    // If axis is innermost V11 and V13 semantics are identical.
    if (axis == rank - 1) {
      rewriter.replaceOpWithNewOp<ONNXSoftmaxOp>(op, resultType, input, axis);
      return success();
    }

    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "non-last-axis requires static shape");

    // Flatten [axis..rank-1] into a single trailing dimension, e.g.
    //   [1, 2, 3, 4, 5] with axis=2  ->  [1, 2, 60]
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t trailingDim = std::accumulate(inputShape.begin() + axis,
        inputShape.end(), int64_t(1), std::multiplies<int64_t>{});
    SmallVector<int64_t> flatShape(inputShape.take_front(axis));
    flatShape.push_back(trailingDim);
    auto flatType =
        RankedTensorType::get(flatShape, inputType.getElementType());

    OnnxBuilder onnx(rewriter, op.getLoc());
    auto inputReshapeOp =
        onnx.reshape(flatType, input, onnx.constantInt64(flatShape));
    auto softmaxOp = onnx.softmax(flatType, inputReshapeOp, axis);
    auto outputReshapeOp =
        onnx.reshape(resultType, softmaxOp, onnx.constantInt64(inputShape));
    rewriter.replaceOp(op, outputReshapeOp);
    return success();
  }
};

/*
 * Push down the transpose after scale (mul op), so the scale can be fused to
 * Layernorm.
 *
 * This means going from:
 *  constant     layernorm
 *     |             |
 *     |         transpose (loc1)
 *     *---------.   /
 *                mul (loc2)
 *                 |
 *
 * to:
 *
 *  constant      layernorm
 *     |              |
 *  transpose (loc1)  /
 *     *---------.   /
 *                mul (loc2)
 *                 |
 *             transpose (loc1)
 *                 |
 */
struct PushTransposeDownScalePattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const final {
    using namespace onnx_mlir;
    Operation *transposeOp = nullptr;
    Operation *layerOp = nullptr;
    Value Y;
    Value scale;
    Value transposedY;
    if (operandOfOpDefinedBy<ONNXTransposeOp>(
            transposeOp, mulOp, transposedY, scale, 0) ||
        operandOfOpDefinedBy<ONNXTransposeOp>(
            transposeOp, mulOp, scale, transposedY, 1)) {
      if (!operandOfOpDefinedBy<ONNXLayerNormalizationOp>(
              layerOp, transposeOp, Y, 0)) {
        return rewriter.notifyMatchFailure(
            mulOp, "transpose without preceding layernorm");
      }
      auto *op = scale.getDefiningOp();
      if (op == nullptr || !isa<ONNXConstantOp>(op)) {
        return rewriter.notifyMatchFailure(
            mulOp, "transpose without preceding constant");
      }
    } else {
      return rewriter.notifyMatchFailure(mulOp, "no preceding transpose found");
    }
    auto oldTranspose = cast<ONNXTransposeOp>(transposeOp);

    MultiDialectBuilder<OnnxBuilder> create(rewriter, oldTranspose->getLoc());

    // we have a transpose that we need to move behind the multiplication
    if (!oldTranspose->hasOneUse())
      return rewriter.notifyMatchFailure(
          mulOp, "more than one use for transpose");

    // use shape helper to get perm (handles default transpose case)
    IndexExprBuilderForAnalysis createIE(oldTranspose->getLoc());
    SmallVector<Value, 1> transposeOperands{oldTranspose.getData()};
    ONNXTransposeOpShapeHelper shapeHelper(
        oldTranspose.getOperation(), transposeOperands, &createIE);
    if (shapeHelper.computeShape().failed())
      return rewriter.notifyMatchFailure(
          mulOp, "could not compute transpose shape");
    ArrayAttr transposePerm = oldTranspose.getPermAttr();

    scale = create.onnx.upRank(scale, getRank(Y.getType()));
    auto transposedMulInput = create.onnx.transposeInt64(
        scale, invertPermutationVector(
                   extractFromIntegerArrayAttr<int64_t>(transposePerm)));
    auto newMulOp = create.onnx.mul(Y, transposedMulInput);
    newMulOp.setLoc(mulOp->getLoc());
    rewriter.replaceOpWithNewOp<ONNXTransposeOp>(mulOp,
        {oldTranspose->getLoc()}, transposedY.getType(), newMulOp,
        transposePerm);
    return llvm::success();
  }
};

// =============================================================================
// Fuses back-to-back maxpools (ONNXMaxPoolSingleOutOps):
// Goes From:
//                    │
//        ┌───────────▼──────────┐
//        │     Upper Maxpool    │
//        │                      │
//        │kernel_size = k1 x k1 │
//        │pads = p1, p1, p1, p1 │
//        │strides = s1 x s1     │
//        └───────────┬──────────┘
//        ┌───────────▼──────────┐
//        │     Lower Maxpool    │
//        │                      │
//        │kernel_size = k2 x k2 │
//        │pads = p2, p2, p2, p2 │
//        │strides = s2 x s2     │
//        └───────────┬──────────┘
//                    ▼
// To:
//                    │
//        ┌───────────▼──────────┐
//        │        Maxpool       │
//        │                      │      Where:
//        │kernel_size = k3 x k3 │          k3 = k1 + (k2 - 1) * s1
//        │pads = p3, p3, p3, p3 │          p3 = p1 + p2 * s1
//        │strides = s3 x s3     │          s3 = s1 * s2
//        └───────────┬──────────┘
//                    ▼
//
// This works for 1D, 2D or 3D maxpools, but only
// on symmetric kernels, strides, and paddings. It can be optimized further to
// work with asymmetric cases using similar logic individually for each dim
// that's being pooled upon.
// =============================================================================
struct FuseBackToBackMaxpools
    : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
  using OpRewritePattern<ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  static bool areAllSame(llvm::ArrayRef<Attribute> array, int64_t sameAs) {
    return llvm::all_of(array, [&](Attribute elem) {
      return cast<IntegerAttr>(elem).getInt() == sameAs;
    });
  }

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp lowerMaxpool,
      PatternRewriter &rewriter) const final {

    // Check that the lower maxpool is the second maxpool in a back-to-back
    // chain
    auto *upperOp = lowerMaxpool.getOperand().getDefiningOp();
    if (!upperOp) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Cannot get defining op for the lower maxpool");
    }

    if (!isa<ONNXDequantizeLinearOp>(upperOp) &&
        !isa<ONNXMaxPoolSingleOutOp>(upperOp)) {
      return rewriter.notifyMatchFailure(
          lowerMaxpool.getLoc(), "Defining op isn't a maxpool or a dequantize");
    }

    ONNXMaxPoolSingleOutOp upperMaxpool = nullptr;
    auto upperDequant = dyn_cast<ONNXDequantizeLinearOp>(upperOp);

    Operation *quantOp = nullptr;
    if (upperDequant) {
      auto *quant = upperDequant->getOperand(0).getDefiningOp();
      if (!quant || !isa<ONNXQuantizeLinearOp>(quant))
        return rewriter.notifyMatchFailure(
            lowerMaxpool->getLoc(), "No Q->Dq chain between the maxpools");
      quantOp = quant;
      Operation *quantInputDef = quant->getOperand(0).getDefiningOp();
      if (!quantInputDef)
        return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
            "QuantizeLinear input is not produced by a MaxPool");
      upperMaxpool = dyn_cast<ONNXMaxPoolSingleOutOp>(quantInputDef);
    } else {
      upperMaxpool = dyn_cast<ONNXMaxPoolSingleOutOp>(upperOp);
    }

    if (!upperMaxpool) {
      return rewriter.notifyMatchFailure(
          lowerMaxpool.getLoc(), "Defining op is not a maxpool");
    }

    // Check that the upper maxpool has only one user
    if (!upperMaxpool->hasOneUse()) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Optimization only works when upper maxpool has one user");
    }
    if (quantOp && !quantOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "QuantizeLinear before the "
          "upper maxpool has more than one user");
    }

    auto upperMaxpoolKernelSizeArr = upperMaxpool.getKernelShape().getValue();
    auto lowerMaxpoolKernelSizeArr = lowerMaxpool.getKernelShape().getValue();

    auto upperMaxpoolStridesArr = upperMaxpool.getStrides()->getValue();
    auto lowerMaxpoolStridesArr = lowerMaxpool.getStrides()->getValue();

    // Check for square kernels and strides
    if (!areAllSame(lowerMaxpoolKernelSizeArr,
            cast<IntegerAttr>(lowerMaxpoolKernelSizeArr[0]).getInt()) ||
        !areAllSame(upperMaxpoolKernelSizeArr,
            cast<IntegerAttr>(upperMaxpoolKernelSizeArr[0]).getInt())) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation only works on symmetric kernels");
    }

    if (!areAllSame(upperMaxpoolStridesArr,
            cast<IntegerAttr>(upperMaxpoolStridesArr[0]).getInt()) ||
        !areAllSame(lowerMaxpoolStridesArr,
            cast<IntegerAttr>(lowerMaxpoolStridesArr[0]).getInt())) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation only works on symmetric strides");
    }

    // Check for symmetric padding
    auto lowerMaxpoolPads = lowerMaxpool.getPads()->getValue();
    auto upperMaxpoolPads = upperMaxpool.getPads()->getValue();
    if (!areAllSame(lowerMaxpoolPads,
            cast<IntegerAttr>(lowerMaxpoolPads[0]).getInt()) ||
        !areAllSame(upperMaxpoolPads,
            cast<IntegerAttr>(upperMaxpoolPads[0]).getInt())) {
      return rewriter.notifyMatchFailure(lowerMaxpool.getLoc(),
          "Transformation only works for symmetric padings");
    }

    // Check for non-dilated maxpools (dilation = 1)
    auto lowerMaxpoolDilations = lowerMaxpool.getDilations();
    auto upperMaxpoolDilations = upperMaxpool.getDilations();
    bool areLowerDilationsOne =
        !lowerMaxpoolDilations ||
        areAllSame(lowerMaxpoolDilations->getValue(), 1);
    bool areUpperDilationsOne =
        !upperMaxpoolDilations ||
        areAllSame(upperMaxpoolDilations->getValue(), 1);
    if (!areLowerDilationsOne || !areUpperDilationsOne) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation only works for non-dilated maxpools");
    }

    // Check for same ceil-mode
    if (lowerMaxpool.getCeilMode() != upperMaxpool.getCeilMode()) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Both maxpools must have same ceil-mode for transformation to apply");
    }

    // Make sure we're doing explicit padding
    // This can also be extended by doing the same calculations as AUTO
    // PAD for the padding
    if (!(lowerMaxpool.getAutoPad() == "NOTSET") ||
        !(upperMaxpool.getAutoPad() == "NOTSET")) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation only supports explicit padding");
    }

    // Make sure both maxpools have the same storage order
    if (lowerMaxpool.getStorageOrder() != upperMaxpool.getStorageOrder()) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation applies only when both "
          "maxpools have the same storage order");
    }

    // Check kernel size >= stride for the upper maxpool
    auto upperMaxpoolKernelSize =
        cast<IntegerAttr>(upperMaxpoolKernelSizeArr[0]).getInt();
    auto upperMaxpoolStride =
        cast<IntegerAttr>(upperMaxpool.getStrides()->getValue()[0]).getInt();
    if (upperMaxpoolKernelSize < upperMaxpoolStride) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Transformation applies only when kernel "
          "size >= stride for the upper maxpool");
    }

    // Finally check that the upper maxpool covers the input completely
    auto upperMaxpoolPad = cast<IntegerAttr>(upperMaxpoolPads[0]).getInt();
    auto inputType = cast<RankedTensorType>(upperMaxpool.getX().getType());
    if (!inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(lowerMaxpool->getLoc(),
          "Upper maxpool has inputs with dynamic shapes");
    }

    auto inputShape = inputType.getShape();

    for (uint64_t pooledDimIdx = 2; pooledDimIdx < inputShape.size();
         pooledDimIdx++) {
      auto effectiveInputDim = inputShape[pooledDimIdx] + 2 * upperMaxpoolPad;
      if ((effectiveInputDim - upperMaxpoolKernelSize) % upperMaxpoolStride !=
          0) {
        return rewriter.notifyMatchFailure(lowerMaxpool.getLoc(),
            "Upper maxpool doesn't completely cover the input");
      }
    }

    // New ceil-mode:
    // Same ceil-mode as either maxpool
    auto newCeilMode =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true),
            lowerMaxpool.getCeilMode());

    // New Kernel Size:
    // k_fused = k_upper + (k_lower - 1) * stride_upper
    auto lowerMaxpoolKernelSize =
        cast<IntegerAttr>(lowerMaxpoolKernelSizeArr[0]).getInt();
    auto newKSize = upperMaxpoolKernelSize +
                    (lowerMaxpoolKernelSize - 1) * upperMaxpoolStride;
    SmallVector<int64_t> newKSizeVec(
        upperMaxpoolKernelSizeArr.size(), newKSize);
    auto newKernelSize = rewriter.getI64ArrayAttr(newKSizeVec);

    // New Stride:
    // stride_fused = stride_upper * stride_lower
    auto lowerMaxpoolStride =
        cast<IntegerAttr>(lowerMaxpool.getStrides()->getValue()[0]).getInt();
    SmallVector<int64_t> newStrideVec(
        upperMaxpoolStridesArr.size(), upperMaxpoolStride * lowerMaxpoolStride);
    auto newStride = rewriter.getI64ArrayAttr(newStrideVec);

    // New Padding:
    // padding_fused = padding_upper + padding_lower * stride_upper
    auto newPaddingVec = llvm::to_vector(
        llvm::map_range(llvm::zip_equal(upperMaxpoolPads, lowerMaxpoolPads),
            [&](auto pads) -> Attribute {
              auto [upperPad, lowerPad] = pads;
              return rewriter.getI64IntegerAttr(
                  cast<IntegerAttr>(upperPad).getInt() +
                  cast<IntegerAttr>(lowerPad).getInt() * upperMaxpoolStride);
            }));

    auto newPadding = rewriter.getArrayAttr(newPaddingVec);

    SmallVector<Location> locsToFuse;
    locsToFuse.push_back(upperMaxpool->getLoc());
    locsToFuse.push_back(lowerMaxpool->getLoc());
    if (upperDequant) {
      locsToFuse.push_back(quantOp->getLoc());
      locsToFuse.push_back(upperDequant->getLoc());
    }
    Location fusedLoc = rewriter.getFusedLoc(locsToFuse);
    MultiDialectBuilder<OnnxBuilder> b(rewriter, fusedLoc);
    auto newMaxpool =
        b.onnx.createTypedOpAndInferShapes<ONNXMaxPoolSingleOutOp>(
            lowerMaxpool->getResultTypes()[0], upperMaxpool.getX(),
            /*autopad = */ rewriter.getStringAttr("NOTSET"), newCeilMode,
            /*dilations = */ nullptr, newKernelSize, newPadding,
            /*storage_order = */
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64, /*isSigned=*/true),
                lowerMaxpool.getStorageOrder()),
            newStride);

    rewriter.replaceOp(lowerMaxpool, newMaxpool);

    return success();
  }
};

// Rewrite pattern for AveragePoolOp
struct FusePadIntoAveragePoolPattern
    : public OpRewritePattern<ONNXAveragePoolOp> {
  using OpRewritePattern<ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAveragePoolOp avgOp, PatternRewriter &rewriter) const override {

    Value input = avgOp.getX();
    auto padOp = input.getDefiningOp<ONNXPadOp>();
    if (!padOp)
      return failure();

    // Check that pad mode is "constant" (default value, so should never be
    // null)
    StringRef mode = padOp.getMode();
    if (mode != "constant")
      return failure();
    float padValue = 0.0f;

    Value padsInput = padOp.getPads();
    Value constantValInput = padOp.getConstantValue();

    auto padsConstOp =
        dyn_cast_or_null<ONNXConstantOp>(padsInput.getDefiningOp());
    if (!padsConstOp)
      return failure();
    auto padsAttr = dyn_cast_or_null<ElementsAttr>(padsConstOp.getValueAttr());
    if (!padsAttr)
      return failure();

    auto constOp =
        dyn_cast_or_null<ONNXConstantOp>(constantValInput.getDefiningOp());
    if (!constOp)
      return failure();
    auto constAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());

    if (!constAttr)
      return failure();

    auto firstAttr = *constAttr.getValues<Attribute>().begin();
    if (auto fAttr = mlir::dyn_cast<FloatAttr>(firstAttr))
      padValue = fAttr.getValueAsDouble();

    if (padValue != 0.0f)
      return failure();

    // Only handle 4D tensors (NCHW format)
    auto inputType = dyn_cast<RankedTensorType>(padOp.getData().getType());
    if (!inputType || inputType.getRank() != 4)
      return failure();

    // Extract pad values (guaranteed to be integers by ONNX spec)
    SmallVector<int64_t> padsVals;
    for (auto val : padsAttr.getValues<Attribute>()) {
      auto pad = cast<IntegerAttr>(val).getInt();
      padsVals.push_back(pad);
    }

    // Validate pads array size (2 * rank for begin/end)
    if (padsVals.size() != 8)
      return failure();

    // Only merge when padding is applied only to spatial dimensions (H, W)
    // padsVals layout: [N_begin, C_begin, H_begin, W_begin, N_end, C_end,
    // H_end, W_end]
    if (padsVals[0] != 0 || padsVals[1] != 0 || // N_begin, C_begin
        padsVals[4] != 0 || padsVals[5] != 0) { // N_end, C_end
      return failure(); // Cannot merge if batch or channel dims are padded
    }

    SmallVector<int64_t> mergedPads;
    if (auto existingPadsAttr = avgOp.getPadsAttr()) {
      for (Attribute v : existingPadsAttr) {
        mergedPads.push_back(cast<IntegerAttr>(v).getInt());
      }
    } else {
      mergedPads.resize(padsVals.size() / 2, 0);
    }

    if (mergedPads.size() != padsVals.size() / 2)
      return failure();

    // Merge spatial dimension padding (H, W)
    mergedPads[0] += padsVals[2]; // H_begin
    mergedPads[1] += padsVals[3]; // W_begin
    mergedPads[2] += padsVals[6]; // H_end
    mergedPads[3] += padsVals[7]; // W_end

    auto mergedPadsAttr =
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>(mergedPads));

    rewriter.modifyOpInPlace(avgOp, [&]() {
      avgOp->setAttr(avgOp.getPadsAttrName(), mergedPadsAttr);
      avgOp.getXMutable().assign(padOp.getData());
      avgOp->setLoc(rewriter.getFusedLoc({padOp.getLoc(), avgOp.getLoc()}));
    });

    rewriter.replaceOp(padOp, avgOp.getResult());

    return success();
  }
};

// Replace onnx.Gather with a scalar constant index by onnx.Slice +
// onnx.Reshape. Gather(data, scalar_constant_index, axis) is equivalent to
// slicing a single element along the axis and then squeezing that axis away.
class ReplaceGatherWithSlicePattern : public OpRewritePattern<ONNXGatherOp> {
public:
  using OpRewritePattern<ONNXGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    Value data = gatherOp.getData();
    Value indices = gatherOp.getIndices();
    int64_t axis = gatherOp.getAxis();

    auto inputType = dyn_cast<RankedTensorType>(data.getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();

    // Check that indices is a scalar
    auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesType || indicesType.getRank() != 0)
      return failure();

    auto gatherOutputType = dyn_cast<RankedTensorType>(gatherOp.getType());
    if (!gatherOutputType)
      return failure();

    // Check that indices is a constant integer value
    auto indicesConstOp = indices.getDefiningOp<ONNXConstantOp>();
    if (!indicesConstOp)
      return failure();
    auto idx = getScalarValue<int64_t>(indicesConstOp);

    const int64_t inputRank = inputType.getRank();
    if (axis < 0)
      axis += inputRank;

    ArrayRef<int64_t> inputShape = inputType.getShape();

    if (idx < 0)
      idx += inputShape[axis];

    OnnxBuilder createONNX(rewriter, loc);

    Value starts = createONNX.constantInt64({idx});
    Value ends = createONNX.constantInt64({idx + 1});
    Value axes = createONNX.constantInt64({axis});
    Value steps = createONNX.constantInt64({1});

    SmallVector<int64_t, 4> sliceShape(inputShape.begin(), inputShape.end());
    sliceShape[axis] = 1;
    auto sliceType =
        RankedTensorType::get(sliceShape, inputType.getElementType());

    Value sliceOp =
        createONNX.slice(sliceType, data, starts, ends, axes, steps);

    // Gather with a scalar index removes the gathered axis from the result,
    // but Slice preserves rank. Reshape to drop the size-1 axis.
    Value shapeConst = createONNX.constantInt64(
        SmallVector<int64_t>(gatherOutputType.getShape()));
    Value reshapeOp = createONNX.reshape(gatherOutputType, sliceOp, shapeConst);
    rewriter.replaceOp(gatherOp, reshapeOp);

    return success();
  }
};

// LeakyRelu with alpha == 0.0 is equivalent to Relu.
class LeakyReluAlphaZeroToReluPattern
    : public OpRewritePattern<ONNXLeakyReluOp> {
public:
  using OpRewritePattern<ONNXLeakyReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLeakyReluOp op, PatternRewriter &rewriter) const override {
    FloatAttr alphaAttr = op.getAlphaAttr();
    assert(alphaAttr);
    if (alphaAttr.getValueAsDouble() != 0.0)
      return failure();
    rewriter.replaceOpWithNewOp<ONNXReluOp>(
        op, op.getResult().getType(), op.getX());
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
  if (!disableBatchNormDecompose) {
    results.insert<RewriteBatchNormInferenceModeConvPattern1>(context);
    results.insert<RewriteBatchNormInferenceModeConvPattern2>(context);
  }
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
  results.insert<PropagateScaleIntoLayerNormPattern<ONNXLayerNormalizationOp>>(
      context);
  results
      .insert<PropagateScaleIntoLayerNormPattern<ONNXRMSLayerNormalizationOp>>(
          context);
  results.insert<
      PropagateBiasIntoLayerNormRewritePattern<ONNXLayerNormalizationOp>>(
      context);
  results.insert<
      PropagateBiasIntoLayerNormRewritePattern<ONNXRMSLayerNormalizationOp>>(
      context);
  results.insert<PropagateReshapeThroughBinaryOpPattern<ONNXAddOp>>(context);
  results.insert<BubbleUpBiasForNormOpPattern<ONNXLayerNormalizationOp>>(
      context);
  results.insert<BubbleUpBiasForNormOpPattern<ONNXRMSLayerNormalizationOp>>(
      context);
}

/// on the ONNXAndOp.
void ONNXAndOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXAndOp>>(context);
}

/// on the ONNXAveragePoolOp.
void ONNXAveragePoolOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FusePadIntoAveragePoolPattern>(context);
}

/// on the ONNXBatchNormOp.
void ONNXBatchNormalizationOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveBatchNormPattern>(context);
}

/// on the ONNXBatchNormV9Op.
void ONNXBatchNormalizationV9Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveBatchNormV9Pattern>(context);
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

/// on the ONNXConcatOp.
void ONNXConcatOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RecomposeConcatPattern>(context);
}

/// on the ONNXClipOp.
void ONNXClipOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FuseConsecutiveClipsPattern>(context);
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

/// on the ONNXGatherOp.
void ONNXGatherOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ReplaceGatherWithSlicePattern>(context);
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

/// on the ONNXLeakyReluOp.
void ONNXLeakyReluOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LeakyReluAlphaZeroToReluPattern>(context);
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

/// on the ONNXMaxPoolSingleOutOp.
void ONNXMaxPoolSingleOutOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ReorderReluMaxPoolPattern>(context);
  results.insert<FuseBackToBackMaxpools>(context);
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
  results.insert<PushTransposeDownScalePattern>(context);
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
  result.insert<ReplaceReshapeAllowZeroByReshape>(context);
}

/// on the ONNXResizeOp.
void ONNXResizeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<EmptyTensorInputsResizePattern>(context);
  result.insert<RemoveRedundantResizePattern>(context);
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

/// on the ONNXSoftmaxOp.
void ONNXSoftmaxOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SoftmaxNegativeAxisPattern>(context);
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

/// on the ONNXSplitOp
void ONNXSplitOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<PullReluLikeOpsThroughSplitPattern>(context);
  ;
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
  // Changed from upstream 64 to 2 because it can break quantization patterns
  result.insert<PowToMulRewritePattern>(context, 2);
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
  result.insert<NotWhereOptPattern>(context);
}

// on the ONNXDequantizeLinearOp.
void ONNXDequantizeLinearOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {}

void onnx_mlir::configureBatchNormCanonicalization(
    bool disableBatchNormDecomposeOption) {
  disableBatchNormDecompose = disableBatchNormDecomposeOption;
}
