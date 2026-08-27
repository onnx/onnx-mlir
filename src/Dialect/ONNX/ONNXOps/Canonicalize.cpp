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

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/TensorName.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "rewrite"

// Populated by configureBatchNormCanonicalization().
static bool disableBatchNormDecompose = false;

// Populated by configureUnsafeMathCanonicalization().
static bool enableUnsafeMath = true;

// Populated by configureReshapeCanonicalization().
static bool enableReshapeCanonicalization = true;

// Populated by configurePositiveAxisCanonicalization().
static bool enablePositiveAxisCanonicalization = true;

// Populated by configureKeepdimsCanonicalization().
static bool enableKeepdimsCanonicalization = true;

// Populated by configureQDQDataMovementCanonicalization().
static bool enableQDQDataMovementCanonicalization = false;

// Populated by configureExpandCanonicalization().
static bool enableExpandCanonicalization = false;

// Populated by configureCastDataMovementPatterns().
static bool enableCastDataMovementPatterns = true;

// Populated by configureGatherElementsTileCanonicalization().
static bool enableGatherElementsTileCanonicalization = true;

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

// Whether `bias` is either NoneType or "effectively zero" for the purpose of
// subsuming an Add constant into a Conv bias under a Q -> DQ identity. We
// accept three forms:
//   1. NoneType: the Conv has no bias at all.
//   2. ONNXConstantOp with dense<0...> value.
//   3. ONNXDequantizeLinearOp(qval, scale, zp) where both qval and zp are
//      constant all-zeros. Then dq = (qval - zp) * scale = 0 regardless of
//      the (possibly per-axis) scale.
// Used by FuseAddConvQDQBiasPattern to dispatch on either a missing or a
// semantically-zero Conv bias in a single pattern.
bool isNoneOrEffectivelyZeroBias(Value bias) {
  if (!bias)
    return false;
  if (mlir::isa<NoneType>(bias.getType()))
    return true;
  if (isConstOf(bias, 0.0))
    return true;
  if (auto dq = bias.getDefiningOp<ONNXDequantizeLinearOp>())
    return isConstOf(dq.getX(), 0.0) && isConstOf(dq.getXZeroPoint(), 0.0);
  return false;
}

// Whether `v` is an `ONNXConstantOp` whose dense value contains exactly one
// element (any rank, e.g. tensor<f32>, tensor<1xf32>, tensor<1x1x1x1xf32>).
// Used to verify that a quantized addend's q/scale/zero-point evaluates to a
// single scalar that can be safely flattened to a 1-D length-1 tensor before
// being broadcast across a Conv's output channels.
//
// Stronger than `isScalarConstantTensor` (which only accepts rank-0 or
// rank-1-with-shape-[1]); we need to also accept e.g. tensor<1x1x1x1xT>.
bool isSingleElementONNXConstant(Value v) {
  ElementsAttr attr = getElementAttributeFromONNXValue(v);
  return attr && attr.getNumElements() == 1;
}

// Build a runtime sub-graph that broadcasts a single-element addend value
// across the output channels of a Conv weight, producing a `<Cout xT>`
// tensor suitable as a Conv bias:
//
//   reshape_shape = onnx.Constant dense<[1]> : tensor<1xi64>
//   expand_shape  = onnx.Constant dense<[Cout]> : tensor<1xi64>
//   reshaped      = onnx.Reshape(addend_dq, reshape_shape) : tensor<1xT>
//   expanded      = onnx.Expand(reshaped, expand_shape) : tensor<Cout xT>
//
// Element type T is taken from `weight` so the resulting bias passes the
// Conv verifier's element-type check. Used by FuseAddConvQDQBiasPattern.
Value buildBroadcastBiasViaReshapeExpand(
    PatternRewriter &rewriter, Value addendDQ, Value weight) {
  auto loc = addendDQ.getLoc();
  auto weightType = mlir::cast<ShapedType>(weight.getType());
  assert(weightType.hasRank() && weightType.getRank() > 0 &&
         !weightType.isDynamicDim(0) &&
         "Conv weight must be ranked with a static Cout");
  auto fpType = weightType.getElementType();
  int64_t cOut = weightType.getShape()[0];
  auto shapeConstTy = RankedTensorType::get({1}, rewriter.getI64Type());

  Value reshaped = rewriter.create<ONNXReshapeOp>(loc,
      RankedTensorType::get({1}, fpType), addendDQ,
      rewriter.create<ONNXConstantOp>(loc, nullptr,
          DenseElementsAttr::get(shapeConstTy, llvm::ArrayRef<int64_t>{1})),
      IntegerAttr());

  return rewriter.create<ONNXExpandOp>(loc,
      RankedTensorType::get({cOut}, fpType), reshaped,
      rewriter.create<ONNXConstantOp>(loc, nullptr,
          DenseElementsAttr::get(shapeConstTy, llvm::ArrayRef<int64_t>{cOut})));
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

namespace {

template <typename ONNXOp>
void inferShapes(ONNXOp op) {
  if (failed(op.inferShapes([](Region &region) {})))
    llvm_unreachable("unexpected inferShapes failure");
}

} // namespace

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

/// Sink DepthToSpace ops beneath elementwise ops
/// Unary:  `UnaryOp(DepthToSpace(x))` -> `DepthToSpace(UnaryOp(x))`
/// Binary: `BinaryOp(DepthToSpace(x), DepthToSpace(y))` ->
///         `DepthToSpace(BinaryOp(x, y))` (no broadcast)
/// Sinking requires the DepthToSpace ops to have the same configuration and the
/// binary operation does not broadcast.
class DepthToSpaceElementwisePattern
    : public OpRewritePattern<ONNXDepthToSpaceOp> {
public:
  using OpRewritePattern<ONNXDepthToSpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDepthToSpaceOp d2sOp, PatternRewriter &rewriter) const override {
    for (Operation *user : d2sOp.getResult().getUsers()) {
      if (!user->hasTrait<OpTrait::ONNXElementwise>())
        continue;

      const unsigned numOperands = user->getNumOperands();
      if (numOperands == 1) {
        // Unary: DepthToSpace(UnaryOp(x)) -> UnaryOp(DepthToSpace(x))
        // rewritten as UnaryOp(DepthToSpace(x)) -> DepthToSpace(UnaryOp(x))
        const Type inputType = d2sOp.getInput().getType();
        const Type outputType = user->getResult(0).getType();

        Operation *newUnary = rewriter.clone(*user);
        newUnary->setOperands({d2sOp.getInput()});
        newUnary->getResult(0).setType(inputType);
        rewriter.replaceOpWithNewOp<ONNXDepthToSpaceOp>(user, {d2sOp.getLoc()},
            outputType, newUnary->getResult(0), d2sOp.getBlocksizeAttr(),
            d2sOp.getModeAttr());
        return success();
      }

      if (numOperands == 2) {
        // Binary: BinaryOp(DepthToSpace(x), DepthToSpace(y)) ->
        //         DepthToSpace(BinaryOp(x, y))
        auto lhs = user->getOperand(0).getDefiningOp<ONNXDepthToSpaceOp>();
        auto rhs = user->getOperand(1).getDefiningOp<ONNXDepthToSpaceOp>();
        if (!lhs || !rhs)
          continue;

        // Different DepthToSpace configurations
        if (lhs.getBlocksize() != rhs.getBlocksize() ||
            lhs.getMode() != rhs.getMode())
          continue;

        // Different input types implies they are broadcast. We should not sink
        // in this case.
        const Type lhsInputType = lhs.getInput().getType();
        const Type rhsInputType = rhs.getInput().getType();
        const Type outputType = user->getResult(0).getType();
        if (lhsInputType != rhsInputType ||
            lhs.getResult().getType() != outputType ||
            rhs.getResult().getType() != outputType)
          continue;

        Operation *newBinary = rewriter.clone(*user);
        newBinary->setOperands({lhs.getInput(), rhs.getInput()});
        newBinary->getResult(0).setType(lhsInputType);
        rewriter.replaceOpWithNewOp<ONNXDepthToSpaceOp>(user,
            {lhs.getLoc(), rhs.getLoc()}, outputType, newBinary->getResult(0),
            lhs.getBlocksizeAttr(), lhs.getModeAttr());
        return success();
      }
    }
    return failure();
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

// Drop reduction axes that point to dimensions of size 1 from
// `onnx.ReduceMean`. Reducing a unit-sized dimension is a no-op, so the axis
// can be removed from the `axes` operand without changing the result.
//
// Only `keepdims = 1` is handled here. With `keepdims = 0`, dropping a
// size-1 axis would change the output rank and require inserting a Squeeze,
// which is left to other rewrites.
//
// The empty-axes + `noop_with_empty_axes = 1` case (no reduction at all) is
// already handled by `ONNXReduceMeanOp::fold`, which forwards `data`.
class DropUnitAxesFromReduceMeanPattern
    : public OpRewritePattern<ONNXReduceMeanOp> {
public:
  using OpRewritePattern<ONNXReduceMeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceMeanOp op, PatternRewriter &rewriter) const override {
    if (op.getKeepdims() != 1)
      return rewriter.notifyMatchFailure(op, "only keepdims=1 is handled");

    auto dataType = mlir::dyn_cast<RankedTensorType>(op.getData().getType());
    if (!dataType || !dataType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "data must have static shape");
    const ArrayRef<int64_t> shape = dataType.getShape();
    const int64_t rank = dataType.getRank();

    // Collect the reduction axes. Axis ranges are validated by shape
    // inference, so we only need to handle the constant-vs-empty cases here.
    SmallVector<int64_t> axes;
    if (!isNoneValue(op.getAxes())) {
      if (!getI64ValuesFromONNXConstantOp(op.getAxes(), axes))
        return rewriter.notifyMatchFailure(op, "axes is not a constant");
    } else if (op.getNoopWithEmptyAxes() == 0) {
      // Empty axes with default semantics means reduce all dims.
      axes.resize(rank);
      std::iota(axes.begin(), axes.end(), int64_t{0});
    } else {
      return rewriter.notifyMatchFailure(op, "noop on empty axes");
    }

    // Drop axes that target unit-sized dimensions.
    SmallVector<int64_t> remainingAxes;
    for (int64_t a : axes) {
      const int64_t normA = a < 0 ? a + rank : a;
      if (shape[normA] != 1)
        remainingAxes.push_back(a);
    }
    if (remainingAxes.size() == axes.size())
      return rewriter.notifyMatchFailure(op, "no unit-sized axes to drop");

    // All reduction axes were unit-sized: with keepdims=1 the result shape
    // equals the input shape, so the op is equivalent to its data input.
    if (remainingAxes.empty()) {
      rewriter.replaceOp(op, op.getData());
      return success();
    }

    // Otherwise update the axes operand in place; all other operands and
    // attributes are unchanged.
    OnnxBuilder create(rewriter, op.getLoc());
    Value newAxes = create.constantInt64(remainingAxes);
    rewriter.modifyOpInPlace(op, [&] { op.getAxesMutable().assign(newAxes); });
    return success();
  }
};

// Upgrade ReduceMeanV13 latest ReduceMean. A present `axes` attribute becomes a
// constant axes operand; an absent one becomes a None operand with
// `noop_with_empty_axes = 0`, preserving the legacy "reduce over every
// dimension" semantics.
class UpgradeReduceMeanV13Pattern
    : public OpRewritePattern<ONNXReduceMeanV13Op> {
public:
  using OpRewritePattern<ONNXReduceMeanV13Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceMeanV13Op op, PatternRewriter &rewriter) const override {
    OnnxBuilder create(rewriter, op.getLoc());
    Value newAxes;
    if (ArrayAttr axesAttr = op.getAxesAttr()) {
      SmallVector<int64_t> axes;
      for (size_t i = 0; i < axesAttr.size(); ++i)
        axes.push_back(ArrayAttrIntVal(axesAttr, i));
      newAxes = create.constantInt64(axes);
    } else {
      newAxes = create.none();
    }
    IntegerAttr noopWithEmptyAxes = IntegerAttr::get(
        rewriter.getIntegerType(64, /*isSigned=*/true), APInt(64, 0, true));
    rewriter.replaceOpWithNewOp<ONNXReduceMeanOp>(op, op.getResult().getType(),
        op.getData(), newAxes, op.getKeepdimsAttr(), noopWithEmptyAxes);
    return success();
  }
};

// Materialize the implicit "reduce all axes" of a ReduceMean whose reduction
// axes operand is absent (None). An omitted `axes` means "reduce over every
// dimension" unless `noop_with_empty_axes = 1`, which is a no-op that
// `ONNXReduceMeanOp::fold` already forwards.
class MaterializeAbsentAxesReduceMeanPattern
    : public OpRewritePattern<ONNXReduceMeanOp> {
public:
  using OpRewritePattern<ONNXReduceMeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceMeanOp op, PatternRewriter &rewriter) const override {
    if (!isNoneValue(op.getAxes()))
      return rewriter.notifyMatchFailure(op, "axes already present");

    if (op.getNoopWithEmptyAxes() != 0)
      return rewriter.notifyMatchFailure(op, "noop on empty axes");

    auto dataType = mlir::dyn_cast<RankedTensorType>(op.getData().getType());
    if (!dataType)
      return rewriter.notifyMatchFailure(op, "data must be ranked");
    const int64_t rank = dataType.getRank();

    SmallVector<int64_t> axes(rank);
    std::iota(axes.begin(), axes.end(), int64_t{0});
    OnnxBuilder create(rewriter, op.getLoc());
    Value newAxes = create.constantInt64(axes);
    rewriter.modifyOpInPlace(op, [&] { op.getAxesMutable().assign(newAxes); });
    return success();
  }
};

// Rewrite keepdims=0 to keepdims=1 + Reshape to the original result shape.
// The op's own shape inference computes the keepdims=1 result type, so this
// applies to both reductions with an `axes` operand and ArgMin/ArgMax with an
// `axis` attribute.
template <typename OP_TYPE>
class KeepdimsCanonicalizationPattern : public OpRewritePattern<OP_TYPE> {
public:
  using OpRewritePattern<OP_TYPE>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OP_TYPE op, PatternRewriter &rewriter) const override {
    if (op.getKeepdims() != 0)
      return failure();

    Type resultType = op.getResult().getType();
    if (!hasStaticShape(resultType))
      return rewriter.notifyMatchFailure(op, "result must have static shape");

    auto keepdimsOp = mlir::cast<OP_TYPE>(rewriter.clone(*op.getOperation()));
    keepdimsOp.setKeepdimsAttr(rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true), 1));
    keepdimsOp.getResult().setType(
        UnrankedTensorType::get(getElementTypeOrSelf(resultType)));
    // The clone must not keep ResultNames: the ONNX tensor name belongs to the
    // Reshape that takes over this op's result. Dropping it here also lets the
    // ResultNames listener re-derive the intermediate as a reshaped view.
    keepdimsOp->removeAttr("ResultNames");
    inferShapes(keepdimsOp);

    OnnxBuilder create(rewriter, op.getLoc());
    Value shape = create.constant(
        createDenseElementsAttrFromShape(rewriter, op.getResult()));
    rewriter.replaceOp(op, create.reshape(resultType, keepdimsOp.getResult(),
                               shape, IntegerAttr()));
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

static bool getStaticExpandShapes(ONNXExpandOp expandOp,
    ArrayRef<int64_t> &inputShape, ArrayRef<int64_t> &resultShape) {
  Type inputType = expandOp.getInput().getType();
  Type resultType = expandOp.getOutput().getType();
  if (!hasStaticShape(inputType) || !hasStaticShape(resultType))
    return false;
  inputShape = getShape(inputType);
  resultShape = getShape(resultType);
  return true;
}

// Rewrite a same-rank static `Expand` into `Tile`. For each dimension the
// repeat is `1` when the sizes already match and `resultDim` when the input
// dim broadcasts from `1`; any other combination cannot be a pure tile.
class ExpandToTilePattern : public OpRewritePattern<ONNXExpandOp> {
public:
  using OpRewritePattern<ONNXExpandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXExpandOp expandOp, PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> inputShape;
    ArrayRef<int64_t> resultShape;
    if (!getStaticExpandShapes(expandOp, inputShape, resultShape))
      return failure();
    // Rank increases are normalized to equal rank by the reshape pattern first.
    if (inputShape.size() != resultShape.size() || inputShape.empty())
      return failure();

    SmallVector<int64_t> repeats;
    for (auto [inputDim, resultDim] : llvm::zip(inputShape, resultShape)) {
      if (inputDim == resultDim)
        repeats.push_back(1);
      else if (inputDim == 1)
        repeats.push_back(resultDim);
      else
        return failure();
    }

    MultiDialectBuilder<OnnxBuilder> create(rewriter, expandOp.getLoc());
    Value repeatsValue = create.onnx.constantInt64(repeats);
    Value tile = rewriter.create<ONNXTileOp>(expandOp.getLoc(),
        expandOp.getOutput().getType(), expandOp.getInput(), repeatsValue);
    rewriter.replaceOp(expandOp, tile);
    return success();
  }
};

// Normalize a rank-increasing static `Expand` by left-padding the input with
// unit dims via `Reshape`, leaving a same-rank `Expand` that the pattern above
// can turn into `Tile`.
class ExpandRankIncreaseToReshapeExpandPattern
    : public OpRewritePattern<ONNXExpandOp> {
public:
  using OpRewritePattern<ONNXExpandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXExpandOp expandOp, PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> inputShape;
    ArrayRef<int64_t> resultShape;
    if (!getStaticExpandShapes(expandOp, inputShape, resultShape))
      return failure();
    if (inputShape.size() >= resultShape.size())
      return failure();

    SmallVector<int64_t> reshapedShape(
        resultShape.size() - inputShape.size(), 1);
    reshapedShape.append(inputShape.begin(), inputShape.end());

    MultiDialectBuilder<OnnxBuilder> create(rewriter, expandOp.getLoc());
    Value input = expandOp.getInput();
    Type reshapedType =
        RankedTensorType::get(reshapedShape, getElementType(input.getType()));
    Value reshaped = create.onnx.reshape(
        reshapedType, input, create.onnx.constantInt64(reshapedShape));
    Value newExpand = create.onnx.expand(
        expandOp.getOutput().getType(), reshaped, expandOp.getShape());
    rewriter.replaceOp(expandOp, newExpand);
    return success();
  }
};

// Rewrite GatherElements(data, Tile(indices, repeats)) to
// Gather(data, Reshape(indices)) when Tile broadcasts a one-dimensional index
// vector across all non-axis dimensions of data.
class FuseGatherElementsTilePattern
    : public OpRewritePattern<ONNXGatherElementsOp> {
public:
  using OpRewritePattern<ONNXGatherElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXGatherElementsOp gatherElementsOp,
      PatternRewriter &rewriter) const override {
    auto tileOp = gatherElementsOp.getIndices().getDefiningOp<ONNXTileOp>();
    if (!tileOp || !tileOp->hasOneUse())
      return failure();

    auto dataType =
        dyn_cast<RankedTensorType>(gatherElementsOp.getData().getType());
    auto indicesType = dyn_cast<RankedTensorType>(tileOp.getInput().getType());
    auto tiledType = dyn_cast<RankedTensorType>(tileOp.getOutput().getType());
    if (!dataType || !indicesType || !tiledType || !dataType.hasStaticShape() ||
        !indicesType.hasStaticShape() || !tiledType.hasStaticShape())
      return failure();

    int64_t rank = dataType.getRank();
    if (indicesType.getRank() != rank || tiledType.getRank() != rank)
      return failure();

    int64_t axis = gatherElementsOp.getAxis();
    if (axis < 0)
      axis += rank;

    ArrayRef<int64_t> dataShape = dataType.getShape();
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    ArrayRef<int64_t> tiledShape = tiledType.getShape();

    // All non-axis dimensions must be broadcastable.
    int64_t numIndices = indicesShape[axis];
    if (numIndices < 1 || tiledShape[axis] != numIndices)
      return failure();
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis && (indicesShape[i] != 1 || tiledShape[i] != dataShape[i]))
        return failure();

    SmallVector<int64_t> gatherShape(dataShape);
    gatherShape[axis] = numIndices;
    auto gatherType =
        RankedTensorType::get(gatherShape, dataType.getElementType());
    if (gatherType != gatherElementsOp.getOutput().getType())
      return failure();

    OnnxBuilder create(rewriter, gatherElementsOp.getLoc());
    Value flattenedIndices = create.reshape(
        RankedTensorType::get({numIndices}, indicesType.getElementType()),
        tileOp.getInput(), create.constantInt64({numIndices}));
    rewriter.replaceOpWithNewOp<ONNXGatherOp>(gatherElementsOp, gatherType,
        gatherElementsOp.getData(), flattenedIndices,
        gatherElementsOp.getAxisAttr());
    return success();
  }
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
    // If reshape1 (the first/inner reshape) fans out to consumers beyond this
    // fold (i.e. reshape1 survives, shared with other users), tag reshape1's
    // parent (its input producer) with MultiUseConflict so the fold does not
    // clobber the shared producer's name. Checked BEFORE creating the fused
    // reshape below.
    if (std::distance(secondData.use_begin(), secondData.use_end()) > 1) {
      Value conflictVal = firstData; // parent of reshape1
      onnx_mlir::TensorName conflictName(conflictVal);
      while (!conflictName) {
        Operation *defOp = conflictVal.getDefiningOp();
        if (!defOp || defOp->getNumResults() != 1 ||
            defOp->getNumOperands() != 1)
          break;
        conflictVal = defOp->getOperand(0);
        conflictName = onnx_mlir::TensorName(conflictVal);
      }
      if (conflictName && llvm::none_of(conflictName.getTransforms(),
                              [](onnx_mlir::Transform *t) {
                                return isa<onnx_mlir::MultiUseConflict>(t);
                              })) {
        conflictName.push_back(std::make_unique<onnx_mlir::MultiUseConflict>());
        (void)conflictName.setTo(conflictVal);
      }
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

// A concat operand whose concat-axis dimension is statically zero contributes
// nothing along that axis and can be dropped. The result shape is unchanged,
// so the original result type is reused. Only a single such operand is
// removed per rewrite; when several are present the greedy driver re-matches.
// Restricted to arity > 1 so the rewrite never produces an operand-less concat;
// a resulting single operand is folded by `ConcatSingleOperandPattern`.
struct RemoveEmptyConcatOperandsPattern
    : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  static bool isEmptyAlongConcatAxis(Value v, int64_t axis) {
    auto type = mlir::dyn_cast<ShapedType>(v.getType());
    if (!type || !type.hasRank())
      return false;
    int64_t rank = type.getRank();
    if (axis < 0)
      axis += rank;
    return type.getDimSize(axis) == 0;
  }

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    if (concatOp.getNumOperands() <= 1)
      return failure();

    int64_t axis = concatOp.getAxis();
    for (auto [idx, operand] : llvm::enumerate(concatOp.getOperands())) {
      if (isEmptyAlongConcatAxis(operand, axis)) {
        rewriter.modifyOpInPlace(
            concatOp, [&] { concatOp->eraseOperand(idx); });
        return success();
      }
    }

    return failure();
  }
};

// A concat with a single operand is an identity.
struct ConcatSingleOperandPattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    if (concatOp.getNumOperands() != 1)
      return failure();
    rewriter.replaceOp(concatOp, concatOp.getOperand(0));
    return success();
  }
};

namespace {

[[nodiscard]] bool isPlainFloatType(Type t) {
  Type elem = getElementTypeOrSelf(t);
  return mlir::isa<FloatType>(elem);
}

// Walk upward through zero-or-more `onnx.Transpose`s, each consumed by a
// single user, appending them to `chain` in walk order (closest-to-`v`
// first). The returned `Value` is the source-side input to the chain (i.e.
// the input of the furthest transpose from `v`, or `v` itself when no
// transposes were found).
//
// Returns nullptr if any traversed transpose has more than one use, or if
// either its input or result element type is not a plain float (so we
// don't pull quantized data through the rewrite), or if any traversed
// transpose has no explicit `perm` attribute
Value walkBackThroughTransposes(
    Value v, SmallVectorImpl<ONNXTransposeOp> &chain) {
  while (auto t = v.getDefiningOp<ONNXTransposeOp>()) {
    if (!t->hasOneUse())
      return nullptr;
    if (!isPlainFloatType(t.getData().getType()) ||
        !isPlainFloatType(t.getTransposed().getType()))
      return nullptr;
    if (!t.getPermAttr())
      return nullptr;
    chain.push_back(t);
    v = t.getData();
  }
  return v;
}

// Build an `onnx.Constant` splat tensor of the requested shape and float
// element type
Value buildSplatConstant(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
    FloatType elemType, APFloat value) {
  auto tensorTy = RankedTensorType::get(shape, elemType);
  return b.create<ONNXConstantOp>(loc, /*sparse_value=*/Attribute(),
      /*value=*/DenseElementsAttr::get(tensorTy, value));
}

// =============================================================================
// Rewrite pattern: prefix slice/concat sandwich elimination around
//                  onnx.RotaryEmbedding
//
//
// This pattern absorbs a prefix slice into the cos/sin tables so the
// slices and the concat disappear entirely.
//
// This is done by padding the cos/sin tables with (cos = 1,
// sin = 0 at the prefix slots), which makes RoPE the identity for them
//
// Match (Q-side; K-side has an optional Transpose pair around RoPE):
//
//   pre    = onnx.Slice(X, starts=[0], ends=[prefixLen], axes=[A],
//   steps=[1])
//   pat    = onnx.Slice(X, starts=[prefixLen], ends=[N], axes=[A],
//   steps=[1])
//   rope   = onnx.RotaryEmbedding(Tpre*(pat), cos, sin, none)
//   {interleaved=0}
//    y     = onnx.Concat(pre, Tpost*(rope), axis=A)
//
// Rewrite to:
//
//   cosId   = splat(1.0, [1, prefixLen, halfD])
//   sinId   = splat(0.0, [1, prefixLen, halfD])
//   cosPad  = onnx.Concat(cosId, cos, axis=1)       // [1, S+prefixLen, halfD]
//   sinPad  = onnx.Concat(sinId, sin, axis=1)
//   y       = Tpost*(onnx.RotaryEmbedding(Tpre*(X), cosPad, sinPad, none))
//
//  The cosPad and sinPad can be constant folded
// =============================================================================

struct EliminateCarveOutAroundRotaryEmbeddingPattern
    : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  // Number of leading positions that bypass RoPE
  static constexpr int64_t prefixLen = 1;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const final {
    if (concatOp.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(concatOp, "expected 2-arm concat");
    auto concatTy = dyn_cast<RankedTensorType>(concatOp.getResult().getType());
    if (!concatTy || !concatTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          concatOp, "concat result must be a ranked, static tensor");
    if (!isPlainFloatType(concatTy))
      return rewriter.notifyMatchFailure(
          concatOp, "concat result element type is not a plain float");

    const int64_t rank = concatTy.getRank();
    int64_t axisA = concatOp.getAxis();
    if (axisA < 0)
      axisA += rank;
    if (axisA < 0 || axisA >= rank)
      return rewriter.notifyMatchFailure(concatOp, "concat axis out of range");

    // Arm 0 is the prefix slice: an `onnx.Slice` of some source `X` carving
    // a clean prefix of size `prefixLen` along `axisA` with unit step.
    // Records `X` and `fullLen = X.shape[axisA]`.
    auto prefixSlice = concatOp.getInputs()[0].getDefiningOp<ONNXSliceOp>();
    if (!prefixSlice)
      return rewriter.notifyMatchFailure(
          concatOp, "first concat arm is not an onnx.Slice");
    if (!prefixSlice->hasOneUse())
      return rewriter.notifyMatchFailure(
          concatOp, "prefix slice must have a single use");
    if (!isPlainFloatType(prefixSlice.getResult().getType()) ||
        !isPlainFloatType(prefixSlice.getData().getType()))
      return rewriter.notifyMatchFailure(
          concatOp, "prefix slice / source has non-float element type");
    int64_t prefixAxis, prefixStart, prefixEnd, prefixStep;
    if (!extractSlice1DConst(
            prefixSlice, prefixAxis, prefixStart, prefixEnd, prefixStep))
      return rewriter.notifyMatchFailure(
          concatOp, "prefix slice operands are not single-axis i64 constants");
    if (prefixAxis != axisA || prefixStart != 0 || prefixEnd != prefixLen ||
        prefixStep != 1)
      return rewriter.notifyMatchFailure(concatOp,
          "prefix slice does not carve a clean prefix of size prefixLen");

    Value X = prefixSlice.getData();
    auto xTy = dyn_cast<RankedTensorType>(X.getType());
    if (!xTy || !xTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          concatOp, "shared source X must have a static ranked shape");
    if (!isPlainFloatType(xTy))
      return rewriter.notifyMatchFailure(
          concatOp, "shared source X has non-float element type");
    const int64_t fullLen = xTy.getShape()[axisA];
    if (fullLen <= prefixLen)
      return rewriter.notifyMatchFailure(
          concatOp, "fullLen must exceed prefixLen");

    // Arm 1 is a`onnx.RotaryEmbedding` surrounded by optional Transposes
    SmallVector<ONNXTransposeOp> permsPost;
    Value postChainInput =
        walkBackThroughTransposes(concatOp.getInputs()[1], permsPost);
    if (!postChainInput)
      return rewriter.notifyMatchFailure(
          concatOp, "post-RoPE transpose chain is malformed");
    auto rope = postChainInput.getDefiningOp<ONNXRotaryEmbeddingOp>();
    if (!rope || !rope->hasOneUse())
      return rewriter.notifyMatchFailure(concatOp,
          "second concat arm does not trace back to "
          "onnx.RotaryEmbedding (single-use)");

    // For now the matched RoPE must take a static rank-4 plain-float input,
    // have no `position_ids`, full rotation (`rotary_embedding_dim == 0`), and
    // the standard non-interleaved layout that the v1 rewrite supports.
    auto ropeInTy = dyn_cast<RankedTensorType>(rope.getX().getType());
    auto ropeOutTy = dyn_cast<RankedTensorType>(rope.getResult().getType());
    if (!ropeInTy || !ropeOutTy || ropeInTy.getRank() != 4 ||
        !ropeInTy.hasStaticShape() || !ropeOutTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          rope, "rope X must be rank-4 with static shape");
    if (!isPlainFloatType(ropeInTy) || !isPlainFloatType(ropeOutTy))
      return rewriter.notifyMatchFailure(
          rope, "rope X / result has non-float element type");
    if (!isa<NoneType>(rope.getPositionIds().getType()))
      return rewriter.notifyMatchFailure(rope, "position_ids must be NoValue");
    if (rope.getInterleaved() != 0)
      return rewriter.notifyMatchFailure(rope, "interleaved must be 0");
    if (rope.getRotaryEmbeddingDim() != 0)
      return rewriter.notifyMatchFailure(
          rope, "rotary_embedding_dim must be 0 (full rotation)");

    // From the RoPE input, chase through an optional pre-RoPE Transpose
    // chain back to a single-use `onnx.Slice`. That slice must read the
    // same `X` as the prefix slice and carve `[prefixLen, fullLen)` along
    // `axisA` with unit step.
    SmallVector<ONNXTransposeOp> permsPre;
    Value preChainInput = walkBackThroughTransposes(rope.getX(), permsPre);
    if (!preChainInput)
      return rewriter.notifyMatchFailure(
          concatOp, "pre-RoPE transpose chain is malformed");
    auto patSlice = preChainInput.getDefiningOp<ONNXSliceOp>();
    if (!patSlice || !patSlice->hasOneUse())
      return rewriter.notifyMatchFailure(
          concatOp, "RoPE input does not trace back to a single-use Slice");
    if (!isPlainFloatType(patSlice.getResult().getType()) ||
        !isPlainFloatType(patSlice.getData().getType()))
      return rewriter.notifyMatchFailure(
          concatOp, "patches slice / source has non-float element type");
    if (patSlice.getData() != X)
      return rewriter.notifyMatchFailure(
          concatOp, "prefix and patches slices read different sources");
    int64_t patAxis, patStart, patEnd, patStep;
    if (!extractSlice1DConst(patSlice, patAxis, patStart, patEnd, patStep))
      return rewriter.notifyMatchFailure(
          concatOp, "patches slice operands are not single-axis i64 constants");
    if (patAxis != axisA || patStart != prefixLen || patEnd != fullLen ||
        patStep != 1)
      return rewriter.notifyMatchFailure(
          concatOp, "patches slice does not carve [prefixLen, fullLen)");

    // Verify that the composed pre-RoPE Transpose chain maps the carve
    // axis (`axisA` of `X`) onto axis 2 of the RoPE input, and the
    // composed post-RoPE Transpose chain maps axis 2 of the RoPE output
    // back to `axisA` of the concat-input. Check this by tagging the
    // carve axis with `ShapedType::kDynamic` and walking the perm
    // sequence; if the sentinel does not land at the expected position
    // the matcher bails.
    const auto composeChainShape = [](ArrayRef<int64_t> startShape,
                                       ArrayRef<ONNXTransposeOp> chain) {
      SmallVector<int64_t> shape(startShape.begin(), startShape.end());
      for (auto t : llvm::reverse(chain))
        shape = applyPermutation(
            shape, extractFromIntegerArrayAttr<int64_t>(t.getPermAttr()));
      return shape;
    };
    {
      auto tagged = llvm::to_vector(xTy.getShape());
      tagged[axisA] = ShapedType::kDynamic;
      const auto out = composeChainShape(tagged, permsPre);
      if (out.size() != 4 || out[2] != ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(concatOp,
            "pre-RoPE transpose chain does not map carve axis to RoPE seq axis "
            "(axis 2)");
    }
    {
      auto tagged = llvm::to_vector(ropeInTy.getShape());
      tagged[2] = ShapedType::kDynamic;
      const auto out = composeChainShape(tagged, permsPost);
      if (out.size() != size_t(rank) || out[axisA] != ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(concatOp,
            "post-RoPE transpose chain does not map RoPE seq axis (axis 2) "
            "back to carve axis");
    }

    // cos/sin must be static rank-3 dense ONNX float constants of shape
    // `[1, fullLen-prefixLen, halfD]` with the same element type. Rank-3
    // pins the seq axis at index 1 (the `[batch=1, S, halfD]` RoPE
    // contract), which is what the padding step below assumes. Per-batch
    // caches (batch != 1) is not supported yet.
    auto cosTy = dyn_cast<RankedTensorType>(rope.getCosCache().getType());
    auto sinTy = dyn_cast<RankedTensorType>(rope.getSinCache().getType());
    if (!cosTy || !sinTy || !cosTy.hasStaticShape() || !sinTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          rope, "cos/sin must have static ranked shapes");
    if (cosTy.getRank() != 3 || sinTy.getRank() != 3)
      return rewriter.notifyMatchFailure(rope, "cos/sin must be rank-3");
    if (!isPlainFloatType(cosTy) || !isPlainFloatType(sinTy))
      return rewriter.notifyMatchFailure(
          rope, "cos/sin have non-float element type");
    if (getElementTypeOrSelf(cosTy) != getElementTypeOrSelf(sinTy))
      return rewriter.notifyMatchFailure(
          rope, "cos and sin must share the same element type");
    if (getElementTypeOrSelf(xTy) != getElementTypeOrSelf(cosTy))
      return rewriter.notifyMatchFailure(
          rope, "X and cos/sin must share the same element type");
    auto elemType = cast<FloatType>(getElementTypeOrSelf(cosTy));
    if (!onnx_mlir::isDenseONNXConstant(rope.getCosCache()) ||
        !onnx_mlir::isDenseONNXConstant(rope.getSinCache()))
      return rewriter.notifyMatchFailure(
          rope, "cos/sin must be dense ONNX constants");
    const int64_t expectedSeq = fullLen - prefixLen;
    const int64_t halfD = cosTy.getShape()[2];
    if (cosTy.getShape() != ArrayRef<int64_t>{int64_t(1), expectedSeq, halfD} ||
        sinTy.getShape() != ArrayRef<int64_t>{int64_t(1), expectedSeq, halfD})
      return rewriter.notifyMatchFailure(
          rope, "cos/sin shape does not match [1, fullLen-prefixLen, halfD]");

    // Rewrite section
    Location loc = FusedLoc::get(
        rewriter.getContext(), {concatOp.getLoc(), rope.getLoc(),
                                   prefixSlice.getLoc(), patSlice.getLoc()});
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);

    // Re-emit the optional pre-RoPE Transpose chain on the full source `X`
    // instead of on the patches slice.
    Value newRopeInput = X;
    SmallVector<int64_t> curShape(xTy.getShape().begin(), xTy.getShape().end());
    for (auto t : llvm::reverse(permsPre)) {
      ArrayAttr permAttr = t.getPermAttr();
      auto perm = extractFromIntegerArrayAttr<int64_t>(permAttr);
      curShape = applyPermutation(curShape, perm);
      auto newTy = RankedTensorType::get(curShape, elemType);
      newRopeInput = create.onnx.transpose(newTy, newRopeInput, permAttr);
    }
    // the rebuilt input must equal the original `rope.X` shape with
    // axis 2 (the seq axis under the `[B, N, S, D]` for RoPE) grown
    // from `fullLen-prefixLen` to `fullLen`.
    const auto newRopeInputTy = cast<RankedTensorType>(newRopeInput.getType());
    {
      [[maybe_unused]] ArrayRef<int64_t> rs = ropeInTy.getShape();
      [[maybe_unused]] ArrayRef<int64_t> ns = newRopeInputTy.getShape();
      assert(ns.size() == 4 && ns[0] == rs[0] && ns[1] == rs[1] &&
             ns[2] == fullLen && ns[3] == rs[3] &&
             "pre-RoPE transpose chain rebuild produced an unexpected shape");
    }

    // Pad cos/sin along the seq axis with `prefixLen` rows of the RoPE
    // identity (cos = 1, sin = 0) so that the carved-out prefix slot(s)
    // are a no-op rotation.
    Value cosId =
        buildSplatConstant(rewriter, loc, {int64_t{1}, prefixLen, halfD},
            elemType, APFloat::getOne(elemType.getFloatSemantics()));
    Value sinId =
        buildSplatConstant(rewriter, loc, {int64_t{1}, prefixLen, halfD},
            elemType, APFloat::getZero(elemType.getFloatSemantics()));
    auto paddedCacheTy =
        RankedTensorType::get({int64_t{1}, fullLen, halfD}, elemType);
    Value paddedCos = create.onnx.concat(
        paddedCacheTy, ValueRange{cosId, rope.getCosCache()}, /*axis=*/1);
    Value paddedSin = create.onnx.concat(
        paddedCacheTy, ValueRange{sinId, rope.getSinCache()}, /*axis=*/1);

    const auto si64 = rewriter.getIntegerType(64, /*isSigned=*/true);
    auto newRope = cast<ONNXRotaryEmbeddingOp>(rewriter.clone(*rope));
    newRope->setLoc(loc);
    newRope.getXMutable().assign(newRopeInput);
    newRope.getCosCacheMutable().assign(paddedCos);
    newRope.getSinCacheMutable().assign(paddedSin);
    newRope.getResult().setType(newRopeInputTy);
    newRope.setNumHeadsAttr(
        IntegerAttr::get(si64, newRopeInputTy.getShape()[1]));

    // Re-emit the optional post-RoPE Transpose chain on the new RoPE
    // result.
    Value newOut = newRope.getResult();
    curShape.assign(
        newRopeInputTy.getShape().begin(), newRopeInputTy.getShape().end());
    for (auto t : llvm::reverse(permsPost)) {
      ArrayAttr permAttr = t.getPermAttr();
      auto perm = extractFromIntegerArrayAttr<int64_t>(permAttr);
      curShape = applyPermutation(curShape, perm);
      auto newTy = RankedTensorType::get(curShape, elemType);
      newOut = create.onnx.transpose(newTy, newOut, permAttr);
    }

    assert(newOut.getType() == concatOp.getResult().getType() &&
           "post-RoPE transpose chain rebuild does not reproduce the original "
           "concat result type");

    rewriter.replaceOp(concatOp, newOut);
    return success();
  }
};

// =============================================================================
// Rewrite pattern: fuse a trailing single-element scalar `onnx.Mul` into the
//                  cos/sin caches of an upstream `onnx.RotaryEmbedding`.
//
// Match:
//
//   rope = onnx.RotaryEmbedding(X, cos, sin, none)
//          {interleaved=0, rotary_embedding_dim=0, num_heads=N}
//   t    = (zero or more) onnx.Transpose(... rope ...)
//   y    = onnx.Mul(t, scale)        // scale is a scalar
//
// Rewrite to:
//
//   cosNew = onnx.Mul(cos, scale)
//   sinNew = onnx.Mul(sin, scale)
//   rope2  = onnx.RotaryEmbedding(X, cosNew, sinNew, none) {same attrs}
//   y      = (the original transpose chain re-emitted on rope2)
//
// Correctness: for ONNX RoPE  the rotated half is built from
//
//     x1, x2   = split(input[..., :rotary_embedding_dim], 2, axis=-1)
//     real     = cos * x1 - sin * x2
//     imag     = sin * x1 + cos * x2
//     rotated  = concat(real, imag, axis=-1)
//     output   = concat(rotated, input[..., rotary_embedding_dim:],
//                       axis=-1)
//
// substituting `cos -> s*cos` and `sin -> s*sin` for any scalar `s` gives
// `s*real` and `s*imag`, hence `rotated' = s * rotated`.
//
// The un-rotated tail `input[..., rotary_embedding_dim:]` is concatenated
// unchanged, so `output' = s * output` only holds when that tail is
// empty. This is enforced by the `rotary_embedding_dim == 0` requirement.
//
// Pushing a scalar scale through transposes is safe, as its invariant to the
// permutation.
//
// =============================================================================

struct FuseScaleIntoRotaryEmbeddingPattern
    : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const final {
    // Scale must be the RHS dense ONNX constant with exactly one element.
    Value scale = mulOp.getB();
    Value data = mulOp.getA();
    if (!onnx_mlir::isDenseONNXConstant(scale) ||
        onnx_mlir::isDenseONNXConstant(data))
      return rewriter.notifyMatchFailure(
          mulOp, "Mul is not in canonical (data, const) form");
    auto scaleTy = dyn_cast<RankedTensorType>(scale.getType());
    if (!scaleTy || !scaleTy.hasStaticShape() || scaleTy.getNumElements() != 1)
      return rewriter.notifyMatchFailure(
          mulOp, "scale must be a single-element static tensor");
    if (!isPlainFloatType(scaleTy))
      return rewriter.notifyMatchFailure(
          mulOp, "scale element type is not a plain float");
    if (!isPlainFloatType(mulOp.getResult().getType()))
      return rewriter.notifyMatchFailure(
          mulOp, "Mul result element type is not a plain float");

    // Skip optional Transposes
    SmallVector<ONNXTransposeOp> permsPost;
    Value preTransposes = walkBackThroughTransposes(data, permsPost);
    if (!preTransposes)
      return rewriter.notifyMatchFailure(
          mulOp, "post-RoPE transpose chain is malformed");

    auto rope = preTransposes.getDefiningOp<ONNXRotaryEmbeddingOp>();
    if (!rope || !rope->hasOneUse())
      return rewriter.notifyMatchFailure(
          mulOp, "Mul does not come from a single-use onnx.RotaryEmbedding");

    // RoPE attribute / shape constraints. These could be relaxed in the future.
    //   - rank-4 plain-float static input
    //   - position_ids must be NoValue;
    //   - interleaved == 0 and rotary_embedding_dim == 0 (full rotation)
    auto ropeInTy = dyn_cast<RankedTensorType>(rope.getX().getType());
    auto ropeOutTy = dyn_cast<RankedTensorType>(rope.getResult().getType());
    if (!ropeInTy || !ropeOutTy || ropeInTy.getRank() != 4 ||
        !ropeInTy.hasStaticShape() || !ropeOutTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          rope, "rope X / result must be rank-4 with static shape");
    if (!isPlainFloatType(ropeInTy) || !isPlainFloatType(ropeOutTy))
      return rewriter.notifyMatchFailure(
          rope, "rope X / result has non-float element type");
    if (!isa<NoneType>(rope.getPositionIds().getType()))
      return rewriter.notifyMatchFailure(rope, "position_ids must be NoValue");
    if (rope.getInterleaved() != 0)
      return rewriter.notifyMatchFailure(rope, "interleaved must be 0");
    if (rope.getRotaryEmbeddingDim() != 0)
      return rewriter.notifyMatchFailure(
          rope, "rotary_embedding_dim must be 0 (full rotation)");

    // cos/sin must be dense ONNX constants of static shape, sharing the
    // scale's element type so the fused Mul is well-typed and constprop-able.
    auto cosTy = dyn_cast<RankedTensorType>(rope.getCosCache().getType());
    auto sinTy = dyn_cast<RankedTensorType>(rope.getSinCache().getType());
    if (!cosTy || !sinTy || !cosTy.hasStaticShape() || !sinTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          rope, "cos/sin must have static ranked shapes");
    if (!onnx_mlir::isDenseONNXConstant(rope.getCosCache()) ||
        !onnx_mlir::isDenseONNXConstant(rope.getSinCache()))
      return rewriter.notifyMatchFailure(
          rope, "cos/sin must be dense ONNX constants");
    if (getElementTypeOrSelf(cosTy) != getElementTypeOrSelf(scaleTy) ||
        getElementTypeOrSelf(sinTy) != getElementTypeOrSelf(scaleTy))
      return rewriter.notifyMatchFailure(
          rope, "cos/sin element type does not match scale");

    // Rewrite section
    SmallVector<Location, 4> locs{mulOp.getLoc(), rope.getLoc()};
    for (auto t : permsPost)
      locs.push_back(t.getLoc());
    Location loc = FusedLoc::get(rewriter.getContext(), locs);
    onnx_mlir::MultiDialectBuilder<onnx_mlir::OnnxBuilder> create(
        rewriter, loc);

    Value cosScaled = create.onnx.mul(rope.getCosCache(), scale);
    Value sinScaled = create.onnx.mul(rope.getSinCache(), scale);

    auto newRope = cast<ONNXRotaryEmbeddingOp>(rewriter.clone(*rope));
    newRope->setLoc(loc);
    newRope.getCosCacheMutable().assign(cosScaled);
    newRope.getSinCacheMutable().assign(sinScaled);

    Value newOut = newRope.getResult();
    SmallVector<int64_t> curShape = llvm::to_vector(ropeOutTy.getShape());
    Type elemType = getElementTypeOrSelf(ropeOutTy);
    for (auto t : llvm::reverse(permsPost)) {
      ArrayAttr permAttr = t.getPermAttr();
      auto perm = extractFromIntegerArrayAttr<int64_t>(permAttr);
      curShape = applyPermutation(curShape, perm);
      const auto newTy = RankedTensorType::get(curShape, elemType);
      newOut = create.onnx.transpose(newTy, newOut, permAttr);
    }

    rewriter.replaceOp(mulOp, newOut);
    return success();
  }
};

// =============================================================================
// Rewrite pattern LayerNormalization
// =============================================================================

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
// Reshape canonicalization patterns.
// Rewrite Flatten/Squeeze/Unsqueeze to onnx.Reshape when the output is fully
// static. Guarded by enableReshapeCanonicalization.
//
/// Rewrite Flatten/Squeeze/Unsqueeze as onnx.Reshape when the result is a
/// fully static ranked tensor.  All three ops have the data tensor as their
/// first operand and the reshaped tensor as their first (and only) result, so
/// no per-op accessors are needed.
template <typename OpT>
struct ReshapeFamilyToReshapePattern : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const final {
    auto resultType =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "output must be a static ranked tensor");
    SmallVector<int64_t> shape(
        resultType.getShape().begin(), resultType.getShape().end());
    Value shapeConst = rewriter.create<ONNXConstantOp>(
        op.getLoc(), nullptr, rewriter.getI64TensorAttr(shape));
    rewriter.replaceOpWithNewOp<ONNXReshapeOp>(
        op, resultType, op->getOperand(0), shapeConst);
    return success();
  }
};

using FlattenToReshapePattern = ReshapeFamilyToReshapePattern<ONNXFlattenOp>;
using SqueezeToReshapePattern = ReshapeFamilyToReshapePattern<ONNXSqueezeOp>;
using UnsqueezeToReshapePattern =
    ReshapeFamilyToReshapePattern<ONNXUnsqueezeOp>;

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

// Softmax along an axis whose dimension has size 1 is the constant tensor 1.0
// because exp(x)/exp(x) == 1.0 for all finite x.
// E.g. Softmax(x: tensor<8x1xf32>) {axis=1} ==> Constant 1.0 : tensor<8x1xf32>
struct SoftmaxSizeOneAxisPattern : public OpRewritePattern<ONNXSoftmaxOp> {
  using OpRewritePattern<ONNXSoftmaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXSoftmaxOp softmaxOp, PatternRewriter &rewriter) const final {
    const auto inputType =
        dyn_cast<RankedTensorType>(softmaxOp.getInput().getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          softmaxOp, "requires ranked, static-shape input");
    const auto elementType = dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          softmaxOp, "only float element types are folded");

    const int64_t rank = inputType.getRank();
    int64_t axis = softmaxOp.getAxis();
    if (axis < 0)
      axis += rank;

    assert(axis >= 0 && axis < rank && "axis is out of range");
    if (inputType.getShape()[axis] != 1)
      return failure();

    const auto resultType =
        RankedTensorType::get(inputType.getShape(), elementType);
    const auto valueAttr = DenseElementsAttr::get(
        resultType, rewriter.getFloatAttr(elementType, 1.0));
    const Value constantOp = rewriter.create<ONNXConstantOp>(
        softmaxOp.getLoc(), Attribute(), valueAttr);
    rewriter.replaceOp(softmaxOp, constantOp);
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

// LeakyRelu with alpha == 1.0 is the identity function f(x) = x.
class LeakyReluAlphaOneToIdentityPattern
    : public OpRewritePattern<ONNXLeakyReluOp> {
public:
  using OpRewritePattern<ONNXLeakyReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXLeakyReluOp op, PatternRewriter &rewriter) const override {
    FloatAttr alphaAttr = op.getAlphaAttr();
    assert(alphaAttr);
    if (alphaAttr.getValueAsDouble() != 1.0)
      return failure();

    // Only eliminate the op when the input and result types match.
    if (op.getX().getType() != op.getResult().getType())
      return failure();

    rewriter.replaceOp(op, op.getX());
    return success();
  }
};

// onnx.Abs(onnx.Abs(x)) -> onnx.Abs(x) by reusing the inner Abs result.
class AbsAbsPattern : public OpRewritePattern<ONNXAbsOp> {
public:
  using OpRewritePattern<ONNXAbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAbsOp op, PatternRewriter &rewriter) const override {
    Value x = op.getX();
    if (mlir::isa<quant::QuantizedType>(getElementTypeOrSelf(x)))
      return failure();
    auto innerAbs = x.getDefiningOp<ONNXAbsOp>();
    if (!innerAbs)
      return failure();
    rewriter.replaceOp(op, innerAbs.getResult());
    return success();
  }
};

namespace {

// Normalizes a scalar `axis` attribute in place.
bool normalizeAxisAttr(
    Operation *op, const ONNXAxisValueSpec &spec, PatternRewriter &rewriter) {
  SmallVector<int64_t, 1> axes;
  if (!getONNXAxisValues(op, spec, axes) || axes.size() != 1)
    return false;
  const int64_t axis = axes[0];
  if (axis >= 0)
    return false;

  const std::optional<int64_t> rank =
      getONNXAxisNormalizationRank(op, spec, axes.size());
  if (!rank)
    return false;

  const auto axisAttr = op->getAttrOfType<IntegerAttr>(spec.attrName);
  if (!axisAttr)
    return false;
  const std::optional<int64_t> normalized =
      normalizeONNXAxisValue(axis, *rank, spec.includeRank);
  if (!normalized || *normalized == axis)
    return false;

  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(
        spec.attrName, IntegerAttr::get(axisAttr.getType(), *normalized));
  });
  return true;
}

// Normalizes an `axes` array attribute in place.
bool normalizeAxesAttr(
    Operation *op, const ONNXAxisValueSpec &spec, PatternRewriter &rewriter) {
  const auto axesAttr = op->getAttrOfType<ArrayAttr>(spec.attrName);
  if (!axesAttr || axesAttr.empty())
    return false;
  SmallVector<IntegerAttr> axisAttrs;
  axisAttrs.reserve(axesAttr.size());
  for (const Attribute attr : axesAttr) {
    const auto axisAttr = mlir::dyn_cast<IntegerAttr>(attr);
    if (!axisAttr)
      return false;
    axisAttrs.push_back(axisAttr);
  }

  SmallVector<int64_t> axes;
  if (!getONNXAxisValues(op, spec, axes))
    return false;
  const std::optional<int64_t> rank =
      getONNXAxisNormalizationRank(op, spec, axes.size());
  if (!rank)
    return false;

  SmallVector<int64_t> normalizedAxes;
  const FailureOr<bool> changed =
      normalizeONNXAxisValues(axes, *rank, spec.includeRank, normalizedAxes);
  if (failed(changed) || !*changed)
    return false;

  SmallVector<Attribute> normalizedAttrs;
  normalizedAttrs.reserve(axesAttr.size());
  for (auto [axisAttr, axis] : llvm::zip_equal(axisAttrs, normalizedAxes)) {
    normalizedAttrs.push_back(IntegerAttr::get(axisAttr.getType(),
        APInt(axisAttr.getValue().getBitWidth(), axis, /*isSigned=*/true)));
  }
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(
        spec.attrName, ArrayAttr::get(op->getContext(), normalizedAttrs));
  });
  return true;
}

// Creates an ONNX integer constant with the same ranked tensor type as
// `oldValue`.
Value createIntegerConstantLike(PatternRewriter &rewriter, Location loc,
    Value oldValue, ArrayRef<int64_t> values) {
  const ElementsAttr elemsAttr = getElementAttributeFromONNXValue(oldValue);
  if (!elemsAttr)
    return {};
  const auto tensorType = mlir::dyn_cast<RankedTensorType>(elemsAttr.getType());
  if (!tensorType)
    return {};
  const auto intType = mlir::dyn_cast<IntegerType>(tensorType.getElementType());
  if (!intType)
    return {};
  SmallVector<APInt> apValues;
  apValues.reserve(values.size());
  for (const int64_t value : values)
    apValues.emplace_back(intType.getWidth(), value, /*isSigned=*/true);
  const auto newAttr = DenseElementsAttr::get(tensorType, apValues);
  return rewriter.create<ONNXConstantOp>(loc, /*sparse_value=*/Attribute(),
      /*value=*/newAttr);
}

// Normalizes a constant `axis` or `axes` operand in place.
bool normalizeAxisOperand(
    Operation *op, const ONNXAxisValueSpec &spec, PatternRewriter &rewriter) {
  assert(spec.index < op->getNumOperands() &&
         "axis operand spec index must be valid for op");

  const Value axisOperand = op->getOperand(spec.index);
  if (mlir::isa<NoneType>(axisOperand.getType()))
    return false;

  SmallVector<int64_t> axes;
  if (!getONNXAxisValues(op, spec, axes))
    return false;

  const std::optional<int64_t> rank =
      getONNXAxisNormalizationRank(op, spec, axes.size());
  if (!rank)
    return false;

  SmallVector<int64_t> normalizedAxes;
  const FailureOr<bool> changed =
      normalizeONNXAxisValues(axes, *rank, spec.includeRank, normalizedAxes);
  if (failed(changed) || !*changed)
    return false;

  const Value newAxisOperand = createIntegerConstantLike(
      rewriter, axisOperand.getLoc(), axisOperand, normalizedAxes);
  if (!newAxisOperand)
    return false;

  rewriter.modifyOpInPlace(
      op, [&]() { op->setOperand(spec.index, newAxisOperand); });
  return true;
}

struct ONNXPositiveAxisCanonicalizationPattern : public RewritePattern {
  ONNXPositiveAxisCanonicalizationPattern(
      MLIRContext *context, PatternBenefit benefit)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getDialectNamespace() !=
        ONNXDialect::getDialectNamespace())
      return failure();

    const std::optional<ONNXAxisValueSpec> spec = getONNXAxisValueSpec(op);
    if (!spec || !spec->canCanonicalize)
      return failure();

    if (spec->source == ONNXAxisValueSource::Operand)
      return success(normalizeAxisOperand(op, *spec, rewriter));
    if (spec->kind == ONNXAxisOperandKind::Scalar)
      return success(normalizeAxisAttr(op, *spec, rewriter));
    return success(normalizeAxesAttr(op, *spec, rewriter));
  }
};

} // namespace

namespace {
bool hasScalarConstantQDQParams(ONNXDequantizeLinearOp op) {
  return isScalarConstantTensor(op.getXScale()) &&
         isScalarConstantTensor(op.getXZeroPoint());
}

bool hasScalarConstantQDQParams(ONNXQuantizeLinearOp op) {
  return isScalarConstantTensor(op.getYScale()) &&
         isScalarConstantTensor(op.getYZeroPoint());
}

bool sameConstantValue(Value lhs, Value rhs) {
  ElementsAttr lhsAttr = getElementAttributeFromONNXValue(lhs);
  ElementsAttr rhsAttr = getElementAttributeFromONNXValue(rhs);
  return lhsAttr && rhsAttr &&
         compareValueFromElementAttribute(lhsAttr, rhsAttr);
}

bool haveSamePerTensorQuantizationParams(
    ONNXDequantizeLinearOp lhs, ONNXDequantizeLinearOp rhs) {
  return hasScalarConstantQDQParams(lhs) && hasScalarConstantQDQParams(rhs) &&
         sameConstantValue(lhs.getXScale(), rhs.getXScale()) &&
         sameConstantValue(lhs.getXZeroPoint(), rhs.getXZeroPoint()) &&
         getElementType(lhs.getX().getType()) ==
             getElementType(rhs.getX().getType()) &&
         getElementType(lhs.getY().getType()) ==
             getElementType(rhs.getY().getType());
}

void getDataInputs(Operation *op, SmallVectorImpl<Value> &dataInputs) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<ONNXConcatOp>([&](ONNXConcatOp concatOp) {
        llvm::copy(concatOp.getInputs(), std::back_inserter(dataInputs));
      })
      .Case<ONNXExpandOp, ONNXPadOp, ONNXReshapeOp, ONNXSliceOp, ONNXTileOp,
          ONNXTransposeOp>([&](Operation *movementOp) {
        dataInputs.push_back(movementOp->getOperand(0));
      });
}

bool doesDataMovementOpUseQuantizedElementType(Operation *op) {
  for (Value operand : op->getOperands())
    if (onnx_mlir::hasQuantizedElementType(operand))
      return true;
  for (Value result : op->getResults())
    if (onnx_mlir::hasQuantizedElementType(result))
      return true;
  return false;
}

LogicalResult hasSupportedQDQMovementSemantics(
    Operation *op, PatternRewriter &rewriter) {
  if (auto padOp = dyn_cast<ONNXPadOp>(op)) {
    // Constant-mode Pad needs fill-value quantization, not implemented yet
    if (padOp.getMode() == "constant" || !isNoneValue(padOp.getConstantValue()))
      return rewriter.notifyMatchFailure(
          padOp, "constant-mode Pad movement needs fill value quantization");
    return success();
  }

  if (isa<ONNXConcatOp, ONNXExpandOp, ONNXReshapeOp, ONNXSliceOp, ONNXTileOp,
          ONNXTransposeOp>(op))
    return success();

  return rewriter.notifyMatchFailure(
      op, "QDQ movement semantics are not explicitly supported for this op");
}

} // namespace

template <typename T>
class SinkDequantLinearOpAfterDataMovementOp : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      T op, PatternRewriter &rewriter) const override {

    if (failed(hasSupportedQDQMovementSemantics(op.getOperation(), rewriter)))
      return failure();
    if (doesDataMovementOpUseQuantizedElementType(op.getOperation()))
      return rewriter.notifyMatchFailure(
          op, "data movement op uses quantized element types");

    SmallVector<Value> dataInputs;
    getDataInputs(op.getOperation(), dataInputs);

    IRMapping mapping;
    ONNXDequantizeLinearOp commonDQOp;
    SmallVector<Location> fusedLoc{op.getLoc()};
    for (Value dataInput : dataInputs) {
      auto dqOp = dataInput.getDefiningOp<ONNXDequantizeLinearOp>();
      if (!dqOp || !hasScalarConstantQDQParams(dqOp))
        return rewriter.notifyMatchFailure(
            op, "data input is not per-tensor DequantizeLinear");
      if (commonDQOp && !haveSamePerTensorQuantizationParams(commonDQOp, dqOp))
        return rewriter.notifyMatchFailure(
            op, "data input DequantizeLinear parameters differ by value");
      commonDQOp = commonDQOp ? commonDQOp : dqOp;
      fusedLoc.push_back(dqOp.getLoc());
      mapping.map(dqOp.getY(), dqOp.getX());
    }
    if (!commonDQOp)
      return rewriter.notifyMatchFailure(op, "no DequantizeLinear inputs");

    SmallVector<Value> newInputs;
    for (Value operand : op->getOperands())
      newInputs.push_back(mapping.lookupOrDefault(operand));

    auto opShapedType = dyn_cast<ShapedType>(op.getType());
    if (!opShapedType)
      return rewriter.notifyMatchFailure(op, "op result is not shaped");

    const Location newLoc = rewriter.getFusedLoc(fusedLoc);
    auto newOp = rewriter.create<T>(newLoc,
        TypeRange{
            opShapedType.clone(getElementType(commonDQOp.getX().getType()))},
        ValueRange{newInputs}, op->getAttrs());

    auto newDQOp = rewriter.create<ONNXDequantizeLinearOp>(newLoc, op.getType(),
        newOp.getResult(), commonDQOp.getXScale(), commonDQOp.getXZeroPoint(),
        commonDQOp.getAxisAttr(), commonDQOp.getBlockSizeAttr());

    rewriter.replaceOp(op, newDQOp.getResult());
    return success();
  }
};

template <typename T>
class BubbleUpQuantLinearOpBeforeDataMovementOp
    : public OpRewritePattern<ONNXQuantizeLinearOp> {
public:
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp quantOp, PatternRewriter &rewriter) const override {
    auto dataMovementOp = quantOp.getX().getDefiningOp<T>();
    if (!dataMovementOp)
      return rewriter.notifyMatchFailure(
          quantOp, "QuantizeLinear input is not a supported data movement op");

    Value dataMovementResult = quantOp.getX();
    if (!dataMovementResult.hasOneUse())
      return rewriter.notifyMatchFailure(
          quantOp, "data movement result has multiple users");
    if (!hasScalarConstantQDQParams(quantOp))
      return rewriter.notifyMatchFailure(
          quantOp, "QuantizeLinear parameters are not per-tensor constants");
    if (failed(hasSupportedQDQMovementSemantics(
            dataMovementOp.getOperation(), rewriter)))
      return failure();
    if (doesDataMovementOpUseQuantizedElementType(
            dataMovementOp.getOperation()))
      return rewriter.notifyMatchFailure(
          quantOp, "data movement op uses quantized element types");

    SmallVector<Value> dataInputs;
    getDataInputs(dataMovementOp.getOperation(), dataInputs);

    Type quantizedElementType = getElementType(quantOp.getY().getType());
    IRMapping mapping;
    SmallVector<Location> fusedLoc{quantOp.getLoc(), dataMovementOp.getLoc()};
    const Location newLoc = rewriter.getFusedLoc(fusedLoc);
    for (Value dataInput : dataInputs) {
      auto dataInputShapedType = dyn_cast<ShapedType>(dataInput.getType());
      if (!dataInputShapedType)
        return rewriter.notifyMatchFailure(
            quantOp, "data movement input is not shaped");

      auto newQOp = rewriter.create<ONNXQuantizeLinearOp>(newLoc,
          dataInputShapedType.clone(quantizedElementType), dataInput,
          quantOp.getYScale(), quantOp.getYZeroPoint(), quantOp.getAxisAttr(),
          quantOp.getBlockSizeAttr(), quantOp.getOutputDtypeAttr(),
          quantOp.getSaturateAttr());
      mapping.map(dataInput, newQOp.getResult());
    }

    SmallVector<Value> newInputs;
    for (Value operand : dataMovementOp->getOperands())
      newInputs.push_back(mapping.lookupOrDefault(operand));

    auto newOp = rewriter.create<T>(newLoc, TypeRange{quantOp.getType()},
        ValueRange{newInputs}, dataMovementOp->getAttrs());

    rewriter.replaceOp(quantOp, newOp.getResult());
    return success();
  }
};

// Fold constant gathers that select the sole element of a singleton axis.
class FoldSingletonGatherPattern : public OpRewritePattern<ONNXGatherOp> {
public:
  using OpRewritePattern<ONNXGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    auto dataType = dyn_cast<RankedTensorType>(gatherOp.getData().getType());
    auto indicesType =
        dyn_cast<RankedTensorType>(gatherOp.getIndices().getType());
    if (!dataType || !indicesType)
      return rewriter.notifyMatchFailure(gatherOp, "inputs are not ranked");
    if (indicesType.getRank() != 0 && indicesType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          gatherOp, "indices must be a scalar or a singleton vector");
    if (indicesType.getRank() == 1 && indicesType.getDimSize(0) != 1)
      return rewriter.notifyMatchFailure(
          gatherOp, "indices must be a singleton vector");

    const int64_t axis = gatherOp.getAxis();
    if (axis < 0)
      return rewriter.notifyMatchFailure(gatherOp, "axis must be non-negative");
    if (axis >= dataType.getRank())
      return rewriter.notifyMatchFailure(gatherOp, "axis is out of bounds");
    if (dataType.getDimSize(axis) != 1)
      return rewriter.notifyMatchFailure(gatherOp, "axis is not a singleton");

    ElementsAttr indicesAttr =
        getElementAttributeFromConstLikeValue(gatherOp.getIndices());
    if (!indicesAttr)
      return rewriter.notifyMatchFailure(gatherOp, "indices are not constant");
    APInt indexAPInt = *indicesAttr.getValues<APInt>().begin();
    const bool indicesAreUnsigned =
        indicesType.getElementType().isUnsignedInteger();
    const int64_t index = indicesAreUnsigned
                              ? static_cast<int64_t>(indexAPInt.getZExtValue())
                              : indexAPInt.getSExtValue();
    if (indicesAreUnsigned ? index != 0 : (index != 0 && index != -1))
      return rewriter.notifyMatchFailure(
          gatherOp, "index does not select the singleton element");

    if (indicesType.getRank() == 1) {
      if (gatherOp.getType() != gatherOp.getData().getType())
        return rewriter.notifyMatchFailure(
            gatherOp, "result type differs from data type");
      rewriter.replaceOp(gatherOp, gatherOp.getData());
      return success();
    }

    OnnxBuilder onnx(rewriter, gatherOp.getLoc());
    Value axes = onnx.constantInt64({axis});
    rewriter.replaceOpWithNewOp<ONNXSqueezeOp>(
        gatherOp, gatherOp.getType(), gatherOp.getData(), axes);
    return success();
  }
};

/// Simplify Reshape(Cast(Reshape(x, s1)), s2) to Cast(x) when the outer
/// Reshape's result shape equals the inner Reshape's input shape (i.e., the
/// two Reshapes together form an identity).
struct FuseCastBetweenReshapesPattern : public OpRewritePattern<ONNXReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXReshapeOp outerReshape, PatternRewriter &rewriter) const final {
    auto castOp = outerReshape.getData().getDefiningOp<ONNXCastOp>();
    if (!castOp)
      return rewriter.notifyMatchFailure(outerReshape, "input is not Cast");
    auto innerReshape = castOp.getInput().getDefiningOp<ONNXReshapeOp>();
    if (!innerReshape)
      return rewriter.notifyMatchFailure(
          outerReshape, "Cast input is not Reshape");
    // Both result and inner input must be static ranked tensors with the
    // same shape
    auto outerTy =
        mlir::dyn_cast<RankedTensorType>(outerReshape.getResult().getType());
    auto innerInTy =
        mlir::dyn_cast<RankedTensorType>(innerReshape.getData().getType());
    if (!outerTy || !innerInTy)
      return rewriter.notifyMatchFailure(outerReshape, "types not ranked");
    if (!outerTy.hasStaticShape() || !innerInTy.hasStaticShape())
      return rewriter.notifyMatchFailure(outerReshape, "types not static");
    if (outerTy.getShape() != innerInTy.getShape())
      return rewriter.notifyMatchFailure(
          outerReshape, "outer result shape != inner input shape");
    Location fusedLoc = rewriter.getFusedLoc(
        {innerReshape.getLoc(), castOp.getLoc(), outerReshape.getLoc()});
    Value newCast = rewriter.create<ONNXCastOp>(fusedLoc, outerTy,
        innerReshape.getData(), castOp.getSaturateAttr(), castOp.getToAttr());
    rewriter.replaceOp(outerReshape, newCast);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNXConvOp canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Derive spatial rank from kernel_shape when present, otherwise from weight
// rank.
static std::optional<int64_t> getConvSpatialRank(ONNXConvOp convOp) {
  if (auto kernelShape = convOp.getKernelShape(); kernelShape.has_value()) {
    const int64_t rank = ArrayAttrSize(kernelShape);
    if (rank >= 1)
      return rank;
    return std::nullopt;
  }

  const Value W = convOp.getW();
  if (!hasShapeAndRank(W))
    return std::nullopt;
  const auto wShape = cast<ShapedType>(W.getType()).getShape();
  const int64_t spatialRank = static_cast<int64_t>(wShape.size()) - 2;
  if (spatialRank >= 1)
    return spatialRank;
  return std::nullopt;
}

} // namespace

// Materialize ONNX-default Conv attributes when omitted:
//   strides/dilations -> 1 on each spatial axis
//   pads              -> 0 when auto_pad is NOTSET or VALID
//
// SAME_UPPER / SAME_LOWER padding is left to NormalizeConvAutoPadPattern.
struct MaterializeDefaultConvParamsPattern
    : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    const bool needStrides = !convOp.getStrides().has_value();
    const bool needDilations = !convOp.getDilations().has_value();
    const StringRef autoPad = convOp.getAutoPad();
    const bool needPads = !convOp.getPads().has_value() &&
                          (autoPad == "NOTSET" || autoPad == "VALID");

    if (!needStrides && !needDilations && !needPads)
      return failure();

    const std::optional<int64_t> spatialRankOpt = getConvSpatialRank(convOp);
    if (!spatialRankOpt.has_value())
      return rewriter.notifyMatchFailure(
          convOp, "cannot derive spatial rank from kernel_shape or weight");

    const int64_t spatialRank = *spatialRankOpt;
    const SmallVector<int64_t> unitSpatial(spatialRank, 1);
    const SmallVector<int64_t> zeroPads(2 * spatialRank, 0);

    rewriter.modifyOpInPlace(convOp, [&] {
      if (needStrides)
        convOp.setStridesAttr(rewriter.getI64ArrayAttr(unitSpatial));
      if (needDilations)
        convOp.setDilationsAttr(rewriter.getI64ArrayAttr(unitSpatial));
      if (needPads)
        convOp.setPadsAttr(rewriter.getI64ArrayAttr(zeroPads));
    });
    return success();
  }
};

// Normalize auto_pad to NOTSET with explicit pads.
//
// SAME_UPPER / SAME_LOWER: compute the padding required to keep the output
// the same spatial size as the input (with ceil-division for stride > 1).
// VALID: all pads are zero.
// NOTSET with no pads attribute: fill with zeros.
//
// Requires static input spatial dims for SAME_*
// After the rewrite auto_pad == "NOTSET" and pads holds the explicit values.
struct NormalizeConvAutoPadPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    if (convOp.getAutoPad() == "NOTSET" && convOp.getPads().has_value())
      return rewriter.notifyMatchFailure(
          convOp, "auto_pad is already NOTSET with explicit pads");

    FailureOr<ConvGeometry> geometry = getConvGeometry(convOp);
    if (failed(geometry))
      return rewriter.notifyMatchFailure(
          convOp, "cannot resolve auto_pad into explicit pads");

    rewriter.modifyOpInPlace(convOp, [&] {
      convOp.setAutoPadAttr(rewriter.getStringAttr("NOTSET"));
      convOp.setPadsAttr(rewriter.getI64ArrayAttr(geometry->pads));
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fuse conv1x1 -> convKxK into a single convKxK.
//
// A 1x1 convolution only mixes channels independently at each spatial
// position. When it feeds exactly one following convolution, the channel mixing
// can be folded into the following convolution's weights. If the first bias is
// nonzero, it can also be folded into the following bias as long as the second
// convolution has no padding; with padding the first bias contribution would be
// smaller at border pixels and could not be represented by one constant bias.
//
// Fires when:
//   - conv1 is 1x1, stride/dilation 1, group 1, zero pads.
//   - conv2 consumes conv1's single output, same stride/dilation/group
//     constraints.
//   - auto_pad is NOTSET on both
//   - No extra op is needed for the bias fold: conv2 pads are all zero OR
//     b1 is None / all-zeros.
//
// The fused weight Wf[Cout,Cin,K,K] = W2 x W1 (channel contraction):
//   Wf[n,c,i,j] = sum_m W1[m,c] * W2[n,m,i,j]
//
// The fused bias bf[Cout]:
//   bf[n] = b2[n] + sum_m (sum_{i,j} W2[n,m,i,j]) * b1[m]
//
//===----------------------------------------------------------------------===//

namespace {

// True when every element of the optional I64 array attr equals 1.
// An absent attribute is treated as "all ones"
bool isAllOnes(std::optional<ArrayAttr> attrOpt) {
  if (!attrOpt.has_value())
    return true;
  return llvm::all_of(attrOpt->getAsValueRange<IntegerAttr>(),
      [](const APInt &v) { return v == 1; });
}

// True when every element of the optional I64 array attr equals 0.
// An absent attribute is treated as "all zeros"
bool isAllZeros(std::optional<ArrayAttr> attrOpt) {
  if (!attrOpt.has_value())
    return true;
  return llvm::all_of(attrOpt->getAsValueRange<IntegerAttr>(),
      [](const APInt &v) { return v == 0; });
}

// True when val is None or a dense constant whose every element is 0.
bool isNoneOrZero(Value val) { return isNoneValue(val) || isConstOf(val, 0.0); }

struct FuseConv1x1IntoConvPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp conv2, PatternRewriter &rewriter) const override {

    // ---- Match conv1 -------------------------------------------------
    const Value conv2X = conv2.getX();
    auto *defOp = conv2X.getDefiningOp();
    if (!defOp)
      return rewriter.notifyMatchFailure(
          conv2, "input has no defining op (block argument)");
    auto conv1 = dyn_cast<ONNXConvOp>(defOp);
    if (!conv1)
      return rewriter.notifyMatchFailure(
          conv2, "producer of input is not an ONNXConvOp");
    // conv1's result must be consumed only by conv2.
    if (!conv2X.hasOneUse())
      return rewriter.notifyMatchFailure(
          conv2, "conv1 result has multiple uses");

    // ---- Common attribute constraints --------------------------------
    if (conv1.getAutoPad() != "NOTSET" || conv2.getAutoPad() != "NOTSET")
      return rewriter.notifyMatchFailure(conv2,
          "auto_pad not yet normalised to NOTSET (run "
          "NormalizeConvAutoPadPattern first)");
    if (conv1.getGroup() != 1 || conv2.getGroup() != 1)
      return rewriter.notifyMatchFailure(conv2, "group != 1");
    if (!isAllOnes(conv1.getStrides()) || !isAllOnes(conv2.getStrides()))
      return rewriter.notifyMatchFailure(conv2, "strides != 1");
    if (!isAllOnes(conv1.getDilations()) || !isAllOnes(conv2.getDilations()))
      return rewriter.notifyMatchFailure(conv2, "dilations != 1");

    // ---- conv1 must be 1x1 with zero pads ---------------------------
    const Value W1 = conv1.getW();
    if (!hasStaticShape(W1.getType()))
      return rewriter.notifyMatchFailure(
          conv2, "conv1 weight has dynamic or unknown shape");
    const auto w1Shape = cast<ShapedType>(W1.getType()).getShape();
    // w1Shape = [Cmid, Cin/group, k1, k2, ...]; all spatial dims must be 1.
    const int64_t w1Rank = w1Shape.size();
    if (w1Rank < 3)
      return rewriter.notifyMatchFailure(
          conv2, "conv1 weight rank must be at least 3");
    for (int64_t i = 2; i < w1Rank; ++i)
      if (w1Shape[i] != 1)
        return rewriter.notifyMatchFailure(
            conv2, "conv1 is not a 1x1 conv (spatial kernel dim != 1)");
    if (!isAllZeros(conv1.getPads()))
      return rewriter.notifyMatchFailure(conv2, "conv1 has nonzero pads");

    // ---- Require conv2 weight to also have a fully static shape ------
    const Value W2 = conv2.getW();
    if (!hasStaticShape(W2.getType()))
      return rewriter.notifyMatchFailure(
          conv2, "conv2 weight has dynamic or unknown shape");
    const auto w2Shape = cast<ShapedType>(W2.getType()).getShape();
    // w2Shape = [Cout, Cmid, K, K]; require at least rank 3.
    if (static_cast<int64_t>(w2Shape.size()) < 3)
      return rewriter.notifyMatchFailure(
          conv2, "conv2 weight rank must be at least 3");
    // Cmid must match between W1 and W2.
    if (w1Shape[0] != w2Shape[1])
      return rewriter.notifyMatchFailure(
          conv2, "Cmid mismatch between conv1 and conv2 weights");

    // Element types must match.
    const Type elemType = getElementTypeOrSelf(W1.getType());
    if (elemType != getElementTypeOrSelf(W2.getType()))
      return rewriter.notifyMatchFailure(
          conv2, "weight element types do not match");

    const Value b1 = conv1.getB();
    const Value b2 = conv2.getB();

    if (hasQuantizedElementType(conv1.getX()) || hasQuantizedElementType(W1) ||
        hasQuantizedElementType(b1) || hasQuantizedElementType(conv1.getY()) ||
        hasQuantizedElementType(W2) || hasQuantizedElementType(b2) ||
        hasQuantizedElementType(conv2.getY()))
      return rewriter.notifyMatchFailure(
          conv2, "quantized Conv fusion is not supported");

    // ---- Require constant weights for fold-ability ------------------
    if (!isDenseONNXConstant(W1))
      return rewriter.notifyMatchFailure(
          conv2, "conv1 weight is not a dense ONNX constant");
    if (!isDenseONNXConstant(W2))
      return rewriter.notifyMatchFailure(
          conv2, "conv2 weight is not a dense ONNX constant");

    // ---- Bias constraints -------------------------------------------
    // The b1 contribution is spatially uniform only when conv2 pads are all
    // zero, unless b1 contributes nothing (None or zero).
    const bool conv2PadsAllZero = isAllZeros(conv2.getPads());
    if (!conv2PadsAllZero && !isNoneOrZero(b1))
      return rewriter.notifyMatchFailure(conv2,
          "conv2 has nonzero pads and conv1 has nonzero bias: "
          "bias fold would require a spatially-varying correction");

    if (!isNoneOrZero(b1) && !isDenseONNXConstant(b1))
      return rewriter.notifyMatchFailure(
          conv1, "conv1 bias is not a dense ONNX constant");
    if (!isNoneOrZero(b2) && !isDenseONNXConstant(b2))
      return rewriter.notifyMatchFailure(
          conv2, "conv2 bias is not a dense ONNX constant");

    // ---- Build fused weight -----------------------------------------
    // Wf[Cout,Cin,K...] = einsum("mc,nm...->nc...", W1_flat, W2)
    //
    // Using matmul:
    //   W1_flat = reshape(W1, [Cmid, Cin])
    //   W2_flat = reshape(W2, [Cout, Cmid, K*K])
    //             (K*K = product of spatial dims)
    //   W2_t    = transpose(W2_flat, [0,2,1])        [Cout, K*K, Cmid]
    //   Wf_flat = matmul(W2_t, W1_flat)              [Cout, K*K, Cin]
    //   Wf_perm = transpose(Wf_flat, [0,2,1])        [Cout, Cin, K*K]
    //   Wf      = reshape(Wf_perm, [Cout, Cin, K, K, ...])

    const Location fusedLoc =
        rewriter.getFusedLoc({conv1.getLoc(), conv2.getLoc()});
    OnnxBuilder ob(rewriter, fusedLoc);

    const int64_t cin = w1Shape[1];
    const int64_t cmid = w1Shape[0];
    const int64_t cout = w2Shape[0];
    const int64_t spatialRank = static_cast<int64_t>(w2Shape.size()) - 2;
    const int64_t kProd = std::accumulate(w2Shape.begin() + 2, w2Shape.end(),
        int64_t{1}, std::multiplies<int64_t>());

    // W1_flat [Cmid, Cin]
    const Value w1Flat =
        ob.reshape(RankedTensorType::get({cmid, cin}, elemType), W1,
            ob.constantInt64({cmid, cin}));

    // W2_flat [Cout, Cmid, K*K]
    const Value w2Flat =
        ob.reshape(RankedTensorType::get({cout, cmid, kProd}, elemType), W2,
            ob.constantInt64({cout, cmid, kProd}));

    // W2_t [Cout, K*K, Cmid]
    const Value w2T = ob.transposeInt64(w2Flat, {0, 2, 1});

    // Wf_flat [Cout, K*K, Cin]
    const Value wfFlat = ob.matmul(
        RankedTensorType::get({cout, kProd, cin}, elemType), w2T, w1Flat);

    // Wf_perm [Cout, Cin, K*K]
    const Value wfPerm = ob.transposeInt64(wfFlat, {0, 2, 1});

    // Wf [Cout, Cin, K, K, ...]
    SmallVector<int64_t> wfShape = {cout, cin};
    wfShape.append(w2Shape.begin() + 2, w2Shape.end());
    const Value Wf = ob.reshape(RankedTensorType::get(wfShape, elemType),
        wfPerm, ob.constantInt64(wfShape));

    // ---- Build fused bias -------------------------------------------
    // bf = b2  (if b1 is None/zero)
    // bf = MatMul(ReduceSum(W2,[2..]), b1) + b2  (if b1 nonzero and pads=0)
    Value bf;
    if (isNoneOrZero(b1)) {
      // b1 contributes nothing; carry b2 through unchanged (may be None).
      bf = b2;
    } else {
      // conv2PadsAllZero and dense constant biases are guaranteed by previous
      // guards.
      // S[Cout, Cmid] = ReduceSum(W2, axes=[2,3,...]) over all spatial
      // dims.
      SmallVector<int64_t> spatialAxes;
      for (int64_t i = 0; i < spatialRank; ++i)
        spatialAxes.push_back(2 + i);
      const Value axes = ob.constantInt64(spatialAxes);
      const Value wSum =
          ob.reduceSum(RankedTensorType::get({cout, cmid}, elemType), W2, axes,
              /*keepDims=*/false);

      // b1_contrib[Cout] = MatMul([Cout,Cmid], [Cmid]) -> [Cout]
      const Value b1Contrib =
          ob.matmul(RankedTensorType::get({cout}, elemType), wSum, b1);

      if (isNoneValue(b2))
        bf = b1Contrib;
      else
        bf = ob.add(b1Contrib, b2);
    }

    // ---- Build attrs for the fused conv -----------------------------
    // kernelShape and pads come from conv2; strides and dilations are
    // always 1 (verified by isAllOnes checks above).
    const SmallVector<int64_t> kernelShape(w2Shape.begin() + 2, w2Shape.end());
    SmallVector<int64_t> pads(2 * spatialRank, 0);
    if (conv2.getPads().has_value())
      for (int64_t i = 0; i < 2 * spatialRank; ++i)
        pads[i] = ArrayAttrIntVal(conv2.getPads(), i);
    const SmallVector<int64_t> ones(spatialRank, 1); // strides and dilations

    // ---- Create fused conv and replace conv2 ------------------------
    const Value fusedConv = ob.conv(conv2.getY().getType(), conv1.getX(), Wf,
        bf, "NOTSET", ones, /*group=*/1, kernelShape, pads, ones);

    rewriter.replaceOp(conv2, fusedConv);
    return success();
  }
};

/// `src <= mid`: casting `src` to `mid` is lossless.
///
///   int   -> int    no truncation or unsigned-to-signed range loss
///   int   -> float  significand covers the integer range
///   float -> float  `mid`'s format represents all of `src`
///   anything else   never (float -> int, quant types)
bool castPreservesAllValues(Type src, Type mid) {
  auto srcIntTy = dyn_cast<IntegerType>(src);
  auto midIntTy = dyn_cast<IntegerType>(mid);
  auto srcFloatTy = dyn_cast<FloatType>(src);
  auto midFloatTy = dyn_cast<FloatType>(mid);

  if (srcIntTy && midIntTy) {
    if (srcIntTy.isUnsigned() == midIntTy.isUnsigned())
      return srcIntTy.getWidth() <= midIntTy.getWidth();

    // An unsigned integer can be represented by a wider signed integer.
    return srcIntTy.isUnsigned() && !midIntTy.isUnsigned() &&
           srcIntTy.getWidth() < midIntTy.getWidth();
  }

  // Signless counts as signed; the mantissa width includes the integer bit.
  if (srcIntTy && midFloatTy) {
    unsigned intBits = srcIntTy.getWidth();
    unsigned neededBits = srcIntTy.isUnsigned() ? intBits : intBits - 1;
    return neededBits <= midFloatTy.getFPMantissaWidth();
  }

  // Not a width comparison: bf16 and f16 are both 16-bit but neither
  // represents the other.
  if (srcFloatTy && midFloatTy)
    return llvm::APFloatBase::isRepresentableBy(
        srcFloatTy.getFloatSemantics(), midFloatTy.getFloatSemantics());

  return false;
}

/// `mid >= dst`: casting to `dst` discards at least as much as casting to
/// `mid` did, hiding the intermediate loss. Integer-only and one signedness
/// throughout: int narrowing drops high bits while float rounding drops low
/// ones, and two float casts round twice, which is not bit-accurate.
bool finalCastSubsumesIntermediate(Type src, Type mid, Type dst) {
  auto srcIntTy = dyn_cast<IntegerType>(src);
  auto midIntTy = dyn_cast<IntegerType>(mid);
  auto dstIntTy = dyn_cast<IntegerType>(dst);
  if (!srcIntTy || !midIntTy || !dstIntTy)
    return false;

  return srcIntTy.getSignedness() == midIntTy.getSignedness() &&
         midIntTy.getSignedness() == dstIntTy.getSignedness() &&
         dstIntTy.getWidth() <= midIntTy.getWidth();
}

/// Replaces a chain of two onnx.Casts `src -> mid -> dst` with a single cast
/// `src -> dst`, whenever `mid` is unobservable: either `src -> mid` loses
/// nothing, or `mid -> dst` loses everything `src -> mid` did.
struct FoldConsecutiveCastPattern : public OpRewritePattern<ONNXCastOp> {
  using OpRewritePattern<ONNXCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXCastOp outerCast, PatternRewriter &rewriter) const override {
    auto innerCast = outerCast.getInput().getDefiningOp<ONNXCastOp>();
    if (!innerCast)
      return failure();

    if (!innerCast.getResult().hasOneUse())
      return failure();

    Type src = getElementType(innerCast.getInput().getType());
    Type mid = getElementType(innerCast.getResult().getType());
    Type dst = getElementType(outerCast.getResult().getType());

    if (!castPreservesAllValues(src, mid) &&
        !finalCastSubsumesIntermediate(src, mid, dst))
      return failure();

    rewriter.modifyOpInPlace(outerCast,
        [&]() { outerCast.getInputMutable().assign(innerCast.getInput()); });
    return success();
  }
};

} // namespace

// =============================================================================
/// Register optimization patterns as "canonicalization" patterns.
/// Add op to OpsWithCanonicalizer in gen_onnx_mlir.py to activate.
/// Please keep in alphabetical order.
// =============================================================================

/// on the ONNXAbsOp.
void ONNXAbsOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<AbsAbsPattern>(context);
}

/// on the ONNXArgMaxOp.
void ONNXArgMaxOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXArgMaxOp>>(context);
}

/// on the ONNXArgMinOp.
void ONNXArgMinOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXArgMinOp>>(context);
}

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
  if (enableUnsafeMath) {
    results.insert<FuseAddConvQDQBiasPattern>(context);
  }
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
  if (enableCastDataMovementPatterns) {
    result.insert<SwapCastConcatPattern>(context);
    result.insert<SwapCastSlicePattern>(context);
  }
  result.insert<FoldConsecutiveCastPattern>(context);
}

/// on the ONNXConcatOp.
void ONNXConcatOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RecomposeConcatPattern>(context);
  results.insert<RemoveEmptyConcatOperandsPattern>(context);
  results.insert<ConcatSingleOperandPattern>(context);
  results.insert<EliminateCarveOutAroundRotaryEmbeddingPattern>(context);
}

/// on the ONNXConvOp.
void ONNXConvOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<MaterializeDefaultConvParamsPattern>(context);
  results.insert<NormalizeConvAutoPadPattern>(context);
  results.insert<FuseConv1x1IntoConvPattern>(context);
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
  results.insert<DepthToSpaceElementwisePattern>(context);
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

/// on the ONNXExpandOp.
void ONNXExpandOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableExpandCanonicalization) {
    result.insert<ExpandToTilePattern>(context);
    result.insert<ExpandRankIncreaseToReshapeExpandPattern>(context);
  }
}

/// on the ONNXGatherElementsOp.
void ONNXGatherElementsOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableGatherElementsTileCanonicalization)
    result.insert<FuseGatherElementsTilePattern>(context);
}

/// on the ONNXGatherOp.
void ONNXGatherOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldSingletonGatherPattern>(context);
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
  results.insert<LeakyReluAlphaOneToIdentityPattern>(context);
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
  results.insert<FuseScaleIntoRotaryEmbeddingPattern>(context);
}

/// on the ONNXOrOp.
void ONNXOrOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<BinaryOpBroadcastAxisPattern<ONNXOrOp>>(context);
}

/// on the ONNXReduceL1Op.
void ONNXReduceL1Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceL1Op>>(context);
}

/// on the ONNXReduceL2Op.
void ONNXReduceL2Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceL2Op>>(context);
}

/// on the ONNXReduceLogSumOp.
void ONNXReduceLogSumOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceLogSumOp>>(context);
}

/// on the ONNXReduceLogSumExpOp.
void ONNXReduceLogSumExpOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceLogSumExpOp>>(
        context);
}

/// on the ONNXReduceMaxOp.
void ONNXReduceMaxOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceMaxOp>>(context);
}

/// on the ONNXReduceMaxV13Op.
void ONNXReduceMaxV13Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ReduceMaxV13ToLatestPattern1>(context);
  result.insert<ReduceMaxV13ToLatestPattern2>(context);
}

/// on the ONNXReduceMaxV18Op.
void ONNXReduceMaxV18Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ReduceMaxV18ToLatestPattern>(context);
}

/// on the ONNXReduceMeanOp.
void ONNXReduceMeanOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<MaterializeAbsentAxesReduceMeanPattern>(context);
  result.insert<DropUnitAxesFromReduceMeanPattern>(context);
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceMeanOp>>(context);
}

/// on the ONNXReduceMeanV13Op.
void ONNXReduceMeanV13Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<UpgradeReduceMeanV13Pattern>(context);
}

/// on the ONNXReduceMinOp.
void ONNXReduceMinOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceMinOp>>(context);
}

/// on the ONNXReduceMinV13Op.
void ONNXReduceMinV13Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ReduceMinV13ToLatestPattern1>(context);
  result.insert<ReduceMinV13ToLatestPattern2>(context);
}

/// on the ONNXReduceMinV18Op.
void ONNXReduceMinV18Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ReduceMinV18ToLatestPattern>(context);
}

/// on the ONNXReduceProdOp.
void ONNXReduceProdOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceProdOp>>(context);
}

/// on the ONNXReduceSumOp.
void ONNXReduceSumOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceSumOp>>(context);
}

/// on the ONNXReduceSumSquareOp.
void ONNXReduceSumSquareOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableKeepdimsCanonicalization)
    result.insert<KeepdimsCanonicalizationPattern<ONNXReduceSumSquareOp>>(
        context);
}

/// on the ONNXReduceSumV11Op.
void ONNXReduceSumV11Op::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<ReduceSumV11ToLatestPattern1>(context);
  result.insert<ReduceSumV11ToLatestPattern2>(context);
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
  result.insert<FuseCastBetweenReshapesPattern>(context);
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
  results.insert<SoftmaxSizeOneAxisPattern>(context);
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
void ONNXFlattenOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  if (enableReshapeCanonicalization)
    result.insert<FlattenToReshapePattern>(context);
}

/// on the ONNXSqueezeOp.
void ONNXSqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveSqueezeUnsqueezePattern>(context);
  result.insert<RemoveSqueezeCastUnsqueezePattern>(context);
  if (enableReshapeCanonicalization)
    result.insert<SqueezeToReshapePattern>(context);
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
}

/// on the ONNXUnsqueezeOp.
void ONNXUnsqueezeOp::getCanonicalizationPatterns(
    RewritePatternSet &result, MLIRContext *context) {
  result.insert<RemoveUnsqueezeSqueezePattern>(context);
  result.insert<RemoveUnsqueezeCastSqueezePattern>(context);
  result.insert<ReplaceUnsqueezeOfExpandRewritePattern>(context);
  if (enableReshapeCanonicalization)
    result.insert<UnsqueezeToReshapePattern>(context);
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

void onnx_mlir::populateQDQDataMovementCanonicalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<SinkDequantLinearOpAfterDataMovementOp<ONNXConcatOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXExpandOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXPadOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXReshapeOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXSliceOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXTileOp>,
      SinkDequantLinearOpAfterDataMovementOp<ONNXTransposeOp>>(
      patterns.getContext(), benefit);

  patterns.add<BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXConcatOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXExpandOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXPadOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXReshapeOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXSliceOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXTileOp>,
      BubbleUpQuantLinearOpBeforeDataMovementOp<ONNXTransposeOp>>(
      patterns.getContext(), benefit);
}

void onnx_mlir::populateONNXPositiveAxisCanonicalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ONNXPositiveAxisCanonicalizationPattern>(
      patterns.getContext(), benefit);
}

void onnx_mlir::configureBatchNormCanonicalization(
    bool disableBatchNormDecomposeOption) {
  disableBatchNormDecompose = disableBatchNormDecomposeOption;
}

void onnx_mlir::configureUnsafeMathCanonicalization(
    bool enableUnsafeMathOptimizations) {
  enableUnsafeMath = enableUnsafeMathOptimizations;
}

void onnx_mlir::configureReshapeCanonicalization(bool enable) {
  enableReshapeCanonicalization = enable;
}

void onnx_mlir::configurePositiveAxisCanonicalization(bool enable) {
  enablePositiveAxisCanonicalization = enable;
}

bool onnx_mlir::isPositiveAxisCanonicalizationEnabled() {
  return enablePositiveAxisCanonicalization;
}

void onnx_mlir::configureKeepdimsCanonicalization(bool enable) {
  enableKeepdimsCanonicalization = enable;
}

void onnx_mlir::configureQDQDataMovementCanonicalization(
    bool enableQDQDataMovementCanonicalizationOption) {
  enableQDQDataMovementCanonicalization =
      enableQDQDataMovementCanonicalizationOption;
}

bool onnx_mlir::isQDQDataMovementCanonicalizationEnabled() {
  return enableQDQDataMovementCanonicalization;
}

void onnx_mlir::configureExpandCanonicalization(bool enable) {
  enableExpandCanonicalization = enable;
}

void onnx_mlir::configureCastDataMovementPatterns(bool enable) {
  enableCastDataMovementPatterns = enable;
}

void onnx_mlir::configureGatherElementsTileCanonicalization(bool enable) {
  enableGatherElementsTileCanonicalization = enable;
}
