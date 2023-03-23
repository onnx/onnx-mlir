/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG 0 /* Log which functions are simdized. */

using namespace mlir;

namespace onnx_mlir {

// =============================================================================

/// Emit post-processing for variadic element-wise ops.
template <typename Op>
Value emitPostProcessingFor(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Type elementType, Value scalarResult) {
  return scalarResult;
}

// =============================================================================
// Template for functions that can be used as is

template <typename Op>
static void CheckIfCustomScalarOpIsSupported(Type elementType) {
  Type actualElementType = MathBuilder::elementTypeWithVector(elementType);
  if (actualElementType.isa<mlir::IntegerType>()) {
    if constexpr (std::is_same<ScalarIOp<Op>, CustomScalarOp>::value)
      return;
    llvm_unreachable("this op does not supports custom scalar for integers");
  }
  if (actualElementType.isa<mlir::FloatType>()) {
    if constexpr (std::is_same<ScalarFOp<Op>, CustomScalarOp>::value)
      return;
    llvm_unreachable("this op does not supports custom scalar for floats");
  }
}

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = math::TanhOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXAbsOp> {
  using FOp = math::AbsFOp;
  using IOp = math::AbsIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = arith::DivFOp;
  using IOp = arith::DivSIOp;
  using SimdEnabled = NoSimdScalarOp; // Disabled for now because of GPT2 error.
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = arith::SubFOp;
  using IOp = arith::SubIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::AndIOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::OrIOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::XOrIOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = math::ExpOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXAtanOp> {
  using FOp = KrnlAtanOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXCeilOp> {
  using FOp = math::CeilOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = math::PowFOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
struct ScalarOp<ONNXErfOp> {
  using FOp = KrnlErfOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXIsInfOp> {
  using FOp = KrnlIsInfOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXIsNaNOp> {
  using FOp = KrnlIsNaNOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXAcosOp> {
  using FOp = KrnlAcosOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXAcoshOp> {
  using FOp = KrnlAcoshOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXAsinOp> {
  using FOp = KrnlAsinOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXAsinhOp> {
  using FOp = KrnlAsinhOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXAtanhOp> {
  using FOp = KrnlAtanhOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
struct ScalarOp<ONNXTanOp> {
  using FOp = KrnlTanOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCastOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXCastOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp; // TODO: can it be simdized?
};

template <>
Value emitScalarOpFor<ONNXCastOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  // TODO: currently don't support String to * or * to String
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.cast(elementType, scalarOperands[0]);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSinhOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSinhOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  CheckIfCustomScalarOpIsSupported<ONNXSinhOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value two = create.math.constant(elementType, 2);
  Value neg = create.math.sub(zero, operand);
  Value exp = create.math.exp(operand);
  Value negExp = create.math.exp(neg);
  return create.math.div(create.math.sub(exp, negExp), two);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCoshOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXCoshOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXCoshOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  CheckIfCustomScalarOpIsSupported<ONNXCoshOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value two = create.math.constant(elementType, 2);
  Value neg = create.math.sub(zero, operand);
  Value exp = create.math.exp(operand);
  Value negExp = create.math.exp(neg);
  return create.math.div(create.math.add(exp, negExp), two);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSigmoidOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSigmoidOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  CheckIfCustomScalarOpIsSupported<ONNXSigmoidOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value neg = create.math.sub(zero, operand);
  Value negExp = create.math.exp(neg);
  return create.math.div(one, create.math.add(one, negExp));
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXHardSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXHardSigmoidOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // %Y = AddFOp(MulFOp(alpha, %X), beta)
  // %Z = SelectOp(CmpFOp(OGT, %Y, Constant 0),
  //               %Y,
  //               Constant 0)
  // ONNXHardSigmoidOp(%X) = SelectOp(CmpFOp(OLT, %Z, Constant 1),
  //                                  %Z,
  //                                  Constant 1)
  CheckIfCustomScalarOpIsSupported<ONNXHardSigmoidOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXHardSigmoidOp>(op).getAlpha().convertToFloat();
  double betaLit = dyn_cast<ONNXHardSigmoidOp>(op).getBeta().convertToFloat();
  // Create constants.
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value alpha = create.math.constant(elementType, alphaLit);
  Value beta = create.math.constant(elementType, betaLit);
  // Perform computations.
  Value add = create.math.add(create.math.mul(alpha, operand), beta);
  Value maxPredicate = create.math.sgt(add, zero);
  Value max = create.math.select(maxPredicate, add, zero);
  Value minPredicate = create.math.slt(max, one);
  return create.math.select(minPredicate, max, one);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXEluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXEluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  CheckIfCustomScalarOpIsSupported<ONNXEluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXEluOp>(op).getAlpha().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value alpha = create.math.constant(elementType, alphaLit);
  Value exp = create.math.exp(operand);
  Value lessThanZero = create.math.slt(operand, zero);
  return create.math.select(
      lessThanZero, create.math.mul(alpha, create.math.sub(exp, one)), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXReluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXReluOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value geZero = create.math.sge(operand, zero);
  return create.math.select(geZero, operand, zero);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLeakyReluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXLeakyReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  CheckIfCustomScalarOpIsSupported<ONNXLeakyReluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXLeakyReluOp>(op).getAlpha().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  auto alpha = create.math.constant(elementType, alphaLit);
  auto lessThanZero = create.math.slt(operand, zero);
  return create.math.select(
      lessThanZero, create.math.mul(alpha, operand), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXPReluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXPReluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXPReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXPReluOp(%X) = (%slope * %X) if %X < 0 else %X
  CheckIfCustomScalarOpIsSupported<ONNXPReluOp>(elementType);
  Value operand = scalarOperands[0];
  Value slope = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value lessThanZero = create.math.slt(operand, zero);
  return create.math.select(
      lessThanZero, create.math.mul(slope, operand), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSeluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSeluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSeluOp(%X) = SelectOp(CmpFOp(OGT, %X, ConstantOp 0),
  //                           MulFOp(gamma, %X),
  //                           MulFOp(gamma,
  //                                  SubFOp(MulFOp(alpha, ExpOp(%X)),
  //                                         alpha)))
  CheckIfCustomScalarOpIsSupported<ONNXSeluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXSeluOp>(op).getAlpha().convertToFloat();
  double gammaLit = dyn_cast<ONNXSeluOp>(op).getGamma().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value alpha = create.math.constant(elementType, alphaLit);
  Value gamma = create.math.constant(elementType, gammaLit);
  Value exp = create.math.exp(operand);
  Value greaterThanZero = create.math.sgt(operand, zero);
  Value select = create.math.select(greaterThanZero, operand,
      create.math.sub(create.math.mul(alpha, exp), alpha));
  return create.math.mul(gamma, select);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReciprocalOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXReciprocalOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXReciprocalOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXReciprocalOp(%X) = DivFOp(ConstantOp 1, %X)
  CheckIfCustomScalarOpIsSupported<ONNXReciprocalOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value one = create.math.constant(elementType, 1);
  return create.math.div(one, operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftplusOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSoftplusOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSoftplusOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftplusOp(%X) = LogOp(AddFOp(ExpOp(%X), ConstantOp 1))
  CheckIfCustomScalarOpIsSupported<ONNXSoftplusOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value exp = create.math.exp(operand);
  Value one = create.math.constant(elementType, 1);
  Value add = create.math.add(exp, one);
  return create.math.log(add);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftsignOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSoftsignOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSoftsignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftsignOp(%X) = DivFOp(ConstantOp 1, %X)
  CheckIfCustomScalarOpIsSupported<ONNXSoftsignOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value abs = create.math.abs(operand);
  Value one = create.math.constant(elementType, 1);
  Value add = create.math.add(abs, one);
  return create.math.div(operand, add);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSignOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSignOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXSignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXSignOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value minusOne = create.math.constant(elementType, -1);
  // %Y = SelectOP(CmpIOp(GT, %X, ConstantOp 0),
  //               ConstantOp 1,
  //               COnstantOp ShapedType::kDynamic)
  // ONNXSignOp(%X) = SelectOP(CmpIOp(EQ, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %Y)
  Value plusSelect;
  if (create.math.isUnsignedIntegerWithVector(elementType)) {
    // Unsigned integers are by definition positive.
    plusSelect = one;
  } else {
    Value plusPredicate = create.math.sgt(operand, zero);
    plusSelect = create.math.select(plusPredicate, one, minusOne);
  }
  Value zeroPredicate = create.math.eq(operand, zero);
  return create.math.select(zeroPredicate, zero, plusSelect);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMaxOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMaxOp(%X, %Y) = SelectOp(CmpFOp(OGT, %X, %Y),
  //                              %X,
  //                              %Y)
  CheckIfCustomScalarOpIsSupported<ONNXMaxOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  // could return create.math.max(lhs, rhs);
  Value cond = create.math.gt(lhs, rhs);
  return create.math.select(cond, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMinOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMinOp(%X, %Y) = SelectOp(CmpFOp(OLT, %X, %Y),
  //                              %X,
  //                              %Y)
  CheckIfCustomScalarOpIsSupported<ONNXMinOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  // could return create.math.min(lhs, rhs);
  Value cond = create.math.lt(lhs, rhs);
  return create.math.select(cond, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNegOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXNegOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXNegOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXNegOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.neg(operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLessOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLessOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXLessOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXLessOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.lt(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLessOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXLessOrEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXLessOrEqualOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.le(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXGreaterOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGreaterOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXGreaterOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXGreaterOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.gt(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGreaterOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXGreaterOrEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXGreaterOrEqualOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.ge(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXEqualOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.eq(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNotOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXNotOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = NoSimdScalarOp; // issue with bit data representation
};

template <>
Value emitScalarOpFor<ONNXNotOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXNotOp>(elementType);
  Value val = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value one = create.math.constant(elementType, 1);
  return create.math.xori(val, one);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXModOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXModOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXModOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXModOp>(elementType);
  Value dividend = scalarOperands[0];
  Value divisor = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);

  // TODO: here we assume fmod=1, what should if that is not the case?
  if (create.math.isFloatWithVector(elementType)) {
    // fmod is always 1. Behavior is like numpy.fmod.
    // The sign of the remainder is the same as the dividend.
    Value rem = create.math.rem(dividend, divisor);
#if 0
    // It seems that the copySign is not needed, from the underlying math and
    // backend test. Leave off for now as it would otherwise fail some lit
    // tests.
    return rem;
#else
    return create.math.copySign(rem, dividend);
#endif
  }
  if (create.math.isIntegerWithVector(elementType)) {
    // TODO: implement
    llvm_unreachable("not support integers at this moment since MLIR integers "
                     "are signless.");
  }
  llvm_unreachable("unsupported element type");
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMeanOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMeanOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitPostProcessingFor<ONNXMeanOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType, Value scalarResult) {
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value n = create.math.constant(elementType, op->getNumOperands());
  // Input and output type are floating point, so it is safe to use DivFOp.
  return create.math.div(scalarResult, n);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXRoundOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXRoundOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
  using SimdEnabled = SimdScalarOp;
};

template <>
Value emitScalarOpFor<ONNXRoundOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value x = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  CheckIfCustomScalarOpIsSupported<ONNXRoundOp>(elementType);
  // Use numpy algorithm for rint as follows.
  // ```
  // double y, r;
  // y = npy_floor(x);
  // r = x - y;
  //
  // if (r > 0.5) {
  //     y += 1.0;
  // }
  //
  // /* Round to nearest even */
  // if (r == 0.5) {
  //     r = y - 2.0*npy_floor(0.5*y);
  //     if (r == 1.0) {
  //         y += 1.0;
  //     }
  // }
  // return y;
  // ```
  Value one = create.math.constant(elementType, 1.0);
  Value two = create.math.constant(elementType, 2.0);
  Value half = create.math.constant(elementType, 0.5);
  Value y = create.math.floor(x);
  Value r = create.math.sub(x, y);
  // r > 0.5
  Value rGreaterThanHalf = create.math.sgt(r, half);
  Value y1 = create.math.select(rGreaterThanHalf, create.math.add(y, one), y);
  // r == 0.5: round to nearest even.
  Value y2 = create.math.mul(half, y);
  y2 = create.math.floor(y2);
  y2 = create.math.mul(y2, two);
  Value rr = create.math.sub(y, y2);
  Value rrEqualOne = create.math.eq(rr, one);
  y2 = create.math.select(rrEqualOne, create.math.add(y, one), y);

  Value rEqualHalf = create.math.eq(r, half);
  return create.math.select(rEqualHalf, y2, y1);
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be fully flattened.
//===----------------------------------------------------------------------===//

static const char *getOpName(Operation *op) {
  return op->getName().getStringRef().str().c_str();
}

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MemRefBuilder, VectorBuilder>;

template <typename ElementwiseUnaryOp>
static LogicalResult getUnaryBinarySimdCodeFullyFlattened(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXOpShapeHelper *shapeHelper, Operation *op, MemRefType outputMemRefType,
    ValueRange operands, int64_t alignment, int64_t simdUnroll) {
  Type outputElementType = outputMemRefType.getElementType();

  if (DEBUG)
    fprintf(stderr, "SIMD code for binary op %s\n", getOpName(op));

  // generate SIMD code of VL elements per vector.
  IndexExprScope allocScope(create.vec, shapeHelper->getScope());
  int64_t VL =
      create.vec.getMachineVectorLength(outputElementType) * simdUnroll;
  // Alloc memory with padding for SIMD.
  Value alloc = create.mem.alignedAllocWithSimdPadding(
      outputMemRefType, shapeHelper->getOutputDims(), simdUnroll, alignment);
  // Create flat inputs.
  llvm::SmallVector<Value, 4> flatOperands;
  for (Value oper : operands) {
    llvm::SmallVector<IndexExpr, 4> operDims;
    Value operSize;
    create.krnlIE.getShapeAsSymbols(oper, operDims);
    Value flatOper = create.mem.reshapeToFlat(oper, operDims, operSize);
    flatOperands.emplace_back(flatOper);
  }
  // Create flat output.
  Value totOutputSize;
  Value flatAlloc = create.mem.reshapeToFlat(
      alloc, shapeHelper->getOutputDims(), totOutputSize);
  IndexExpr totSize = SymbolIndexExpr(totOutputSize);
  // Create loop iteration (flattened to one dim) and blocked by mVL.
  ValueRange loopDef = create.krnl.defineLoops(1);
  ValueRange blockedLoopDef = create.krnl.block(loopDef[0], VL);
  SmallVector<IndexExpr, 1> lbs(1, LiteralIndexExpr(0));
  SmallVector<IndexExpr, 1> ubs(1, totSize);
  // Create the vector type to operate over.
  VectorType vecElementType = VectorType::get({VL}, outputElementType);
  // Iterate only over the blocks.
  create.krnl.iterateIE(loopDef, {blockedLoopDef[0]}, lbs, ubs,
      [&](KrnlBuilder &ck, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(ck);
        llvm::SmallVector<Value, 4> loadedVals;
        for (Value flatOper : flatOperands) {
          MemRefType memRefType = flatOper.getType().dyn_cast<MemRefType>();
          assert(memRefType && "expected memref");
          VectorType vecType =
              VectorType::get({VL}, memRefType.getElementType());
          Value loadedVal = create.vec.load(vecType, flatOper, loopInd);
          loadedVals.emplace_back(loadedVal);
        }
        Value loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
            rewriter, create.getLoc(), op, vecElementType, loadedVals);
        // Store result in the resulting array.
        create.vec.store(loweredOpResult, flatAlloc, loopInd);
      });
  rewriter.replaceOp(op, alloc);
  return success();
}

template <typename ElementwiseVariadicOp>
static LogicalResult getVariadicSimdCodeFullyFlattened(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXOpShapeHelper *shapeHelper, Operation *op, MemRefType outputMemRefType,
    ValueRange operands, int64_t alignment, int64_t simdUnroll) {
  Type outputElementType = outputMemRefType.getElementType();
  unsigned numArgs = op->getNumOperands();

  if (DEBUG)
    fprintf(stderr, "SIMD code for variadic op %s\n", getOpName(op));

  // generate SIMD code of VL elements per vector.
  IndexExprScope allocScope(create.vec, shapeHelper->getScope());
  int64_t VL =
      create.vec.getMachineVectorLength(outputElementType) * simdUnroll;
  // Alloc memory with padding for SIMD.
  Value alloc = create.mem.alignedAllocWithSimdPadding(
      outputMemRefType, shapeHelper->getOutputDims(), simdUnroll, alignment);
  // Create flat inputs.
  llvm::SmallVector<Value, 4> flatOperands;
  for (Value oper : operands) {
    llvm::SmallVector<IndexExpr, 4> operDims;
    Value operSize;
    create.krnlIE.getShapeAsSymbols(oper, operDims);
    Value flatOper = create.mem.reshapeToFlat(oper, operDims, operSize);
    flatOperands.emplace_back(flatOper);
  }
  // Create flat output.
  Value totOutputSize;
  Value flatAlloc = create.mem.reshapeToFlat(
      alloc, shapeHelper->getOutputDims(), totOutputSize);
  IndexExpr totSize = SymbolIndexExpr(totOutputSize);
  // Create loop iteration (flattened to one dim) and blocked by mVL.
  ValueRange loopDef = create.krnl.defineLoops(1);
  ValueRange blockedLoopDef = create.krnl.block(loopDef[0], VL);
  SmallVector<IndexExpr, 1> lbs(1, LiteralIndexExpr(0));
  SmallVector<IndexExpr, 1> ubs(1, totSize);
  // Create the vector type to operate over.
  VectorType vecElementType = VectorType::get({VL}, outputElementType);
  // Iterate only over the blocks.
  create.krnl.iterateIE(loopDef, {blockedLoopDef[0]}, lbs, ubs,
      [&](KrnlBuilder &ck, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(ck);
        llvm::SmallVector<Value, 4> loadedVals;
        // Load all the values
        for (Value flatOper : flatOperands) {
          MemRefType memRefType = flatOper.getType().dyn_cast<MemRefType>();
          assert(memRefType && "expected memref");
          VectorType vecType =
              VectorType::get({VL}, memRefType.getElementType());
          Value loadedVal = create.vec.load(vecType, flatOper, loopInd);
          loadedVals.emplace_back(loadedVal);
        }
        // Use the first operand as temporary result.
        Value accumulated = loadedVals[0];
        // Iterate over the remaining operands.
        for (unsigned i = 1; i < numArgs; ++i) {
          Value next = loadedVals[i];
          // Fold.
          accumulated = emitScalarOpFor<ElementwiseVariadicOp>(rewriter,
              create.getLoc(), op, vecElementType, {accumulated, next});
        }
        // Postprocessing (dummy op if none).
        Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
            rewriter, create.getLoc(), op, vecElementType, accumulated);
        // Store result in the resulting array.
        create.vec.store(finalResult, flatAlloc, loopInd);
      });
  rewriter.replaceOp(op, alloc);
  return success();
}

//===----------------------------------------------------------------------===//
// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering
    : public OpConversionPattern<ElementwiseUnaryOp> {
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  bool enableSIMD = false;

  ONNXElementwiseUnaryOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : OpConversionPattern<ElementwiseUnaryOp>(typeConverter, ctx),
        enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(ElementwiseUnaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    Location loc = ONNXLoc<ElementwiseUnaryOp>(op);
    Value X = operands[0];

    // If type is scalar or vector, there is no need to allocate a buffer. Just
    // call scalar computation and return the result. This is efficient when
    // elementwise ops are used as activations for ops like LSTM/GRU/RNN.
    if (!X.getType().isa<TensorType>() && !X.getType().isa<MemRefType>()) {
      Value res = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, X.getType(), {X});
      rewriter.replaceOp(op, res);
      return success();
    }

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    bool scalar = hasAllScalarValues(operands);
    if constexpr (SimdizableOp<ElementwiseUnaryOp>::value) {
      // SIMD is enabled for this operation, test if desired and feasible
      if (enableSIMD && !scalar && !hasNonIdentityLayout(operands)) {
        int64_t simdUnroll = 1;
        return getUnaryBinarySimdCodeFullyFlattened<ElementwiseUnaryOp>(
            rewriter, create, &shapeHelper, op, memRefType, operands, alignment,
            simdUnroll);
      }
    }

    // Insert an allocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        memRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!scalar) {
      ValueRange loopDef = create.krnl.defineLoops(memRefType.getRank());
      SmallVector<IndexExpr, 4> lbs(memRefType.getRank(), LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(X, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            Value loadedVal = createKrnl.load(X, loopInd);
            auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
                rewriter, loc, op, elementType, {loadedVal});
            // Store result in the resulting array.
            createKrnl.store(loweredOpResult, alloc, loopInd);
          });
    } else {
      Value loadedVal = create.krnl.load(X);
      auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, elementType, {loadedVal});
      // Store result in the resulting array.
      create.krnl.store(loweredOpResult, alloc);
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Element-wise binary ops lowering to Krnl dialect.
// This template can be used for binary ops that return a result whose type is
// different from the input type.
//===----------------------------------------------------------------------===//

template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLowering
    : public OpConversionPattern<ElementwiseBinaryOp> {
  using OpAdaptor = typename ElementwiseBinaryOp::Adaptor;
  bool enableSIMD = false;
  bool isUniBroadcasting = false;

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableSIMD, bool isUniBroadcasting = false)
      : OpConversionPattern<ElementwiseBinaryOp>(typeConverter, ctx),
        enableSIMD(enableSIMD), isUniBroadcasting(isUniBroadcasting) {}

  LogicalResult matchAndRewrite(ElementwiseBinaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ElementwiseBinaryOp>(op);

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE, nullptr, isUniBroadcasting);
    shapeHelper.computeShapeAndAssertOnFailure();

    bool scalar = hasAllScalarValues(operands);
    if constexpr (SimdizableOp<ElementwiseBinaryOp>::value) {
      // SIMD is enabled for this operation, test if desired and feasible
      if (enableSIMD && !scalar && !hasNonIdentityLayout(operands) &&
          shapeHelper.hasNoBroadcast()) {
        int64_t simdUnroll = 1;
        return getUnaryBinarySimdCodeFullyFlattened<ElementwiseBinaryOp>(
            rewriter, create, &shapeHelper, op, outputMemRefType, operands,
            alignment, simdUnroll);
      }
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!scalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            LogicalResult res = shapeHelper.getAccessExprs(
                operands[0], 0, outputAccessExprs, lhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs = createKrnl.loadIE(operands[0], lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(
                operands[1], 1, outputAccessExprs, rhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value rhs = createKrnl.loadIE(operands[1], rhsAccessExprs);

            // Apply the element-wise function.
            Value result = emitScalarOpFor<ElementwiseBinaryOp>(
                rewriter, loc, op, outputElementType, {lhs, rhs});

            // Store result in the resulting array.
            createKrnl.store(result, alloc, loopInd);
          });
    } else {
      Value lhs = create.krnl.load(operands[0]);
      Value rhs = create.krnl.load(operands[1]);

      // Apply the element-wise function.
      Value result = emitScalarOpFor<ElementwiseBinaryOp>(
          rewriter, loc, op, outputElementType, {lhs, rhs});

      // Store result in the resulting array.
      create.krnl.store(result, alloc);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering
    : public OpConversionPattern<ElementwiseVariadicOp> {
  using OpAdaptor = typename ElementwiseVariadicOp::Adaptor;
  bool enableSIMD = false;

  ONNXElementwiseVariadicOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : OpConversionPattern<ElementwiseVariadicOp>(typeConverter, ctx),
        enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(ElementwiseVariadicOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    Location loc = ONNXLoc<ElementwiseVariadicOp>(op);
    ValueRange operands = adaptor.getOperands();
    unsigned numArgs = elmsOp.getNumOperands();

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    bool scalar = hasAllScalarValues(operands);
    if constexpr (SimdizableOp<ElementwiseVariadicOp>::value) {
      // SIMD is enabled for this operation, test if desired and feasible
      if (enableSIMD && !scalar && !hasNonIdentityLayout(operands) &&
          shapeHelper.hasNoBroadcast()) {
        int64_t simdUnroll = 1;
        return getVariadicSimdCodeFullyFlattened<ElementwiseVariadicOp>(
            rewriter, create, &shapeHelper, op, outputMemRefType, operands,
            alignment, simdUnroll);
      }
    }

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);

      // Try to fuse the unary elementwise consumers
      bool isFusable = false;
      typedef Value(*EmitScalarFunc)(ConversionPatternRewriter &rewriter, Location loc, Operation *op, Type elementType, ArrayRef<Value> scalarOperands);
      SmallVector<Operation*,2> fusionList;
      SmallVector<EmitScalarFunc,2> fusionFunction;;
      Operation *currentProducer = elmsOp;
      while (currentProducer->hasOneUse()) {
        // Check the users is an elementwise op
        // I do not have a good solution for this yet
        // Assume that the candidates are Sqrt and Relu
        Operation *user;
        for (Operation *temp : currentProducer->getUsers()) {
          user = temp;
          break;
        }
        if (isa<ONNXSqrtOp>(user)) {
          fusionList.emplace_back(user);
          fusionFunction.emplace_back(emitScalarOpFor<ONNXSqrtOp>);
        } else if (isa<ONNXReluOp>(user)) {
          fusionFunction.emplace_back(emitScalarOpFor<ONNXReluOp>);
          fusionList.emplace_back(user);
        } else {
          break;
        }
        isFusable = true;
        currentProducer = user;
      }
      for(Operation *tempOp :fusionList)
        tempOp->dump();
      
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Fold over operands for each of their scalar values.
            // Obtain the first operand.
            SmallVector<IndexExpr, 4> oprdAccessExprs;
            LogicalResult res = shapeHelper.getAccessExprs(
                operands[0], 0, outputAccessExprs, oprdAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value accumulated = createKrnl.loadIE(operands[0], oprdAccessExprs);

            // Iterate over the remaining operands.
            for (unsigned i = 1; i < numArgs; ++i) {
              // Obtain the next operand.
              SmallVector<IndexExpr, 4> oprdAccessExprs;
              LogicalResult res = shapeHelper.getAccessExprs(
                  operands[i], i, outputAccessExprs, oprdAccessExprs);
              assert(succeeded(res) && "Could not compute access indices");
              Value next = createKrnl.loadIE(operands[i], oprdAccessExprs);
              // Fold.
              accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
                  rewriter, loc, op, outputElementType, {accumulated, next});
            }

            Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
                rewriter, loc, op, outputElementType, accumulated);

            // Handle the fused Ops
            for(auto *emitScalar : fusionFunction)
              finalResult = emitScalar(rewriter, loc, op, outputElementType, finalResult);

            // Store result in the resulting array.
            createKrnl.storeIE(finalResult, alloc, outputAccessExprs);
          });
      auto previous = op;
      for (Operation *fusedOp : fusionList) {
        rewriter.eraseOp(previous);
        previous = fusedOp;
      }
      rewriter.replaceOp(previous, alloc);
    } else {
      Value accumulated = create.krnl.load(operands[0]);

      // Iterate over the remaining operands.
      for (unsigned i = 1; i < numArgs; i++) {
        // Obtain the next operand.
        Value next = create.krnl.load(operands[i]);
        // Fold.
        accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
            rewriter, loc, op, outputElementType, {accumulated, next});
      }
      Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
          rewriter, loc, op, outputElementType, accumulated);
      // Store result in the resulting array.
      create.krnl.store(finalResult, alloc);
      rewriter.replaceOp(op, alloc);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// where op lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

struct ONNXWhereOpLowering : public ConversionPattern {
  bool enableSIMD = false;

  ONNXWhereOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : ConversionPattern(
            typeConverter, ONNXWhereOp::getOperationName(), 1, ctx),
        enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXWhereOp::getOperationName()),
        op->getLoc());

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    uint64_t outputRank = outputMemRefType.getRank();
    ONNXWhereOpAdaptor operandAdaptor(operands);

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(&rewriter, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the condition value.
            SmallVector<IndexExpr, 4> condAccessExprs;
            LogicalResult res =
                shapeHelper.getAccessExprs(operandAdaptor.getCondition(), 0,
                    outputAccessExprs, condAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value cond = createKrnl.loadIE(
                operandAdaptor.getCondition(), condAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            res = shapeHelper.getAccessExprs(
                operandAdaptor.getX(), 1, outputAccessExprs, lhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs =
                createKrnl.loadIE(operandAdaptor.getX(), lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(
                operandAdaptor.getY(), 2, outputAccessExprs, rhsAccessExprs);
            assert(succeeded(res) && "Could not compute access indices");
            Value rhs =
                createKrnl.loadIE(operandAdaptor.getY(), rhsAccessExprs);

            // Return lhs if cond is true else rhs.
            Value result =
                rewriter.create<arith::SelectOp>(loc, cond, lhs, rhs);

            // Store result in the resulting array.
            createKrnl.storeIE(result, alloc, outputAccessExprs);
          });
    } else {
      // Load the condition value.
      Value cond = create.krnl.load(operandAdaptor.getCondition());

      // Load the first value.
      Value lhs = create.krnl.load(operandAdaptor.getX());

      // Load the second value.
      Value rhs = create.krnl.load(operandAdaptor.getY());

      // Return lhs if cond is true else rhs.
      Value result = rewriter.create<arith::SelectOp>(loc, cond, lhs, rhs);

      // Store result in the resulting array.
      create.krnl.store(result, alloc);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXElementwiseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXAbsOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCastOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXDivOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXErfOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAcosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAcoshOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAsinOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAsinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAtanhOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXFloorOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXGreaterOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXGreaterOrEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXIsInfOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXIsNaNOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXLessOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXLessOrEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMaxOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMeanOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMinOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXModOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMulOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXNegOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXNotOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXOrOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXPowOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXRoundOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSqrtOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSubOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSumOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>, ONNXWhereOpLowering,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>>(
      typeConverter, ctx, enableSIMD);
  patterns.insert<ONNXElementwiseBinaryOpLowering<mlir::ONNXPReluOp>>(
      typeConverter, ctx, enableSIMD, /*isUniBroadcasting=*/true);
}

} // namespace onnx_mlir
