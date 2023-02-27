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

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = math::TanhOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXAbsOp> {
  using FOp = math::AbsFOp;
  using IOp = math::AbsIOp;
};

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = arith::DivFOp;
  using IOp = arith::DivSIOp;
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = arith::SubFOp;
  using IOp = arith::SubIOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = void; // Not used.
  using IOp = arith::AndIOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = void; // Not used.
  using IOp = arith::OrIOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = void; // Not used.
  using IOp = arith::XOrIOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = math::ExpOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAtanOp> {
  using FOp = KrnlAtanOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXCeilOp> {
  using FOp = math::CeilOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = math::PowFOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXErfOp> {
  using FOp = KrnlErfOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXIsInfOp> {
  using FOp = KrnlIsInfOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXIsNaNOp> {
  using FOp = KrnlIsNaNOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAcosOp> {
  using FOp = KrnlAcosOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAcoshOp> {
  using FOp = KrnlAcoshOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAsinOp> {
  using FOp = KrnlAsinOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAsinhOp> {
  using FOp = KrnlAsinhOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXAtanhOp> {
  using FOp = KrnlAtanhOp;
  using IOp = void; // Not used.
};

template <>
struct ScalarOp<ONNXTanOp> {
  using FOp = KrnlTanOp;
  using IOp = void; // Not used.
};

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCastOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXCastOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  // TODO: currently don't support String to * or * to String
  MathBuilder createMath(rewriter, loc);
  return createMath.cast(elementType, scalarOperands[0]);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSinhOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value two = createMath.constant(elementType, 2);
  Value neg = createMath.sub(zero, operand);
  Value exp = createMath.exp(operand);
  Value negExp = createMath.exp(neg);
  return createMath.div(createMath.sub(exp, negExp), two);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCoshOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXCoshOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value two = createMath.constant(elementType, 2);
  Value neg = createMath.sub(zero, operand);
  Value exp = createMath.exp(operand);
  Value negExp = createMath.exp(neg);
  return createMath.div(createMath.add(exp, negExp), two);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSigmoidOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value one = createMath.constant(elementType, 1);
  Value neg = createMath.sub(zero, operand);
  Value negExp = createMath.exp(neg);
  return createMath.div(one, createMath.add(one, negExp));
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
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
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXHardSigmoidOp>(op).getAlpha().convertToFloat();
  double betaLit = dyn_cast<ONNXHardSigmoidOp>(op).getBeta().convertToFloat();
  // Create constants.
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value one = createMath.constant(elementType, 1);
  Value alpha = createMath.constant(elementType, alphaLit);
  Value beta = createMath.constant(elementType, betaLit);
  // Perform computations.
  Value add = createMath.add(createMath.mul(alpha, operand), beta);
  Value maxPredicate = createMath.sgt(add, zero);
  Value max = createMath.select(maxPredicate, add, zero);
  Value minPredicate = createMath.slt(max, one);
  return createMath.select(minPredicate, max, one);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXEluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXEluOp>(op).getAlpha().convertToFloat();
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value one = createMath.constant(elementType, 1);
  Value alpha = createMath.constant(elementType, alphaLit);
  Value exp = createMath.exp(operand);
  Value lessThanZero = createMath.slt(operand, zero);
  return createMath.select(
      lessThanZero, createMath.mul(alpha, createMath.sub(exp, one)), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value geZero = createMath.sge(operand, zero);
  return createMath.select(geZero, operand, zero);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXLeakyReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXLeakyReluOp>(op).getAlpha().convertToFloat();
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  auto alpha = createMath.constant(elementType, alphaLit);
  auto lessThanZero = createMath.slt(operand, zero);
  return createMath.select(
      lessThanZero, createMath.mul(alpha, operand), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXPReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXPReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXPReluOp(%X) = (%slope * %X) if %X < 0 else %X
  Value operand = scalarOperands[0];
  Value slope = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value lessThanZero = createMath.slt(operand, zero);
  return createMath.select(
      lessThanZero, createMath.mul(slope, operand), operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSeluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSeluOp(%X) = SelectOp(CmpFOp(OGT, %X, ConstantOp 0),
  //                           MulFOp(gamma, %X),
  //                           MulFOp(gamma,
  //                                  SubFOp(MulFOp(alpha, ExpOp(%X)),
  //                                         alpha)))
  Value operand = scalarOperands[0];
  double alphaLit = dyn_cast<ONNXSeluOp>(op).getAlpha().convertToFloat();
  double gammaLit = dyn_cast<ONNXSeluOp>(op).getGamma().convertToFloat();
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value alpha = createMath.constant(elementType, alphaLit);
  Value gamma = createMath.constant(elementType, gammaLit);
  Value exp = createMath.exp(operand);
  Value greaterThanZero = createMath.sgt(operand, zero);
  Value select = createMath.select(greaterThanZero, operand,
      createMath.sub(createMath.mul(alpha, exp), alpha));
  return createMath.mul(gamma, select);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReciprocalOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReciprocalOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXReciprocalOp(%X) = DivFOp(ConstantOp 1, %X)
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value one = createMath.constant(elementType, 1);
  return createMath.div(one, operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftplusOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSoftplusOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftplusOp(%X) = LogOp(AddFOp(ExpOp(%X), ConstantOp 1))
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value exp = createMath.exp(operand);
  Value one = createMath.constant(elementType, 1);
  Value add = createMath.add(exp, one);
  return createMath.log(add);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSoftsignOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSoftsignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftsignOp(%X) = DivFOp(ConstantOp 1, %X)
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value abs = createMath.abs(operand);
  Value one = createMath.constant(elementType, 1);
  Value add = createMath.add(abs, one);
  return createMath.div(operand, add);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSignOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value zero = createMath.constant(elementType, 0);
  Value one = createMath.constant(elementType, 1);
  Value minusOne = createMath.constant(elementType, -1);
  // %Y = SelectOP(CmpIOp(GT, %X, ConstantOp 0),
  //               ConstantOp 1,
  //               COnstantOp ShapedType::kDynamic)
  // ONNXSignOp(%X) = SelectOP(CmpIOp(EQ, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %Y)
  Value plusSelect;
  if (createMath.isUnsignedIntegerWithVector(elementType)) {
    // Unsigned integers are by definition positive.
    plusSelect = one;
  } else {
    Value plusPredicate = createMath.sgt(operand, zero);
    plusSelect = createMath.select(plusPredicate, one, minusOne);
  }
  Value zeroPredicate = createMath.eq(operand, zero);
  return createMath.select(zeroPredicate, zero, plusSelect);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXMaxOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMaxOp(%X, %Y) = SelectOp(CmpFOp(OGT, %X, %Y),
  //                              %X,
  //                              %Y)
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  // could return createMath.max(lhs, rhs);
  Value cond = createMath.gt(lhs, rhs);
  return createMath.select(cond, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXMinOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMinOp(%X, %Y) = SelectOp(CmpFOp(OLT, %X, %Y),
  //                              %X,
  //                              %Y)
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  // could return createMath.min(lhs, rhs);
  Value cond = createMath.lt(lhs, rhs);
  return createMath.select(cond, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNegOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXNegOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  return createMath.neg(operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLessOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXLessOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  return createMath.lt(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXLessOrEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  return createMath.le(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXGreaterOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXGreaterOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];

  MathBuilder createMath(rewriter, loc);
  return createMath.gt(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXGreaterOrEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  return createMath.ge(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEqualOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);
  return createMath.eq(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNotOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXNotOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value val = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  Value one = createMath.constant(elementType, 1);
  return createMath.xori(val, one);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXModOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXModOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value dividend = scalarOperands[0];
  Value divisor = scalarOperands[1];
  MathBuilder createMath(rewriter, loc);

  // TODO: here we assume fmod=1, what should if that is not the case?
  if (createMath.isFloatWithVector(elementType)) {
    // fmod is always 1. Behavior is like numpy.fmod.
    // The sign of the remainder is the same as the dividend.
    Value rem = createMath.rem(dividend, divisor);
#if 0
    // It seems that the copySign is not needed, from the underlying math and
    // backend test. Leave off for now as it would otherwise fail some lit
    // tests.
    return rem;
#else
    return createMath.copySign(rem, dividend);
#endif
  }
  if (createMath.isIntegerWithVector(elementType)) {
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
};

template <>
Value emitPostProcessingFor<ONNXMeanOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType, Value scalarResult) {
  MathBuilder createMath(rewriter, loc);
  Value n = createMath.constant(elementType, op->getNumOperands());
  // Input and output type are floating point, so it is safe to use DivFOp.
  return createMath.div(scalarResult, n);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXRoundOp
//===----------------------------------------------------------------------===//

template <>
Value emitScalarOpFor<ONNXRoundOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value x = scalarOperands[0];
  MathBuilder createMath(rewriter, loc);
  assert(
      createMath.isFloatWithVector(elementType) && "expect float for round op");
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
  Value one = createMath.constant(elementType, 1.0);
  Value two = createMath.constant(elementType, 2.0);
  Value half = createMath.constant(elementType, 0.5);
  Value y = createMath.floor(x);
  Value r = createMath.sub(x, y);
  // r > 0.5
  Value rGreaterThanHalf = createMath.sgt(r, half);
  Value y1 = createMath.select(rGreaterThanHalf, createMath.add(y, one), y);
  // r == 0.5: round to nearest even.
  Value y2 = createMath.mul(half, y);
  y2 = createMath.floor(y2);
  y2 = createMath.mul(y2, two);
  Value rr = createMath.sub(y, y2);
  Value rrEqualOne = createMath.eq(rr, one);
  y2 = createMath.select(rrEqualOne, createMath.add(y, one), y);

  Value rEqualHalf = createMath.eq(r, half);
  return createMath.select(rEqualHalf, y2, y1);
}
// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering : public ConversionPattern {
  bool enableSIMD = false;
  ONNXElementwiseUnaryOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : ConversionPattern(
            typeConverter, ElementwiseUnaryOp::getOperationName(), 1, ctx),
        enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
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
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder,
        VectorBuilder>
        create(rewriter, loc);
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert an allocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        memRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
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

// Element-wise binary ops lowering to Krnl dialect.
// This template can be used for binary ops that return a result whose type is
// different from the input type.
//===----------------------------------------------------------------------===//
template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLowering : public ConversionPattern {
  bool enableSIMD = false;
  bool isUniBroadcasting = false;

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableSIMD, bool isUniBroadcasting = false)
      : ConversionPattern(
            typeConverter, ElementwiseBinaryOp::getOperationName(), 1, ctx),
        enableSIMD(enableSIMD), isUniBroadcasting(isUniBroadcasting) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseBinaryOp::getOperationName()),
        op->getLoc());

    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE, nullptr, isUniBroadcasting);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
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

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering : public ConversionPattern {
  bool enableSIMD = false;

  ONNXElementwiseVariadicOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : ConversionPattern(
            typeConverter, ElementwiseVariadicOp::getOperationName(), 1, ctx),
        enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(StringAttr::get(op->getContext(),
                                    ElementwiseVariadicOp::getOperationName()),
        op->getLoc());
    unsigned numArgs = op->getNumOperands();

    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type convertedType = typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);

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
            for (unsigned i = 1; i < numArgs; i++) {
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

            // Store result in the resulting array.
            createKrnl.storeIE(finalResult, alloc, outputAccessExprs);
          });
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
    }
    rewriter.replaceOp(op, alloc);
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
