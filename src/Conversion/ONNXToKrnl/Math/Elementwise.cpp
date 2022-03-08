/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

/// Emit post-processing for variadic element-wise ops.
template <typename Op>
Value emitPostProcessingFor(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Type elementType, Value scalarResult) {
  return scalarResult;
}

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = math::TanhOp;
  using IOp = math::TanhOp; // Not used.
};

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
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
  using FOp = arith::AndIOp; // Not used.
  using IOp = arith::AndIOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = arith::OrIOp; // Not used.
  using IOp = arith::OrIOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = arith::XOrIOp; // Not used.
  using IOp = arith::XOrIOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = math::ExpOp;
  using IOp = math::ExpOp; // Not used.
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = math::CosOp; // Not used.
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = math::LogOp; // Not used.
};

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = math::SqrtOp; // Not used.
};

template <>
struct ScalarOp<ONNXAtanOp> {
  using FOp = KrnlAtanOp;
  using IOp = KrnlAtanOp; // Not used.
};

template <>
struct ScalarOp<ONNXCeilOp> {
  using FOp = math::CeilOp;
  using IOp = math::CeilOp; // Not used.
};

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = math::FloorOp; // Not used.
};

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = math::SinOp; // Not used.
};

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = math::PowFOp;
  using IOp = math::PowFOp; // Not used.
};

template <>
struct ScalarOp<ONNXErfOp> {
  using FOp = KrnlErfOp;
  using IOp = KrnlErfOp; // Not used.
};

template <>
struct ScalarOp<ONNXAcosOp> {
  using FOp = KrnlAcosOp;
  using IOp = KrnlAcosOp; // Not used.
};

template <>
struct ScalarOp<ONNXAcoshOp> {
  using FOp = KrnlAcoshOp;
  using IOp = KrnlAcoshOp; // Not used.
};

template <>
struct ScalarOp<ONNXAsinOp> {
  using FOp = KrnlAsinOp;
  using IOp = KrnlAsinOp; // Not used.
};

template <>
struct ScalarOp<ONNXAsinhOp> {
  using FOp = KrnlAsinhOp;
  using IOp = KrnlAsinhOp; // Not used.
};

template <>
struct ScalarOp<ONNXAtanhOp> {
  using FOp = KrnlAtanhOp;
  using IOp = KrnlAtanhOp; // Not used.
};

template <>
struct ScalarOp<ONNXTanOp> {
  using FOp = KrnlTanOp;
  using IOp = KrnlTanOp; // Not used.
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

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<arith::SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<math::ExpOp>(loc, operand);
  auto negExp = rewriter.create<math::ExpOp>(loc, neg);
  auto result = rewriter.create<arith::DivFOp>(
      loc, rewriter.create<arith::SubFOp>(loc, exp, negExp), two);

  return result;
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

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two = emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<arith::SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<math::ExpOp>(loc, operand);
  auto negExp = rewriter.create<math::ExpOp>(loc, neg);
  auto result = rewriter.create<arith::DivFOp>(
      loc, rewriter.create<arith::AddFOp>(loc, exp, negExp), two);

  return result;
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

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto neg = rewriter.create<arith::SubFOp>(loc, zero, operand);
  auto negExp = rewriter.create<math::ExpOp>(loc, neg);
  auto result = rewriter.create<arith::DivFOp>(
      loc, one, rewriter.create<arith::AddFOp>(loc, one, negExp));

  return result;
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
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).alpha().convertToFloat());
  auto betaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXHardSigmoidOp>(op).beta().convertToFloat());

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto alpha = rewriter.create<arith::ConstantOp>(loc, alphaAttribute);
  auto beta = rewriter.create<arith::ConstantOp>(loc, betaAttribute);

  auto add = rewriter.create<arith::AddFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, alpha, operand), beta);
  auto maxPredicate =
      rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, add, zero);
  auto max = rewriter.create<arith::SelectOp>(loc, maxPredicate, add, zero);
  auto minPredicate =
      rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, max, one);
  auto result = rewriter.create<arith::SelectOp>(loc, minPredicate, max, one);

  return result;
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

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXEluOp>(op).alpha().convertToFloat());
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto alpha = rewriter.create<arith::ConstantOp>(loc, alphaAttribute);
  auto exp = rewriter.create<math::ExpOp>(loc, operand);
  auto lessThanZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<arith::SelectOp>(loc, lessThanZero,
      rewriter.create<arith::MulFOp>(
          loc, alpha, rewriter.create<arith::SubFOp>(loc, exp, one)),
      operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %X)
  Value operand = scalarOperands[0];

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto lessThanZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, operand, zero);
  auto result =
      rewriter.create<arith::SelectOp>(loc, lessThanZero, zero, operand);

  return result;
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

  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXLeakyReluOp>(op).alpha().convertToFloat());
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto alpha = rewriter.create<arith::ConstantOp>(loc, alphaAttribute);
  auto lessThanZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<arith::SelectOp>(loc, lessThanZero,
      rewriter.create<arith::MulFOp>(loc, alpha, operand), operand);

  return result;
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

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  Value lessThanZero, result;

  if (elementType.isa<FloatType>()) {
    lessThanZero = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, operand, zero);
    result = rewriter.create<arith::SelectOp>(loc, lessThanZero,
        rewriter.create<arith::MulFOp>(loc, slope, operand), operand);
  } else if (elementType.isa<IntegerType>()) {
    lessThanZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, operand, zero);
    result = rewriter.create<arith::SelectOp>(loc, lessThanZero,
        rewriter.create<arith::MulIOp>(loc, slope, operand), operand);
  } else
    llvm_unreachable("unsupported element type");

  return result;
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
  auto alphaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).alpha().convertToFloat());
  auto gammaAttribute = FloatAttr::get(rewriter.getF32Type(),
      llvm::dyn_cast<ONNXSeluOp>(op).gamma().convertToFloat());

  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto alpha = rewriter.create<arith::ConstantOp>(loc, alphaAttribute);
  auto gamma = rewriter.create<arith::ConstantOp>(loc, gammaAttribute);
  auto exp = rewriter.create<math::ExpOp>(loc, operand);
  auto greaterThanZero = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, operand, zero);
  auto select = rewriter.create<arith::SelectOp>(loc, greaterThanZero, operand,
      rewriter.create<arith::SubFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, alpha, exp), alpha));
  auto result = rewriter.create<arith::MulFOp>(loc, gamma, select);

  return result;
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
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto result = rewriter.create<arith::DivFOp>(loc, one, operand);

  return result;
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

  auto exp = rewriter.create<math::ExpOp>(loc, operand);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto add = rewriter.create<arith::AddFOp>(loc, exp, one);
  auto result = rewriter.create<math::LogOp>(loc, add);

  return result;
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

  auto abs = rewriter.create<math::AbsOp>(loc, operand);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto add = rewriter.create<arith::AddFOp>(loc, abs, one);
  auto result = rewriter.create<arith::DivFOp>(loc, operand, add);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSignOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXSignOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];
  // TODO: unsigned int should be supported separately?
  if (elementType.isa<IntegerType>()) {
    // %Y = SelectOP(CmpIOp(GT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               COnstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpIOp(EQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto one = emitConstantOp(rewriter, loc, elementType, 1);
    auto minusOne = emitConstantOp(rewriter, loc, elementType, -1);
    auto plusPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, operand, zero);
    auto plusSelect =
        rewriter.create<arith::SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, operand, zero);
    auto result =
        rewriter.create<arith::SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else if (elementType.isa<FloatType>()) {
    // %Y = SelectOP(CmpFOp(OGT, %X, ConstantOp 0),
    //               ConstantOp 1,
    //               ConstantOp -1)
    // ONNXSignOp(%X) = SelectOP(CmpFOp(OEQ, %X, ConstantOp 0),
    //                           ConstantOp 0,
    //                           %Y)
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto one = emitConstantOp(rewriter, loc, elementType, 1);
    auto minusOne = emitConstantOp(rewriter, loc, elementType, -1);
    auto plusPredicate = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, operand, zero);
    auto plusSelect =
        rewriter.create<arith::SelectOp>(loc, plusPredicate, one, minusOne);
    auto zeroPredicate = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, operand, zero);
    auto result =
        rewriter.create<arith::SelectOp>(loc, zeroPredicate, zero, plusSelect);
    return result;
  } else {
    llvm_unreachable("unsupported element type");
  }
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
  Value max;
  if (elementType.isa<FloatType>()) {
    max = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, lhs, rhs);
    return rewriter.create<arith::SelectOp>(loc, max, lhs, rhs);
  } else if (elementType.isa<IntegerType>()) {
    if (elementType.isUnsignedInteger()) {
      max = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, lhs, rhs);
    } else {
      max = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, lhs, rhs);
    }
  } else {
    llvm_unreachable("unsupported element type");
  }
  return rewriter.create<arith::SelectOp>(loc, max, lhs, rhs);
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
  Value min;
  if (elementType.isa<FloatType>()) {
    min = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, lhs, rhs);
  } else if (elementType.isa<IntegerType>()) {
    if (elementType.isUnsignedInteger())
      min = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, lhs, rhs);
    else
      min = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
  return rewriter.create<arith::SelectOp>(loc, min, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXAbsOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXAbsOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];

  if (elementType.isa<FloatType>()) {
    return rewriter.create<math::AbsOp>(loc, operand);
  } else if (elementType.isa<IntegerType>()) {
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    auto lessThanZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, operand, zero);
    auto negativeOperand = rewriter.create<arith::SubIOp>(loc, zero, operand);
    return rewriter.create<arith::SelectOp>(
        loc, lessThanZero, negativeOperand, operand);
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNegOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXNegOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];

  if (elementType.isa<FloatType>()) {
    return rewriter.create<arith::NegFOp>(loc, operand);
  } else if (elementType.isa<IntegerType>()) {
    auto zero = emitConstantOp(rewriter, loc, elementType, 0);
    return rewriter.create<arith::SubIOp>(loc, zero, operand); // 0 - X = -X
  } else {
    llvm_unreachable("unsupported element type");
  }
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

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
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

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLE, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
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

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
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

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
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

  Type inputType = lhs.getType();
  if (inputType.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, lhs, rhs);
  } else if (inputType.isa<IntegerType>()) {
    return rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, rhs);
  } else {
    llvm_unreachable("unsupported element type");
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNotOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXNotOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value val = scalarOperands[0];
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  Value one = emitConstantOp(rewriter, loc, elementType, 1);
  Value isZero =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, val, zero);
  return rewriter.create<arith::SelectOp>(loc, isZero, one, zero);
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

  if (elementType.isa<FloatType>()) {
    // fmod is always 1. Behavior is like numpy.fmod.
    // The sign of the remainder is the same as the dividend.
    Value rem = rewriter.create<arith::RemFOp>(loc, dividend, divisor);
    return rewriter.create<math::CopySignOp>(loc, rem, dividend);
  } else if (elementType.isa<IntegerType>()) {
    llvm_unreachable("not support integers at this moment since MLIR integers "
                     "are signless.");
  } else {
    llvm_unreachable("unsupported element type");
  }
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
  Value n = emitConstantOp(rewriter, loc, elementType, op->getNumOperands());
  // Input and output type are floating point, so it is safe to use DivFOp.
  return rewriter.create<arith::DivFOp>(loc, scalarResult, n);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXRoundOp
//===----------------------------------------------------------------------===//
template <>
Value emitScalarOpFor<ONNXRoundOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value x = scalarOperands[0];
  if (elementType.isa<FloatType>()) {
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
    Value one = emitConstantOp(rewriter, loc, elementType, 1.0);
    Value two = emitConstantOp(rewriter, loc, elementType, 2.0);
    Value half = emitConstantOp(rewriter, loc, elementType, 0.5);
    Value y = rewriter.create<math::FloorOp>(loc, x);
    Value r = rewriter.create<arith::SubFOp>(loc, x, y);

    // r > 0.5
    Value rGreaterThanHalf =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, r, half);
    Value y1 = rewriter.create<arith::SelectOp>(
        loc, rGreaterThanHalf, rewriter.create<arith::AddFOp>(loc, y, one), y);

    // r == 0.5: round to nearest even.
    Value y2 = rewriter.create<arith::MulFOp>(loc, half, y);
    y2 = rewriter.create<math::FloorOp>(loc, y2);
    y2 = rewriter.create<arith::MulFOp>(loc, y2, two);
    Value rr = rewriter.create<arith::SubFOp>(loc, y, y2);
    Value rrEqualOne =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, rr, one);
    y2 = rewriter.create<arith::SelectOp>(
        loc, rrEqualOne, rewriter.create<arith::AddFOp>(loc, y, one), y);

    Value rEqualHalf =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, r, half);
    return rewriter.create<arith::SelectOp>(loc, rEqualHalf, y2, y1);
  } else {
    llvm_unreachable("unsupported element type");
  }
}

// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXElementwiseUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ElementwiseUnaryOp>(op);
    auto X = operands[0];

    // If type is scalar or vector, there is no need to allocate a buffer. Just
    // call scalar computation and return the result. This is efficient when
    // elementwise ops are used as activations for ops like LSTM/GRU/RNN.
    if (!X.getType().isa<TensorType>() && !X.getType().isa<MemRefType>()) {
      Value res = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, X.getType(), {X});
      rewriter.replaceOp(op, res);
      return success();
    }

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);

    SmallVector<Value, 4> loopIVs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, memRefType.getRank());
      loops.createDefineAndIterateOp(X);
      Block *iterationBlock = loops.getIterateBlock();

      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);

      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        loopIVs.push_back(arg);
    }

    KrnlBuilder createKrnl(rewriter, loc);
    Value loadedVal = createKrnl.load(X, loopIVs);
    auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
        rewriter, loc, op, memRefType.getElementType(), {loadedVal});
    // Store result in the resulting array.
    createKrnl.store(loweredOpResult, alloc, loopIVs);

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
  bool isUniBroadcasting = false;

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool isUniBroadcasting = false)
      : ConversionPattern(
            typeConverter, ElementwiseBinaryOp::getOperationName(), 1, ctx) {
    this->isUniBroadcasting = isUniBroadcasting;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = NameLoc::get(StringAttr::get(op->getContext(),
                                ElementwiseBinaryOp::getOperationName()),
        op->getLoc());
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputElementType = outputMemRefType.getElementType();
    auto outputRank = outputMemRefType.getRank();

    // Shape helper.
    ONNXGenericOpBroadcastedShapeHelper shapeHelper(op, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex, /*in scope*/ nullptr,
        isUniBroadcasting);
    DimsExpr empty;
    auto shapecomputed = shapeHelper.computeShape(operands, empty);
    assert(succeeded(shapecomputed));
    // Scope for krnl ops
    IndexExprScope outerScope(&rewriter, shapeHelper.scope);
    KrnlBuilder createKrnl(rewriter, loc);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Emit main computation.
    SmallVector<IndexExpr, 4> outputAccessExprs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, outputRank);
      loops.createDefineAndIterateOp(alloc);
      Block *iterationBlock = loops.getIterateBlock();
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        outputAccessExprs.emplace_back(DimIndexExpr(arg));
    }

    // Load the first value.
    SmallVector<IndexExpr, 4> lhsAccessExprs;
    LogicalResult res = shapeHelper.GetAccessExprs(
        operands[0], 0, outputAccessExprs, lhsAccessExprs);
    assert(succeeded(res));
    Value lhs = createKrnl.loadIE(operands[0], lhsAccessExprs);

    // Load the second value.
    SmallVector<IndexExpr, 4> rhsAccessExprs;
    res = shapeHelper.GetAccessExprs(
        operands[1], 1, outputAccessExprs, rhsAccessExprs);
    assert(succeeded(res));
    Value rhs = createKrnl.loadIE(operands[1], rhsAccessExprs);

    // Apply the element-wise function.
    Value result = emitScalarOpFor<ElementwiseBinaryOp>(
        rewriter, loc, op, outputElementType, {lhs, rhs});

    // Store result in the resulting array.
    createKrnl.storeIE(result, alloc, outputAccessExprs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering : public ConversionPattern {
  ONNXElementwiseVariadicOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = NameLoc::get(StringAttr::get(op->getContext(),
                                ElementwiseVariadicOp::getOperationName()),
        op->getLoc());
    auto numArgs = op->getNumOperands();
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputElementType = outputMemRefType.getElementType();
    auto outputRank = outputMemRefType.getRank();

    // Shape helper.
    ONNXGenericOpBroadcastedShapeHelper shapeHelper(op, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);

    // The following call is used to force no broadcasting check at runtime
    // Even when the dim is unknown at compile time
    DimsExpr empty;
    LogicalResult shapecomputed = shapeHelper.computeShape(operands, empty);
    assert(succeeded(shapecomputed));
    IndexExprScope outerScope(&rewriter, shapeHelper.scope);
    KrnlBuilder createKrnl(rewriter, loc);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Emit main computation.
    SmallVector<IndexExpr, 4> outputAccessExprs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, outputRank);
      loops.createDefineAndIterateOp(alloc);

      Block *iterationBlock = loops.getIterateBlock();
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        outputAccessExprs.emplace_back(DimIndexExpr(arg));
    }

    // Fold over operands for each of their scalar values.
    // Obtain the first operand.
    SmallVector<IndexExpr, 4> oprdAccessExprs;
    LogicalResult res = shapeHelper.GetAccessExprs(
        operands[0], 0, outputAccessExprs, oprdAccessExprs);
    assert(succeeded(res));
    Value accumulated = createKrnl.loadIE(operands[0], oprdAccessExprs);

    // Iterate over the remaining operands.
    for (unsigned i = 1; i < numArgs; i++) {
      // Obtain the next operand.
      SmallVector<IndexExpr, 4> oprdAccessExprs;
      LogicalResult res = shapeHelper.GetAccessExprs(
          operands[i], i, outputAccessExprs, oprdAccessExprs);
      assert(succeeded(res));
      Value next = createKrnl.loadIE(operands[i], oprdAccessExprs);
      // Fold.
      accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
          rewriter, loc, op, outputElementType, {accumulated, next});
    }

    Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
        rewriter, loc, op, outputElementType, accumulated);

    // Store result in the resulting array.
    createKrnl.storeIE(finalResult, alloc, outputAccessExprs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// where op lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
struct ONNXWhereOpLowering : public ConversionPattern {
  ONNXWhereOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXWhereOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXWhereOp::getOperationName()),
        op->getLoc());
    auto outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputRank = outputMemRefType.getRank();

    // Shape helper.
    ONNXGenericOpBroadcastedShapeHelper shapeHelper(op, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    DimsExpr empty;
    auto shapecomputed = shapeHelper.computeShape(operands, empty);
    assert(succeeded(shapecomputed));
    // Scope for krnl ops
    IndexExprScope outerScope(&rewriter, shapeHelper.scope);
    KrnlBuilder createKrnl(rewriter, loc);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Emit main computation.
    SmallVector<IndexExpr, 4> outputAccessExprs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, outputRank);
      loops.createDefineAndIterateOp(alloc);
      Block *iterationBlock = loops.getIterateBlock();
      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);
      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        outputAccessExprs.emplace_back(DimIndexExpr(arg));
    }

    // Load the condition value.
    SmallVector<IndexExpr, 4> condAccessExprs;
    LogicalResult res = shapeHelper.GetAccessExprs(
        operands[0], 0, outputAccessExprs, condAccessExprs);
    assert(succeeded(res));
    Value cond = createKrnl.loadIE(operands[0], condAccessExprs);

    // Load the first value.
    SmallVector<IndexExpr, 4> lhsAccessExprs;
    res = shapeHelper.GetAccessExprs(
        operands[1], 1, outputAccessExprs, lhsAccessExprs);
    assert(succeeded(res));
    Value lhs = createKrnl.loadIE(operands[1], lhsAccessExprs);

    // Load the second value.
    SmallVector<IndexExpr, 4> rhsAccessExprs;
    res = shapeHelper.GetAccessExprs(
        operands[2], 2, outputAccessExprs, rhsAccessExprs);
    assert(succeeded(res));
    Value rhs = createKrnl.loadIE(operands[2], rhsAccessExprs);

    // Return lhs if cond is true else rhs.
    Value result = rewriter.create<arith::SelectOp>(loc, cond, lhs, rhs);

    // Store result in the resulting array.
    createKrnl.storeIE(result, alloc, outputAccessExprs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXElementwiseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
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
      ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>>(typeConverter, ctx);
  patterns.insert<ONNXElementwiseBinaryOpLowering<mlir::ONNXPReluOp>>(
      typeConverter, ctx, /*isUniBroadcasting=*/true);
}
