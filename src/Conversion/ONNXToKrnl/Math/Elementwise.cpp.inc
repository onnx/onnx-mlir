/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp.inc - Elementwise Ops ---------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
// Contains all of the templated code needed to generate individual operations.
//
//===----------------------------------------------------------------------===//

#define _USE_MATH_DEFINES
#include <cmath>

#include "llvm/Support/Debug.h"

#include "src/Compiler/CompilerOptions.hpp"
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

template <typename Op>
static void CheckIfCustomScalarOpIsSupported(Type elementType) {
  Type actualElementType =
      MathBuilder::elementTypeOfScalarOrVector(elementType);
  if (mlir::isa<mlir::IntegerType>(actualElementType)) {
    if constexpr (std::is_same<ScalarIOp<Op>, CustomScalarOp>::value)
      return;
    llvm_unreachable("this op does not support custom scalar for integers");
  }
  if (mlir::isa<mlir::FloatType>(actualElementType)) {
    if constexpr (std::is_same<ScalarFOp<Op>, CustomScalarOp>::value)
      return;
    llvm_unreachable("this op does not support custom scalar for floats");
  }
}

// =============================================================================
// Scalar ops handling

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = math::TanhOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXTanhOp>(Type t, Operation *op) {
  return {{GenericOps::TrigHyperbolicGop, 1}};
}

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
GenOpMix getGenOpMix<ONNXAddOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
struct ScalarOp<ONNXAbsOp> {
  using FOp = math::AbsFOp;
  using IOp = math::AbsIOp;
};
template <>
GenOpMix getGenOpMix<ONNXAbsOp>(Type t, Operation *op) {
  return {{GenericOps::AbsGop, 1}};
}

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};
template <>
GenOpMix getGenOpMix<ONNXMulOp>(Type t, Operation *op) {
  return {{GenericOps::MulGop, 1}};
}

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = arith::DivFOp;
  using IOp = arith::DivSIOp;
};
template <>
GenOpMix getGenOpMix<ONNXDivOp>(Type t, Operation *op) {
  return {{GenericOps::DivGop, 1}};
}

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = arith::SubFOp;
  using IOp = arith::SubIOp;
};
template <>
GenOpMix getGenOpMix<ONNXSubOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::AndIOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::OrIOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::XOrIOp;
};

template <>
struct ScalarOp<ONNXBitwiseAndOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::AndIOp;
};

template <>
struct ScalarOp<ONNXBitwiseOrOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::OrIOp;
};

template <>
struct ScalarOp<ONNXBitwiseXorOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = arith::XOrIOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = math::ExpOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXExpOp>(Type t, Operation *op) {
  return {{GenericOps::ExpGop, 1}};
}

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
GenOpMix getGenOpMix<ONNXSumOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXCosOp>(Type t, Operation *op) {
  return {{GenericOps::TrigGop, 1}};
}

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXLogOp>(Type t, Operation *op) {
  return {{GenericOps::LogGop, 1}};
}

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXSqrtOp>(Type t, Operation *op) {
  return {{GenericOps::SqrtGop, 1}};
}

template <>
struct ScalarOp<ONNXAtanOp> {
  using FOp = KrnlAtanOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXCeilOp> {
  using FOp = math::CeilOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXCeilOp>(Type t, Operation *op) {
  return {{GenericOps::CeilGop, 1}};
}

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXFloorOp>(Type t, Operation *op) {
  return {{GenericOps::FloorGop, 1}};
}

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = NotSuportedScalarOp;
};
template <>
GenOpMix getGenOpMix<ONNXSinOp>(Type t, Operation *op) {
  return {{GenericOps::TrigGop, 1}};
}

template <>
struct ScalarOp<ONNXIsNaNOp> {
  using FOp = KrnlIsNaNOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXAcosOp> {
  using FOp = KrnlAcosOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXAcoshOp> {
  using FOp = KrnlAcoshOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXAsinOp> {
  using FOp = KrnlAsinOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXAsinhOp> {
  using FOp = KrnlAsinhOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXAtanhOp> {
  using FOp = KrnlAtanhOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXTanOp> {
  using FOp = KrnlTanOp;
  using IOp = NotSuportedScalarOp;
};

//===----------------------------------------------------------------------===//
// Scalar binary ops for lowering ONNXBitShiftOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXBitShiftOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXBitShiftOp>(Type t, Operation *op) {
  return {{GenericOps::ShiftGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXBitShiftOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value input = scalarOperands[0];
  Value shift = scalarOperands[1];

  CheckIfCustomScalarOpIsSupported<ONNXBitShiftOp>(elementType);
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);

  // If the attribute "direction" is RIGHT, this operator moves its binary
  // representation toward the right side so that the input value is effectively
  // decreased. If the attribute "direction" is LEFT, bits of binary
  // representation moves toward the left side, which results the increase of
  // its actual value

  StringRef direction = mlir::dyn_cast<ONNXBitShiftOp>(op).getDirection();

  if (direction.equals_insensitive("LEFT")) {
    return create.math.shli(input, shift);
  }
  if (direction.equals_insensitive("RIGHT")) {
    return create.math.shri(input, shift);
  }
  llvm_unreachable("unsupported case for this particular op.");
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXGeluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGeluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXGeluOp>(Type t, Operation *op) {
  StringRef approximate = mlir::dyn_cast<ONNXGeluOp>(op).getApproximate();
  if (approximate.equals_insensitive("none"))
    return {{GenericOps::ArithmeticGop, 1}, {GenericOps::ErfGop, 1},
        {GenericOps::MulGop, 3}};
  if (approximate.equals_insensitive("tanh"))
    return {{GenericOps::ArithmeticGop, 2}, {GenericOps::MulGop, 5},
        {GenericOps::TrigHyperbolicGop, 1}};
  llvm_unreachable("approximate should be only none or tanh");
}

template <>
Value emitScalarOpFor<ONNXGeluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value operand = scalarOperands[0];

  CheckIfCustomScalarOpIsSupported<ONNXGeluOp>(elementType);
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);

  // Approximate is a required attribute and should have a default value of
  // "none". "approximate = none" simply implies no approximation will take
  // place. However, "approximate" can also have a string value of "tanh" which
  // indicates the use of tanh approximation.
  StringRef approximate = mlir::dyn_cast<ONNXGeluOp>(op).getApproximate();

  // Local constants
  Value half = create.math.constant(elementType, 0.5);
  Value one = create.math.constant(elementType, 1);
  Value halfTimesOperand = create.math.mul(half, operand);

  // Approximate = none returns an output of y = 0.5 * x * (1 +
  // erf(x/sqrt(2)))
  if (approximate.equals_insensitive("none")) {
    // Create constant
    Value oneOverSqrtTwo = create.math.constant(elementType, 1 / sqrt(2));
    // Calculations
    Value mul = create.math.mul(operand, oneOverSqrtTwo);
    Value erfApprox = create.math.erf(mul);
    Value add = create.math.add(one, erfApprox);
    return create.math.mul(halfTimesOperand, add);
  }
  // Approximate = tanh returns an output of y = 0.5 * x * (1 + Tanh(sqrt(2/pi)
  // * (x + 0.044715 * x^3)))
  if (approximate.equals_insensitive("tanh")) {
    // Create constants
    Value three = create.math.constant(elementType, 3);
    Value decimal = create.math.constant(elementType, 0.044715);
    Value sqrtTwoOverPi = create.math.constant(elementType, sqrt(2 / M_PI));
    // Calculations
    Value dec = create.math.add(
        operand, create.math.mul(decimal, create.math.pow(operand, three)));
    Value tanhApprox = create.math.tanh(create.math.mul(sqrtTwoOverPi, dec));
    return create.math.mul(halfTimesOperand, create.math.add(one, tanhApprox));
  }
  llvm_unreachable("unsupported case for this particular op.");
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXIsInfOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXIsInfOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

// Currently, SIMD code gen does not support handling operations where the data
// size of the inputs is different than the data size of the outputs. As the
// output of isInf is a bit, and the input is a float, there is size reduction;
// thus this operation cannot be simdized at this time.

template <>
Value emitScalarOpFor<ONNXIsInfOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  Value operand = scalarOperands[0];
  // Get the type from the operand, as they determine the type for the compares.
  Type inputType = operand.getType();
  CheckIfCustomScalarOpIsSupported<ONNXIsInfOp>(inputType);
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value negInf = create.math.negativeInf(inputType);
  Value posInf = create.math.positiveInf(inputType);

  double detectNegAttribute =
      mlir::dyn_cast<ONNXIsInfOp>(op).getDetectNegative();
  double detectPosAttribute =
      mlir::dyn_cast<ONNXIsInfOp>(op).getDetectPositive();

  // Three different cases: Infinity, Negative Infinity and Positive Infinity
  bool detectInf = detectPosAttribute == 1 && detectNegAttribute == 1;
  bool detectNeg = detectPosAttribute == 0 && detectNegAttribute == 1;
  bool detectPos = detectPosAttribute == 1 && detectNegAttribute == 0;

  if (detectInf)
    // If infinity return true for both positive and negative infinity
    return create.math.ori(
        create.math.eq(operand, posInf), create.math.eq(operand, negInf));
  if (detectPos)
    // If positive infinity return true else false
    return create.math.eq(operand, posInf);
  if (detectNeg)
    // If negative infinity return true else false
    return create.math.eq(operand, negInf);
  llvm_unreachable("unsupported case for this particular op.");
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCastOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXCastOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
Value emitScalarOpFor<ONNXCastOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  CheckIfCustomScalarOpIsSupported<ONNXCastOp>(elementType);
  // TODO: currently don't support String to * or * to String
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  return create.math.cast(elementType, scalarOperands[0]);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXBinarizerOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXBinarizerOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXBinarizerOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 1}, {GenericOps::ConversionGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXBinarizerOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXBinarizerOp: If x > threshold ? 1 : 0;
  CheckIfCustomScalarOpIsSupported<ONNXBinarizerOp>(elementType);
  Value operand = scalarOperands[0];
  double thresholdLit =
      mlir::dyn_cast<ONNXBinarizerOp>(op).getThreshold().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value threshold = create.math.constant(elementType, thresholdLit);
  Value isGreaterThanThreshold = create.math.sgt(operand, threshold);
  return create.math.cast(elementType, isGreaterThanThreshold);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSinhOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXSinhOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 2}, {GenericOps::ExpGop, 2},
      {GenericOps::DivGop, 1}};
}

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
};

template <>
GenOpMix getGenOpMix<ONNXCoshOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 2}, {GenericOps::ExpGop, 2},
      {GenericOps::DivGop, 1}};
}

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
};

template <>
GenOpMix getGenOpMix<ONNXSigmoidOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 2}, {GenericOps::ExpGop, 1},
      {GenericOps::DivGop, 1}};
}

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
// Scalar unary ops for lowering ONNXShrinkOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXShrinkOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXShrinkOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 3}, {GenericOps::CompareGop, 2},
      {GenericOps::SelectGop, 2}};
}

template <>
Value emitScalarOpFor<ONNXShrinkOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXShrinkOp: If x < -lambd, y = x + bias;
  // If x > lambd, y = x - bias;Otherwise, y = 0.
  CheckIfCustomScalarOpIsSupported<ONNXShrinkOp>(elementType);
  Value operand = scalarOperands[0];
  double biasLit = mlir::dyn_cast<ONNXShrinkOp>(op).getBias().convertToFloat();
  double lambdLit =
      mlir::dyn_cast<ONNXShrinkOp>(op).getLambd().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value bias = create.math.constant(elementType, biasLit);
  Value lambd = create.math.constant(elementType, lambdLit);
  Value negLambd = create.math.sub(zero, lambd);
  Value isLessThanNegLambd = create.math.slt(operand, negLambd);
  Value isGreaterThanLambd = create.math.sgt(operand, lambd);
  Value isInputSubBiasOrZero = create.math.select(
      isGreaterThanLambd, create.math.sub(operand, bias), zero);
  return create.math.select(
      isLessThanNegLambd, create.math.add(operand, bias), isInputSubBiasOrZero);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXHardSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXHardSigmoidOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 3}, {GenericOps::MulGop, 1}};
}

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
  double alphaLit =
      mlir::dyn_cast<ONNXHardSigmoidOp>(op).getAlpha().convertToFloat();
  double betaLit =
      mlir::dyn_cast<ONNXHardSigmoidOp>(op).getBeta().convertToFloat();
  // Create constants.
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value alpha = create.math.constant(elementType, alphaLit);
  Value beta = create.math.constant(elementType, betaLit);
  // Perform computations.
  Value add = create.math.add(create.math.mul(alpha, operand), beta);
  Value clipLowest = create.math.max(add, zero);
  Value clipHighest = create.math.min(clipLowest, one);
  return clipHighest;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSwishOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXHardSwishOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXHardSwishOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 3}, {GenericOps::MulGop, 2}};
}

template <>
Value emitScalarOpFor<ONNXHardSwishOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // HardSwish(x) = x * max(0, min(1, (x / 6) + 0.5))
  CheckIfCustomScalarOpIsSupported<ONNXHardSwishOp>(elementType);
  Value operand = scalarOperands[0];

  // Define constants: alpha = 1/6, beta = 0.5
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  Value one = create.math.constant(elementType, 1);
  Value alpha = create.math.constant(elementType, 1.0 / 6.0);
  Value beta = create.math.constant(elementType, 0.5);

  // Compute (x / 6) + 0.5
  Value scaledX = create.math.mul(operand, alpha);
  Value shiftedX = create.math.add(scaledX, beta);

  // Apply min(1, shiftedX)
  Value minOp = create.math.min(shiftedX, one);

  // Apply max(0, minOp)
  Value maxOp = create.math.max(minOp, zero);

  // Compute final HardSwish: x * max(0, min(1, (x / 6) + 0.5))
  Value result = create.math.mul(operand, maxOp);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXEluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXEluOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}, {GenericOps::MulGop, 1},
      {GenericOps::CompareGop, 1}, {GenericOps::SelectGop, 1},
      {GenericOps::ExpGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXEluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  CheckIfCustomScalarOpIsSupported<ONNXEluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit = mlir::dyn_cast<ONNXEluOp>(op).getAlpha().convertToFloat();
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
};

template <>
GenOpMix getGenOpMix<ONNXReluOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXReluOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  return create.math.max(zero, operand);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCeLUOp
//===----------------------------------------------------------------------===//

template <>
struct ScalarOp<ONNXCeluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXCeluOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 2}, {GenericOps::MulGop, 1},
      {GenericOps::MinMaxGop, 2}, {GenericOps::ExpGop, 1},
      {GenericOps::DivGop, 1}};
}

template <>
// celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
Value emitScalarOpFor<ONNXCeluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXCeluOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);

  // Get the 'alpha' attribute from the Celu operation.
  auto celuOp = cast<ONNXCeluOp>(op);

  double alphaValue = celuOp.getAlpha().convertToDouble();

  // Create constants for 0, 1, and alpha.
  Value zero = create.math.constant(elementType, 0.0);
  Value one = create.math.constant(elementType, 1.0);
  Value alpha = create.math.constant(elementType, alphaValue);

  // Compute positive part: max(0, x)
  Value positivePart = create.math.max(zero, operand);

  // Compute negative part: alpha * (exp(x / alpha) - 1)
  Value xOverAlpha = create.math.div(operand, alpha);
  Value expVal = create.math.exp(xOverAlpha);
  Value expMinusOne = create.math.sub(expVal, one);
  Value scaled = create.math.mul(alpha, expMinusOne);

  // Combine parts: positivePart + min(0, scaled)
  Value negativePart = create.math.min(zero, scaled);
  return create.math.add(positivePart, negativePart);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXBitWiseNotOp
//===----------------------------------------------------------------------===//

template <>
struct ScalarOp<ONNXBitwiseNotOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXBitwiseNotOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXBitwiseNotOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  CheckIfCustomScalarOpIsSupported<ONNXBitwiseNotOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  // NOT(x) = XOR(x,-1)
  Value one = create.math.constant(elementType, -1);
  return create.math.xori(operand, one);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLeakyReluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXLeakyReluOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 1}, {GenericOps::SelectGop, 1},
      {GenericOps::MulGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXLeakyReluOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  CheckIfCustomScalarOpIsSupported<ONNXLeakyReluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit =
      mlir::dyn_cast<ONNXLeakyReluOp>(op).getAlpha().convertToFloat();
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
};

template <>
GenOpMix getGenOpMix<ONNXPReluOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 1}, {GenericOps::SelectGop, 1},
      {GenericOps::MulGop, 1}};
}

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
// Scalar unary ops for lowering ONNXThresholdedReluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXThresholdedReluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXThresholdedReluOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 1}, {GenericOps::SelectGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXThresholdedReluOp>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Type elementType, ArrayRef<Value> scalarOperands) {
  // ONNXThresholdedRelu: y = (x > alpha) ? x : 0
  CheckIfCustomScalarOpIsSupported<ONNXThresholdedReluOp>(elementType);
  Value operand = scalarOperands[0];
  double alphaLit =
      mlir::dyn_cast<ONNXThresholdedReluOp>(op).getAlpha().convertToFloat();
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value zero = create.math.constant(elementType, 0);
  auto alpha = create.math.constant(elementType, alphaLit);
  auto greaterThanAlpha = create.math.sgt(operand, alpha);
  return create.math.select(greaterThanAlpha, operand, zero);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSeluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXSeluOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 1}, {GenericOps::SelectGop, 1},
      {GenericOps::MulGop, 2}, {GenericOps::ArithmeticGop, 1},
      {GenericOps::ExpGop, 1}};
}

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
  double alphaLit = mlir::dyn_cast<ONNXSeluOp>(op).getAlpha().convertToFloat();
  double gammaLit = mlir::dyn_cast<ONNXSeluOp>(op).getGamma().convertToFloat();
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
};

template <>
GenOpMix getGenOpMix<ONNXReciprocalOp>(Type t, Operation *op) {
  return {{GenericOps::DivGop, 1}};
}

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
};

template <>
GenOpMix getGenOpMix<ONNXSoftplusOp>(Type t, Operation *op) {
  return {{GenericOps::ExpGop, 1}, {GenericOps::ArithmeticGop, 1},
      {GenericOps::LogGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXSoftplusOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXSoftplusOp(%X) = LoGop(AddFOp(ExpOp(%X), ConstantOp 1))
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
};

template <>
GenOpMix getGenOpMix<ONNXSoftsignOp>(Type t, Operation *op) {
  return {{GenericOps::AbsGop, 1}, {GenericOps::ArithmeticGop, 1},
      {GenericOps::DivGop, 1}};
}

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
};

template <>
GenOpMix getGenOpMix<ONNXSignOp>(Type t, Operation *op) {
  return {{GenericOps::CompareGop, 2}, {GenericOps::SelectGop, 2}};
}

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
  if (create.math.isScalarOrVectorUnsignedInteger(elementType)) {
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
// Scalar unary ops for lowering ONNXErfOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXErfOp> {
  using FOp = math::ErfOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXErfOp>(Type t, Operation *op) {
  return {{GenericOps::ErfGop, 1}};
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMaxOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXMaxOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

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
  return create.math.max(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMinOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXMinOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

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
  return create.math.min(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMishOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXMishOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXMishOp>(Type t, Operation *op) {
  return {{GenericOps::ExpGop, 1}, {GenericOps::ArithmeticGop, 1},
      {GenericOps::LogGop, 1}, {GenericOps::MulGop, 1},
      {GenericOps::TrigHyperbolicGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXMishOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  // ONNXMishOp(%X) = x * tanh(ln(1 + e^{x}))
  CheckIfCustomScalarOpIsSupported<ONNXMishOp>(elementType);
  Value operand = scalarOperands[0];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value exp = create.math.exp(operand);
  Value one = create.math.constant(elementType, 1);
  Value softplusX = create.math.log(create.math.add(exp, one));
  Value tanHSoftplusX = create.math.tanh(softplusX);
  return create.math.mul(operand, tanHSoftplusX);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNegOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXNegOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXNegOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}};
}

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
// Scalar binary ops for lowering ONNXPowOp
//===----------------------------------------------------------------------===//

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXPowOp>(Type t, Operation *op) {
  return {{GenericOps::PowGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXPowOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXPowOp>(elementType);
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  // Cover the float ^ float, int ^ int, float ^ int cases.
  return create.math.pow(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Scalar binary ops for lowering ONNXLessOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLessOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
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
// Scalar binary ops for lowering ONNXLessOrEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXLessOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
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
// Scalar binary ops for lowering ONNXGreaterOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGreaterOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
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
// Scalar binary ops for lowering ONNXGreaterOrEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGreaterOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
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
// Scalar binary ops for lowering ONNXEqualOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
Value emitScalarOpFor<ONNXEqualOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  CheckIfCustomScalarOpIsSupported<ONNXEqualOp>(elementType);
  Value results;
  Value lhs = scalarOperands[0];
  Value rhs = scalarOperands[1];
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  Type inputElemType = getElementType(lhs.getType());

  // If the two input values are a string then we want to use the krnlStrnCmp.
  // However, if the input values are a float or an int we can simply use the
  // equal function.
  if (mlir::isa<krnl::StringType>(inputElemType)) {
    Value strlenRes = create.krnl.strlen(lhs);
    Value strncmpRes = create.krnl.strncmp(lhs, rhs, strlenRes);
    // Confirm the strncmp is indeed valid. strncmp returns a value of 0 if the
    // strings are equal. So we need to verify the returned results is equal to
    // 0.
    Value zeroVal = create.math.constant(strncmpRes.getType(), 0);
    results = create.math.eq(strncmpRes, zeroVal);
  } else {
    results = create.math.eq(lhs, rhs);
  }
  return results;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXNotOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXNotOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
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
// Scalar binary ops for lowering ONNXModOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXModOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXModOp>(Type t, Operation *op) {
  return {{GenericOps::RemGop, 1}, {GenericOps::CopySignGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXModOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  CheckIfCustomScalarOpIsSupported<ONNXModOp>(elementType);
  Value dividend = scalarOperands[0];
  Value divisor = scalarOperands[1];
  MultiDialectBuilder<MathBuilder, KrnlBuilder> create(rewriter, loc);

  // TODO: here we assume fmod=1, what should if that is not the case?
  if (create.math.isScalarOrVectorFloat(elementType)) {
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
  if (create.math.isScalarOrVectorInteger(elementType)) {
    // "math.rem" returns "minus" for minus dividend and "plus or zero" for plus
    // dividend. We call the math.rem's return value "mathRemainder". However
    // onnx.ModOp should return "minus" for minus divisor and "plus or zero" for
    // plus divisor. we call the value that onnx.Mod op should return "onnxMod".
    // The following table shows mathRemainder, onnxMod and their difference
    // (=onnxMod-mathRemainder) for some inputs.
    //
    // dividend                |  7  |  7 | -7 | -7 |  6 |  6 | -6 | -6 |
    // divisor                 |  3  | -3 |  3 | -3 |  3 | -3 |  3 | -3 |
    // ------------------------+-----+----+----+----+----+----+----+----+
    // mathRemainder           |  1  |  1 | -1 | -1 |  0 |  0 |  0 |  0 |
    // onnxMod                 |  1  | -2 |  2 | -1 |  0 |  0 |  0 |  0 |
    // onnxMod - mathRemainder |  0  | -3 |  3 |  0 |  0 |  0 |  0 |  0 |
    //
    // The following code shows logic to get onnxMod from mathRemainder
    //
    // int dividend, divisor;
    // int mathRemainder = dividend % divisor;
    // int adjustedRemainder = mathRemainder + divisor;
    //
    // if ((mathRemainder != 0) && ((dividend < 0) ^ (divisor < 0))) # c.f. "^"
    // shows "exclusive or".
    //   return adjustedRemainder;
    // return mathRemainder;

    Value mathRemainder = create.math.rem(dividend, divisor);
    Value adjustedRemainder = create.math.add(mathRemainder, divisor);
    Value zero = create.math.constant(elementType, 0);
    Value falseVal = create.math.constant(rewriter.getI1Type(), 0);
    Value isMathRemainderNonZero =
        create.math.eq(create.math.eq(mathRemainder, zero), falseVal);
    Value isDividendMinus = create.math.slt(dividend, zero);
    Value isDivisorMinus = create.math.slt(divisor, zero);
    Value exclusiveOrOfIsDividendMinusAndIsDivisorMinus = create.math.eq(
        create.math.eq(isDividendMinus, isDivisorMinus), falseVal);
    Value needAdjust = create.math.andi(
        isMathRemainderNonZero, exclusiveOrOfIsDividendMinusAndIsDivisorMinus);
    Value answer =
        create.math.select(needAdjust, adjustedRemainder, mathRemainder);

    return answer;
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
GenOpMix getGenOpMix<ONNXMeanOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}, {GenericOps::DivGop, 1}};
}

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
};

// Keep in sync with with KrnlBuilder::roundEven algorithm.
template <>
GenOpMix getGenOpMix<ONNXRoundOp>(Type t, Operation *op) {
  // Custom?
  Type inputType = op->getOperand(0).getType();
  if (VectorMachineSupport::requireCustomASM(
          GenericOps::roundEvenGop, getElementTypeOrSelf(inputType)))
    return {{GenericOps::ArithmeticGop, 1}};

  // Change depending on whether KrnlBuilder use roundEven or
  // RoundEvenEmulation.
  bool useEmulation = true;
  if (useEmulation)
    return {{GenericOps::ArithmeticGop, 1}, {GenericOps::MulGop, 2},
        {GenericOps::CompareGop, 3}, {GenericOps::SelectGop, 3},
        {GenericOps::FloorGop, 2},
        {GenericOps::EstimatedVectorRegisterPressure,
            8 /* Little parallelism in code. */}};

  // Assume here that there is a hw op to handle this.
  return {{GenericOps::ArithmeticGop, 1}};
}

template <>
Value emitScalarOpFor<ONNXRoundOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  Value x = scalarOperands[0];
  MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
  CheckIfCustomScalarOpIsSupported<ONNXRoundOp>(elementType);
  return create.krnl.roundEven(x);
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXClipOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXClipOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXClipOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 2}};
}

template <>
Value emitScalarOpFor<ONNXClipOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  Value res = scalarOperands[0];
  Value minVal = scalarOperands[1];
  Value maxVal = scalarOperands[2];
  if (!isNoneValue(minVal))
    res = create.math.max(minVal, res);
  if (!isNoneValue(maxVal))
    res = create.math.min(maxVal, res);
  return res;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXDequantizeLinearOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXDequantizeLinearOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};

template <>
GenOpMix getGenOpMix<ONNXDequantizeLinearOp>(Type t, Operation *op) {
  return {{GenericOps::ArithmeticGop, 1}, {GenericOps::MulGop, 1},
      {GenericOps::ConversionGop, 2}};
}

template <>
Value emitScalarOpFor<ONNXDequantizeLinearOp>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Type elementType, ArrayRef<Value> scalarOperands) {
  MultiDialectBuilder<MathBuilder, KrnlBuilder> create(rewriter, loc);
  // Dequantization formulas: y = (x - x_zero_point) * x_scale
  // x and x_zero_point can be of type i8, ui8, int32.
  // y is of type f32.
  Value XInt = scalarOperands[0];
  Value scaleFloat = scalarOperands[1];
  Value zeroPointInt = scalarOperands[2];

  Value xFloat = create.math.cast(elementType, XInt);

  Value sub;
  if (!disableQuantZeroPoint && !isNoneValue(zeroPointInt)) {
    Value zeroPointFloat = create.math.cast(elementType, zeroPointInt);
    sub = create.math.sub(xFloat, zeroPointFloat);
  } else {
    sub = xFloat;
  }
  Value res = create.math.mul(sub, scaleFloat);
  return res;
}

} // namespace onnx_mlir
