/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#define _USE_MATH_DEFINES
#include <cmath>

#include "llvm/Support/Debug.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

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
// Template for SIMD analysis

// Helper for function that support SIMD.
static double simdAnalysis(ArrayRef<GenericOps> GOps, ArrayRef<int64_t> GOpsNum,
    Type elementType, int64_t &vectorizedOpNum, int64_t &scalarOpNum) {
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();
  return vms->getAvgVectorLength(
      GOps, GOpsNum, elementType, vectorizedOpNum, scalarOpNum);
}

// Default template for ops that do not support SIMD. For the ones that support
// SIMD, we must create an `analyzeSimdFor` template that returns the right
// values.

static double noSimd(int64_t &vectorizedOpNum, int64_t &scalarOpNum) {
  vectorizedOpNum = 0;
  scalarOpNum = 1;
  return 1.0;
}

template <typename Op>
double analyzeSimdFor(Type elementType, Operation *op, int64_t &vectorizedOpNum,
    int64_t &scalarOpNum) {
  return noSimd(vectorizedOpNum, scalarOpNum);
}

// =============================================================================
// Scalar ops handling

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = math::TanhOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXTanhOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigHyperbolicGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
double analyzeSimdFor<ONNXAddOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXAbsOp> {
  using FOp = math::AbsFOp;
  using IOp = math::AbsIOp;
};
template <>
double analyzeSimdFor<ONNXAbsOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::AbsGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};
template <>
double analyzeSimdFor<ONNXMulOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::MulGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = arith::DivFOp;
  using IOp = arith::DivSIOp;
};
template <>
double analyzeSimdFor<ONNXDivOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::DivGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = arith::SubFOp;
  using IOp = arith::SubIOp;
};
template <>
double analyzeSimdFor<ONNXSubOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
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
double analyzeSimdFor<ONNXExpOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ExpGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
double analyzeSimdFor<ONNXSumOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXCosOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXLogOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::LogGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXSqrtOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::SqrtGop}, {1}, t, von, son);
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
double analyzeSimdFor<ONNXCeilOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::CeilGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXFloorOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::FloorGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXSinOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = math::PowFOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXPowOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::PowGop}, {1}, t, von, son);
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
// Scalar unary ops for lowering ONNXGeluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXGeluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
double analyzeSimdFor<ONNXGeluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  StringRef approximate = dyn_cast<ONNXGeluOp>(op).getApproximate();
  if (approximate.equals_insensitive("none"))
    return simdAnalysis(
        {GenericOps::ArithmeticGop, GenericOps::ErfGop, GenericOps::MulGop},
        {1, 1, 3}, t, von, son);
  if (approximate.equals_insensitive("tanh"))
    return simdAnalysis({GenericOps::ArithmeticGop, GenericOps::MulGop,
                            GenericOps::TrigHyperbolicGop},
        {2, 5, 1}, t, von, son);
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
  StringRef approximate = dyn_cast<ONNXGeluOp>(op).getApproximate();

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

  double detectNegAttribute = dyn_cast<ONNXIsInfOp>(op).getDetectNegative();
  double detectPosAttribute = dyn_cast<ONNXIsInfOp>(op).getDetectPositive();

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
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSinhOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
double analyzeSimdFor<ONNXSinhOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::ExpGop, GenericOps::DivGop},
      {2, 2, 1}, t, von, son);
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
double analyzeSimdFor<ONNXCoshOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::ExpGop, GenericOps::DivGop},
      {2, 2, 1}, t, von, son);
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
double analyzeSimdFor<ONNXSigmoidOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::ExpGop, GenericOps::DivGop},
      {2, 1, 1}, t, von, son);
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
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXHardSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
double analyzeSimdFor<ONNXHardSigmoidOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop, GenericOps::MulGop,
                          GenericOps::CompareGop, GenericOps::SelectGop},
      {1, 1, 2, 2}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXEluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::MulGop, GenericOps::CompareGop,
          GenericOps::SelectGop, GenericOps::ExpGop},
      {1, 1, 1, 1, 1}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXReluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop}, {1, 1}, t, von, son);
}

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
};

template <>
double analyzeSimdFor<ONNXLeakyReluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop, GenericOps::MulGop},
      {1, 1, 1}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXPReluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop, GenericOps::MulGop},
      {1, 1, 1}, t, von, son);
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
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXSeluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};

template <>
double analyzeSimdFor<ONNXSeluOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop, GenericOps::MulGop,
          GenericOps::ArithmeticGop, GenericOps::ExpGop},
      {1, 1, 2, 1, 1}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXReciprocalOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::DivGop}, {1}, t, von, son);
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
double analyzeSimdFor<ONNXSoftplusOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ExpGop, GenericOps::ArithmeticGop, GenericOps::LogGop},
      {1, 1, 1}, t, von, son);
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
double analyzeSimdFor<ONNXSoftsignOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::AbsGop, GenericOps::ArithmeticGop, GenericOps::DivGop},
      {1, 1, 1}, t, von, son);
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
double analyzeSimdFor<ONNXSignOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop}, {2, 2}, t, von, son);
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
// Scalar unary ops for lowering ONNXErfOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXErfOp> {
  using FOp = math::ErfOp;
  using IOp = NotSuportedScalarOp;
};

template <>
double analyzeSimdFor<ONNXErfOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ErfGop}, {1}, t, von, son);
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
double analyzeSimdFor<ONNXMaxOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop}, {1, 1}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXMinOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop}, {1, 1}, t, von, son);
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
};

template <>
double analyzeSimdFor<ONNXNegOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
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
double analyzeSimdFor<ONNXModOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::RemGop, GenericOps::CopySignGop}, {1, 1}, t, von, son);
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

#ifdef DEBUG_ONNX_MOD
    create.krnl.printf("XXXX emitScalarOpFor<ONNXModOp>: dividend=", dividend);
    create.krnl.printf(", divisor=", divisor);
    create.krnl.printf(", mathReminder=", mathRemainder);
    create.krnl.printf(", adjustedReminder=", adjustedRemainder);
    create.krnl.printf(", Answer=", answer);
    create.krnl.printf("\n");
#endif

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
double analyzeSimdFor<ONNXMeanOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::DivGop}, {1, 1}, t, von, son);
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

template <>
double analyzeSimdFor<ONNXRoundOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::ArithmeticGop, GenericOps::MulGop, GenericOps::CompareGop,
          GenericOps::SelectGop, GenericOps::FloorGop},
      {4, 2, 3, 3, 2}, t, von, son);
}

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
// Scalar unary ops for lowering ONNXClipOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXClipOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
double analyzeSimdFor<ONNXClipOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  return simdAnalysis(
      {GenericOps::CompareGop, GenericOps::SelectGop}, {2, 2}, t, von, son);
}

template <>
Value emitScalarOpFor<ONNXClipOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
  Value res = scalarOperands[0];
  Value min = scalarOperands[1];
  Value max = scalarOperands[2];
  if (!isNoneValue(min)) {
    Value lessThanMin = create.math.slt(res, min); // (input[i,j,k]<min)
    res = create.math.select(lessThanMin, min, res);
  }
  if (!isNoneValue(max)) {
    Value lessThanMax = create.math.slt(res, max); // (input[i,j,k]>max)
    res = create.math.select(lessThanMax, res, max);
  }
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
double analyzeSimdFor<ONNXDequantizeLinearOp>(
    Type t, Operation *op, int64_t &von, int64_t &son) {
  // Right now, MLIR vector:splat does not support unsigned int types.
  // Thus we must disable SIMD here for now.
  return noSimd(von, son);
  // return simdAnalysis({GenericOps::ArithmeticGop, GenericOps::MulGop,
  //                        GenericOps::ConversionGop},
  //    {1, 1, 2}, t, von, son);
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

  Value zeroPointFloat = create.math.cast(elementType, zeroPointInt);
  Value xFloat = create.math.cast(elementType, XInt);
  Value sub = create.math.sub(xFloat, zeroPointFloat);
  Value res = create.math.mul(sub, scaleFloat);
  return res;
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be fully flattened.
//===----------------------------------------------------------------------===//

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MemRefBuilder, VectorBuilder>;

// Return unrolled vector length; no simd -> return 0;
// collapsedLiteralSize is ignored when we can collapse every loop iterations as
// we then rely on padding of the allocated memory to enable arbitrary output
// array simdization. When partial simd is requested, then we must ensure that
// the collapsed loop cumulative static size is a multiple of the VL.
template <typename ShapeHelperType, typename ElementwiseOp>
int64_t canBeVectorized(ShapeHelperType &shapeHelper, MDBuilder &create,
    Operation *op, MemRefType memRefType, int64_t collapsedInnermostLoops,
    int64_t &estimatedSimdLoopTripCount) {
  estimatedSimdLoopTripCount = 0; // Initially assume no SIMD.
  int64_t simdUnroll;
  int64_t uVL = 0;
  // SIMD is enabled for this operation, test if profitable.
  Type elementType = memRefType.getElementType();
  int64_t vectorizedOpNum, scalarOpNum;
  double avgSimdWidth = analyzeSimdFor<ElementwiseOp>(
      elementType, op, vectorizedOpNum, scalarOpNum);
  if (avgSimdWidth < 1.5) {
    LLVM_DEBUG(llvm::dbgs() << "  simd disabled: avg simd width  "
                            << avgSimdWidth << " too small\n");
    return 0;
  }
  // Determine empirical unroll factor.
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();

  int64_t vrNum = vms->VectorRegisterNum();
  if (vectorizedOpNum >= vrNum / 2)
    simdUnroll = 1; // TODO, it would appear to be beneficial to always have 2.
  else if (vectorizedOpNum >= vrNum / 4)
    simdUnroll = 4;
  else
    simdUnroll = 8;
  uVL = create.vec.computeSuitableUnrollFactor(vms, memRefType,
      shapeHelper.getOutputDims(), collapsedInnermostLoops, simdUnroll,
      /*canPad*/ true, estimatedSimdLoopTripCount);
  LLVM_DEBUG({
    if (uVL)
      llvm::dbgs() << "  simd enabled with vector length " << uVL << "\n";
    else
      LLVM_DEBUG(
          llvm::dbgs() << "  simd disabled, no feasible with unroll factor\n");
  });
  return uVL;
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be partially or fully flattened.
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
static LogicalResult getPartiallyFlattenedSimdCode(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXBroadcastOpShapeHelper *shapeHelper, Operation *op,
    MemRefType outputMemRefType, ValueRange operands, int64_t alignment,
    int64_t VL, int64_t collapsedInnermostLoops, bool ruledOutBroadcast,
    bool isUnaryOp, bool enableParallel) {
  Type outputElementType = outputMemRefType.getElementType();
  unsigned numArgs = op->getNumOperands();
  LLVM_DEBUG(llvm::dbgs() << "  partial SIMD code for elementwise op "
                          << op->getName() << " flattening "
                          << collapsedInnermostLoops << " inner dims\n");

  // generate SIMD code of VL elements per vector.
  IndexExprScope allocScope(create.vec, shapeHelper->getScope());
  DimsExpr outputDims;
  getIndexExprList<SymbolIndexExpr>(shapeHelper->getOutputDims(), outputDims);
  // Alloc memory with padding for SIMD.
  // For the moment, its ok to go here; if we truly have partial flattening of
  // the simd code, then we only do it with static memref size that are
  // multiples of VL * simdUnroll, so there should be no padding anyway. This
  // will change if we do partial flattening with non-multiple of VL *
  // simdUnroll.
  Value alloc = create.mem.alignedAllocWithSimdPadding(
      outputMemRefType, outputDims, VL, alignment);
  // Create flat inputs in the last innerDinNum dims.
  llvm::SmallVector<Value, 4> flatOperands;
  for (Value oper : operands) {
    if (isNoneValue(oper) || hasOneElement(oper)) {
      // If its a none / scalar, it is not meant to be flattened.
      flatOperands.emplace_back(oper);
      continue;
    }
    DimsExpr operDims, flattenOperDims;
    create.krnlIE.getShapeAsSymbols(oper, operDims);
    Value flatOper = create.mem.reshapeToFlatInnermost(
        oper, operDims, flattenOperDims, collapsedInnermostLoops);
    flatOperands.emplace_back(flatOper);
  }

  // Create flat output.
  int64_t rank = outputDims.size() - collapsedInnermostLoops + 1;
  LLVM_DEBUG(
      llvm::dbgs() << "SIMD partial flatten with loop rank " << rank << "\n");
  int64_t flattenedDim = rank - 1;
  SmallVector<IndexExpr, 4> flattenedOutputDims;
  Value flatAlloc = create.mem.reshapeToFlatInnermost(
      alloc, outputDims, flattenedOutputDims, collapsedInnermostLoops);
  // Create loop iteration (flattened to output dim - inner dim + 1) with inner
  // one and blocked by mVL.
  ValueRange loopDef = create.krnl.defineLoops(rank);
  ValueRange blockedLoopDef = create.krnl.block(loopDef[flattenedDim], VL);
  SmallVector<Value, 4> optimizedLoopDef;
  for (int64_t r = 0; r < rank - 1; ++r) {
    optimizedLoopDef.emplace_back(loopDef[r]);
  }
  optimizedLoopDef.emplace_back(blockedLoopDef[0]);
  // Create the vector type to operate over.
  VectorType vecElementType = VectorType::get({VL}, outputElementType);
  // Iterate only over the blocks.
  SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
  if (enableParallel) {
    int64_t parId;
    if (findSuitableParallelDimension(
            lbs, flattenedOutputDims, 0, std::min((int64_t)2, rank), parId)) {
      create.krnl.parallel(optimizedLoopDef[parId]);
      onnxToKrnlParallelReport(op, true, parId, lbs[parId],
          flattenedOutputDims[parId], "elementwise simd partially flattened");
    } else {
      onnxToKrnlParallelReport(op, false, -1, -1,
          "no dim with enough work in elementwise simd partially flattened");
    }
  }
  create.krnl.iterateIE(loopDef, optimizedLoopDef, lbs, flattenedOutputDims,
      [&](KrnlBuilder &ck, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder, VectorBuilder> create(ck);
        SmallVector<IndexExpr, 4> outputAccessExprs;
        getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

        llvm::SmallVector<Value, 4> loadedVals;
        // Load all the values
        for (int64_t i = 0; i < (int64_t)flatOperands.size(); ++i) {
          Value flatOper = flatOperands[i];
          if (isNoneValue(flatOper)) {
            // None, just pass it on unmodified.
            loadedVals.emplace_back(flatOper);
            continue;
          }
          MemRefType memRefType =
              mlir::dyn_cast<MemRefType>(flatOper.getType());
          assert(memRefType && "expected memref");
          VectorType vecType =
              VectorType::get({VL}, memRefType.getElementType());
          if (hasOneElementInInnermostDims(flatOper, 1)) {
            // If its a scalar, do a scalar load and splat.
            llvm::SmallVector<IndexExpr, 4> scalarAccessFct;
            if (hasOneElement(flatOper)) {
              // Not flattened, with only 1 dims, just put zeros as needed.
              int64_t scalarRank =
                  mlir::dyn_cast<ShapedType>(flatOper.getType()).getRank();
              for (int r = 0; r < scalarRank; ++r)
                scalarAccessFct.emplace_back(LiteralIndexExpr(0));

            } else {
              // Was flattened, with non 1 dims, use get access expr.
              LogicalResult res =
                  shapeHelper->getAccessExprs(flatOper, i, outputAccessExprs,
                      scalarAccessFct, /*flattened*/ true, ruledOutBroadcast);
              assert(succeeded(res) && "Could not compute access indices");
            }
            Value loadedVal = create.krnl.loadIE(flatOper, scalarAccessFct);
            Value splatValue = create.vec.splat(vecType, loadedVal);
            loadedVals.emplace_back(splatValue);
          } else {
            llvm::SmallVector<IndexExpr, 4> loadAccessFct;
            LogicalResult res =
                shapeHelper->getAccessExprs(flatOper, i, outputAccessExprs,
                    loadAccessFct, /*flattened*/ true, ruledOutBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value loadedVal =
                create.vec.loadIE(vecType, flatOper, loadAccessFct, {});
            loadedVals.emplace_back(loadedVal);
          }
        }
        Value finalResult;
        if (isUnaryOp) {
          // For unary op, we through all operands at once as the other ones are
          // scalars / none values.
          finalResult = emitScalarOpFor<OP_TYPE>(
              rewriter, create.getLoc(), op, vecElementType, loadedVals);
        } else {
          // For non-unary ops, each op is a flattened array that need to be
          // processed; process the two first ones, and then "accumulate" one
          // value at a time. Use the first operand as temporary result.
          Value accumulated = loadedVals[0];
          // Iterate over the remaining operands.
          for (unsigned i = 1; i < numArgs; ++i) {
            Value next = loadedVals[i];
            // Fold.
            accumulated = emitScalarOpFor<OP_TYPE>(rewriter, create.getLoc(),
                op, vecElementType, {accumulated, next});
          }
          // Postprocessing (dummy op if none).
          finalResult = emitPostProcessingFor<OP_TYPE>(
              rewriter, create.getLoc(), op, vecElementType, accumulated);
        }
        // Store result in the resulting array.
        create.vec.store(finalResult, flatAlloc, loopInd);
      });
  rewriter.replaceOp(op, alloc);
  return success();
}

//===----------------------------------------------------------------------===//
// Utilities for Op fusion at lowering
//===----------------------------------------------------------------------===//

// Function pointer type for the emitScalarOpFor<T> of elementwise Ops.
typedef mlir::Value (*EmitScalarFunc)(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Operation *op, mlir::Type elementType,
    mlir::ArrayRef<mlir::Value> scalarOperands);

// Utility class for Op fusion.
// Start from the root op, which is being lowered as an Elementwise Op.
// Following the def-use chain from the root op, a line of fusible ops are
// identified. The fusible Ops have to be elementwise Op, and satisfy shape
// and dependence requirement.
// The scalar operations of these fusible elementwise ops are fused into the
// loop nest generated for the root Op.
// Finally the last op is replaced with the allocated memref and the other ops
// are deleted.
// ToFix: fusion for a graph structure, not just line, could be added in future.
class OpFusionHelper {
public:
  // Constructor
  OpFusionHelper(mlir::ConversionPatternRewriter &rewriter,
      mlir::Operation *rootOp, DimAnalysis *dimAnalysis)
      : rootOp(rootOp), rewriter(rewriter), dimAnalysis(dimAnalysis),
        fusibleOps(), fuseEmitFuctions() {}

  // Fusion should not break any control dependence
  static bool isControlFlowValidForFusion(Operation *useOp, Operation *defOp);

  // Check whether the inputs of the useOp are valid for useOp to be fused
  // with the defOp. The defOp defines one of useOp's inputs.
  static bool areInputsValidForFusion(
      Operation *useOp, Operation *defOp, DimAnalysis *dimAnalysis);

  // Check whether the op is fusible along the use-def chain from the defOp.
  // If true, record the op and its scalar op.
  bool checkFusibleOp(Operation *useOp, Operation *defOp,
      SmallVector<Operation *, 2> &fusibleOps,
      SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions);

  // Find the fusible op chain from the root op
  void findFusibleOps();

  bool isFusibleListEmpty() { return fusibleOps.size() == 0; }

  // The final output type after fusion.
  // The element type of an elementwise op may be different from its inputs.
  // For example, comparison ops, and cast Op.
  MemRefType getOutputType(MemRefType outputType);

  // Generate the code for the ops to be fused
  // procedureResult is the scalar value from producer
  // alloc is used to get the tensor for the producer, which is required by
  // by the shape helper.
  Value emitFuseOps(
      Value producerResult, const Value alloc, ValueRange loopInd = {});

  void replaceOrEraseONNXOps(Value alloc);

private:
  mlir::Operation *rootOp;
  mlir::ConversionPatternRewriter &rewriter;
  DimAnalysis *dimAnalysis;
  llvm::SmallVector<mlir::Operation *, 2> fusibleOps;
  llvm::SmallVector<EmitScalarFunc, 2> fuseEmitFuctions;
}; // End of OpFusionHelper Declaration

// Check a node with type T is fusible or not.
// If true, record the op to data structure
template <typename T>
bool enqueueFusibleOpImpl(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  if (isa<T>(useOp)) {
    if (OpFusionHelper::isControlFlowValidForFusion(useOp, defOp) &&
        OpFusionHelper::areInputsValidForFusion(useOp, defOp, dimAnalysis)) {
      fusibleOps.emplace_back(useOp);
      fuseEmitFunctions.emplace_back(emitScalarOpFor<T>);
      return true;
    }
  }
  return false;
}

// Variadic template to iterate all the Elementwise Ops
template <typename T = void, class... Ts>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis);

template <typename T, class... Ts>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  if (enqueueFusibleOpImpl<T>(
          useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis))
    return true;
  return enqueueFusibleOp<Ts...>(
      useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis);
}

template <>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  return false;
}

// Give a list of Elementwise ONNX Op for the template enqueueFusibleOp
// to iterate.
bool OpFusionHelper::checkFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions) {

  // Notice: Though ClipOp is classified as unary element op in this file,
  // ClipOp requires one required input and two optional input

  return enqueueFusibleOp<
      // Unary Op
      mlir::ONNXAbsOp, mlir::ONNXAtanOp, mlir::ONNXCastOp, mlir::ONNXCeilOp,
      mlir::ONNXCosOp, mlir::ONNXCoshOp, mlir::ONNXDequantizeLinearOp,
      mlir::ONNXEluOp, mlir::ONNXErfOp, mlir::ONNXAcosOp, mlir::ONNXAcoshOp,
      mlir::ONNXAsinOp, mlir::ONNXAsinhOp, mlir::ONNXAtanhOp, mlir::ONNXExpOp,
      mlir::ONNXFloorOp, mlir::ONNXGeluOp, mlir::ONNXHardSigmoidOp,
      mlir::ONNXIsInfOp, mlir::ONNXIsNaNOp, mlir::ONNXLeakyReluOp,
      mlir::ONNXLogOp, mlir::ONNXNegOp, mlir::ONNXNotOp, mlir::ONNXReciprocalOp,
      mlir::ONNXReluOp, mlir::ONNXRoundOp, mlir::ONNXSeluOp,
      mlir::ONNXSigmoidOp, mlir::ONNXSignOp, mlir::ONNXSinOp, mlir::ONNXSinhOp,
      mlir::ONNXSoftplusOp, mlir::ONNXSoftsignOp, mlir::ONNXSqrtOp,
      mlir::ONNXTanOp, mlir::ONNXTanhOp,
      // Binary Op
      mlir::ONNXEqualOp, mlir::ONNXGreaterOp, mlir::ONNXGreaterOrEqualOp,
      mlir::ONNXLessOp, mlir::ONNXLessOrEqualOp, mlir::ONNXModOp,
      mlir::ONNXPowOp,
      // Variadic Op
      mlir::ONNXAddOp, mlir::ONNXAndOp, mlir::ONNXDivOp, mlir::ONNXMaxOp,
      mlir::ONNXMeanOp, mlir::ONNXMinOp, mlir::ONNXMulOp, mlir::ONNXOrOp,
      mlir::ONNXSubOp, mlir::ONNXSumOp, mlir::ONNXXorOp>(
      useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis);
}

// Only operations are in the same block are allowed to fuse.
// ToFix: This requirement may be too conservative.
bool OpFusionHelper::isControlFlowValidForFusion(
    Operation *useOp, Operation *defOp) {
  if (useOp->getBlock() != defOp->getBlock())
    return false;
  return true;
}

// Check whether the inputs of the useOp are valid for useOp to be fused
// with the defOp. The defOp defines one of useOp's inputs.
// If fused, the two ops will use the same loop nests and the iteration space
// be the same. Fusion is not allowed along the d-u chain that is broadcasted
// in the useOp.
// clang-format off
// Some discussion can be found at https://github.com/onnx/onnx-mlir/issues/2199
// Example of shape valid for fusion:
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// Example of shape not valid for fusion
// %1 = "onnx.Sqrt"(%0) : (tensor<24xf32>) -> tensor<24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<24xf32>, tensor<16x24xf32>) -> tensor<16x24xf32>
// clang-format on

// Since the fusion will move the read of the inputs of useOp
// to the root Op, the other inputs of the useOp (not the one from the defOp)
// should be located before the root op to make the SSA correct.
// clang-format off
// Example:
// %3 = ...
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// Example of invalid fusion without code motion:
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %3 = ...
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// clang-format on
// In this implementation, no data dependence analysis and
// code motion for fusion is implemented yet. The only other inputs allowed are
// block argument and constant to guarantee they are before the root op. It is
// assumed the canonicalization has hoisted all constant to the beginning of the
// function by fold function.
bool OpFusionHelper::areInputsValidForFusion(
    Operation *useOp, Operation *defOp, DimAnalysis *dimAnalysis) {
  // Elementwise unary operation is always fusible
  if (useOp->getOperands().size() == 1)
    return true;

  Type defOutputType = defOp->getResultTypes()[0];
  Type useOutputType = useOp->getResultTypes()[0];
  ArrayRef<int64_t> defShape = getShape(defOutputType);
  ArrayRef<int64_t> useShape = getShape(useOutputType);
  if (defShape != useShape) {
    return false;
  }

  // Check the inputs in the useOp
  for (size_t i = 0; i < useOp->getOperands().size(); i++) {
    // Only input from block argument and constant is allowed,
    // if the input does not come from the defining Op
    if (!isa<BlockArgument>(useOp->getOperand(i))) {
      Operation *input = useOp->getOperand(i).getDefiningOp();
      if (input == defOp)
        continue;
      if (!isa<ONNXConstantOp>(useOp->getOperand(i).getDefiningOp())) {
        return false;
      }
    }
  }

  // Check whether this shape of the defOp is the same as the shape of
  // the output of use op. If true, the iteration space from the defOp is
  // sufficient for the element-wise operation for the useOp, even if
  // MDBroadcast occurs in the useOp.
  // Otherwise, the loop nest should be defined according to the tensor with
  // larger space.

  // First check the rank
  if (getRank(defOutputType) != getRank(useOutputType))
    return false;

  if (dimAnalysis) {
    if (!dimAnalysis->sameShape(defOp->getResult(0), useOp->getResult(0)))
      return false;
  } else {
    // If there is no dimAnalysis, check the simplest case.
    // Static and the same shape
    if (!hasStaticShape(useOutputType))
      return false;

    ArrayRef<int64_t> inputShape = getShape(useOutputType);
    if (inputShape != defShape)
      return false;
  }

  return true;
}

// The seach for fusible ops starts from the rootOp, an elementwise operation.
// A successor op (user) is fusible if it is the only user, it is in the
// fusible elementwise op list, and its inputs are valid for fusion.
void OpFusionHelper::findFusibleOps() {
  // Direct return if fusion is disabled
  if (disableKrnlOpFusion)
    return;
  Operation *defOp = rootOp;
  while (defOp->hasOneUse()) {
    // the possible ONNX Ops.
    Operation *useOp = *defOp->getUsers().begin();
    if (!checkFusibleOp(useOp, defOp, fusibleOps, fuseEmitFuctions))
      break;

    // Current useOp becomes the defOp for the next Op
    defOp = useOp;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "op fusion: fusible ops " << fusibleOps.size() << "\n";
    rootOp->dump();
    llvm::dbgs() << "begin fusible op list\n";
    for (auto op : fusibleOps)
      op->dump();
    llvm::dbgs() << "end fusible op list\n";
  });
}

// After fusion, the only store is for the last Op.
// Therefore, the allocation should be the output of the last Op
MemRefType OpFusionHelper::getOutputType(MemRefType outputType) {
  if (!isFusibleListEmpty()) {
    Operation *lastOp = fusibleOps[fusibleOps.size() - 1];
    return MemRefType::get(outputType.getShape(),
        getElementType(lastOp->getResults()[0].getType()));
  }
  return outputType;
}

// Emit fusion Ops
Value OpFusionHelper::emitFuseOps(
    Value defOpResult, const Value alloc, ValueRange loopInd) {
  if (isFusibleListEmpty())
    return defOpResult;

  // Handle the fused Ops
  Operation *defOp = rootOp;
  for (size_t i = 0; i < fusibleOps.size(); i++) {
    Operation *useOp = fusibleOps[i];
    auto emitScalar = fuseEmitFuctions[i];
    // ToFix: use the ONNX location(ONNXLoc) of each Op.
    // The current obstacle is that no easy way to know the type of Op,
    // which is needed by ONNXLoc<T>(op).
    Location loc = useOp->getLoc();
    MDBuilder create(rewriter, loc);
    Type currentElementType = getElementType(useOp->getResults()[0].getType());

    // useOperands is used for ShapeHelper and load op.
    // getRemappedValue is needed for load op.
    SmallVector<Value, 4> useOperands;
    for (auto oper : useOp->getOperands()) {
      if (oper.getDefiningOp() != defOp)
        useOperands.emplace_back(rewriter.getRemappedValue(oper));
      else
        // Due to the op fusion, we will not generate a tensor for the current
        // oper, but only the scalar result from defOp.
        // This scalar value cannot be used to initialize ShapeHelper.
        // Instead, alloc is used because it has the same shape as the oper.
        // This is one of the prerequisites for fusion.
        // However, they may have different element type for some ops, such as
        // comparison and cast.
        // ONNXBroadcastOpShapeHelper cares only the shape of the operands,
        // not the element type.
        // In a previous implementation, the original output of defOp is used
        // with 'alloc = defOp->getResult(0)' at the end of the loop.
        // But ONNXBroadcastOpShapeHelper.computeShape() unexpectedly used
        // this parameter to generate some code (memref.dim) that is not really
        // needed. Due to this live user, the original op can not be erased.
        // This error occurred when there were more than one op with dynamic dim
        // to be fused in the previous implementation.
        // Therefore, alloc is used for all the fused op.
        useOperands.emplace_back(alloc);
    }
    // Use shape helper to generate load index
    ONNXBroadcastOpShapeHelper shapeHelper(
        useOp, useOperands, &create.krnlIE, nullptr, false);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare Values for EmitScalarOpFor<T>
    SmallVector<Value, 2> inputValues;
    for (size_t i = 0; i < useOp->getOperands().size(); i++) {
      Value inputValue = useOp->getOperand(i);
      Operation *inputOp = inputValue.getDefiningOp();
      if (inputOp == defOp) {
        inputValues.emplace_back(defOpResult);
      } else {
        IndexExprScope innerScope(create.krnl, shapeHelper.getScope());
        SmallVector<IndexExpr, 4> outputAccessExprs;
        getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);
        SmallVector<IndexExpr, 4> loadAccessExprs;
        LogicalResult res = shapeHelper.getAccessExprs(
            inputValue, i, outputAccessExprs, loadAccessExprs, true);
        assert(succeeded(res) && "Could not compute access indices");
        Value load = create.krnl.loadIE(useOperands[i], loadAccessExprs);
        inputValues.emplace_back(load);
      }
    }
    defOpResult =
        emitScalar(rewriter, loc, useOp, currentElementType, inputValues);
    defOp = useOp;
  }
  return defOpResult;
}

// Replace the last Op with allocated memref and erase the other Ops.
// When the fusible list is empty, the starting Op is the last.
void OpFusionHelper::replaceOrEraseONNXOps(Value alloc) {
  if (isFusibleListEmpty()) {
    rewriter.replaceOp(rootOp, alloc);
    return;
  }

  Operation *previous = rootOp;
  for (Operation *fusedOp : fusibleOps) {
    rewriter.eraseOp(previous);
    previous = fusedOp;
  }
  rewriter.replaceOp(previous, alloc);
}

//===----------------------------------------------------------------------===//
// End of Op Fusion Support
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering
    : public OpConversionPattern<ElementwiseUnaryOp> {
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXElementwiseUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD, bool enableParallel)
      : OpConversionPattern<ElementwiseUnaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseUnaryOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseUnaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    Location loc = ONNXLoc<ElementwiseUnaryOp>(op);
    Value X = operands[0];

    // If type is scalar or vector, there is no need to allocate a buffer.
    // Just call scalar computation and return the result. This is efficient
    // when elementwise ops are used as activations for ops like LSTM/GRU/RNN.
    if (!mlir::isa<TensorType>(X.getType()) &&
        !mlir::isa<MemRefType>(X.getType())) {
      SmallVector<Value> args;
      args.emplace_back(X);
      // Load the remaining (scalar) values.
      for (uint64_t i = 1; i < operands.size(); i++) {
        if (isNoneValue(operands[i])) {
          args.emplace_back(operands[i]);
          continue;
        }
        assert(!mlir::isa<TensorType>(operands[i].getType()) &&
               !mlir::isa<MemRefType>(operands[i].getType()) &&
               "unary expected scalar additional values");
        args.emplace_back(operands[i]);
      }
      Value res = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, X.getType(), args);
      rewriter.replaceOp(op, res);
      return success();
    }

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();
    Type elementType = outputMemRefType.getElementType();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "Look at unary elementwise op: " << op->getName() << "\n";
      op->dump();
    });
    bool isScalar = hasAllScalarValues(operands);
    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && !hasNonIdentityLayout(operands)) {
      int64_t estimatedSimdLoopTripCount;
      int64_t uVL = canBeVectorized<ONNXUnaryOpShapeHelper, ElementwiseUnaryOp>(
          shapeHelper, create, op, outputMemRefType, outputRank,
          estimatedSimdLoopTripCount);
      if (uVL > 0) {
        onnxToKrnlSimdReport(op, /*successful*/ true, uVL,
            estimatedSimdLoopTripCount, "unary fully flattened");
        return getPartiallyFlattenedSimdCode<ElementwiseUnaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            uVL, /*collapsedInnermostLoop*/ outputRank,
            /*ruleOutBroadcast*/ true, /*unary*/ true, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0,
          estimatedSimdLoopTripCount,
          "no simd in unary because could not find beneficial VL");
    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in unary because scalar/layouts");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(X, ubs);
      if (enableParallel) {
        int64_t parId;
        if (findSuitableParallelDimension(
                lbs, ubs, 0, std::min((int64_t)2, outputRank), parId)) {
          create.krnl.parallel(loopDef[parId]);
          onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
              "elementwise unary not simdized");
        } else {
          onnxToKrnlParallelReport(op, false, -1, -1,
              "no dim with enough work in elementwise unary not simdized");
        }
      }
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            SmallVector<Value> args;
            Value loadedVal = createKrnl.load(X, loopInd);
            args.emplace_back(loadedVal);
            // Load the remaining (scalar) values.
            for (uint64_t i = 1; i < operands.size(); i++) {
              if (isNoneValue(operands[i])) {
                args.emplace_back(operands[i]);
                continue;
              }
              assert(isScalarValue(operands[i]) &&
                     "unary expected scalar additional values");
              Value loadedVal = create.krnl.load(operands[i]);
              args.emplace_back(loadedVal);
            }
            auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
                rewriter, loc, op, elementType, args);
            loweredOpResult =
                opFusionHelper.emitFuseOps(loweredOpResult, alloc, loopInd);
            // Store result in the resulting array.
            createKrnl.store(loweredOpResult, alloc, loopInd);
          });
    } else {
      Value loadedVal = create.krnl.load(X);
      SmallVector<Value> args;
      args.emplace_back(loadedVal);
      // Load the remaining (scalar) values.
      for (uint64_t i = 1; i < operands.size(); i++) {
        if (isNoneValue(operands[i])) {
          args.emplace_back(operands[i]);
          continue;
        }
        assert(isScalarValue(operands[i]) &&
               "unary expected scalar additional values");
        Value loadedVal = create.krnl.load(operands[i]);
        args.emplace_back(loadedVal);
      }
      auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, elementType, args);
      loweredOpResult = opFusionHelper.emitFuseOps(loweredOpResult, alloc);
      // Store result in the resulting array.
      create.krnl.store(loweredOpResult, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);
    return success();
  }
}; // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Element-wise binary ops lowering to Krnl dialect.
// This template can be used for binary ops that return a result whose type is
// different from the input type.
//===----------------------------------------------------------------------===//

template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLowering
    : public OpConversionPattern<ElementwiseBinaryOp> {
  using OpAdaptor = typename ElementwiseBinaryOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool isUniBroadcasting = false;
  bool enableParallel = false;

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD,
      bool isUniBroadcasting = false, bool enableParallel = false)
      : OpConversionPattern<ElementwiseBinaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD),
        isUniBroadcasting(isUniBroadcasting) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseBinaryOp::getOperationName());
  }

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
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE, nullptr, isUniBroadcasting);
    shapeHelper.computeShapeAndAssertOnFailure();

    LLVM_DEBUG({
      llvm::dbgs() << "Look at binary elementwise op: " << op->getName()
                   << "\n";
      op->dump();
    });

    int64_t collapsedInnermostLoops, collapsedLiteralSize;
    IndexExpr collapsedDynamicSize;
    bool isScalar = hasAllScalarValues(operands);
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);
    bool hasManageableBroadcast =
        shapeHelper.hasManageableBroadcastForInnerDims(collapsedInnermostLoops,
            collapsedLiteralSize, collapsedDynamicSize, dimAnalysis);
    LLVM_DEBUG({
      if (hasManageableBroadcast)
        llvm::dbgs() << "  simd with manageable broadcast: inner dims "
                     << collapsedInnermostLoops << " of lit size "
                     << collapsedLiteralSize << "\n";
      else
        llvm::dbgs() << "  simd not possible, unmanageable broadcast\n";
      if (hasNoBroadcast)
        llvm::dbgs() << "  simd possible: hasNoBroadcast\n";
    });

    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && hasManageableBroadcast &&
        !hasNonIdentityLayout(operands)) {
      int64_t estimatedSimdLoopTripCount;
      int64_t uVL =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseBinaryOp>(
              shapeHelper, create, op, outputMemRefType,
              collapsedInnermostLoops, estimatedSimdLoopTripCount);
      if (uVL > 0) {
        if (collapsedInnermostLoops == (int64_t)outputRank)
          onnxToKrnlSimdReport(op, /*successful*/ true, uVL,
              estimatedSimdLoopTripCount, "binary fully flattened");
        else
          onnxToKrnlSimdReport(op, /*successful*/ true, uVL,
              estimatedSimdLoopTripCount, "binary with manageable broadcast");
        return getPartiallyFlattenedSimdCode<ElementwiseBinaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            uVL, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0,
          estimatedSimdLoopTripCount,
          "no simd in binary because no beneficial VL");
    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in binary because no manageable broadcast/layout ");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      // TODO adjust in the future
      if (enableParallel) {
        int64_t parId;
        if (findSuitableParallelDimension(
                lbs, ubs, 0, std::min((uint64_t)2, outputRank), parId)) {
          create.krnl.parallel(loopDef[parId]);
          onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
              "elementwise binary not simdized");
        } else {
          onnxToKrnlParallelReport(op, false, -1, -1,
              "no dim with enough work in elementwise binary not simdized");
        }
      }
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            LogicalResult res =
                shapeHelper.getAccessExprs(operands[0], 0, outputAccessExprs,
                    lhsAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs = createKrnl.loadIE(operands[0], lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operands[1], 1, outputAccessExprs,
                rhsAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value rhs = createKrnl.loadIE(operands[1], rhsAccessExprs);

            // Apply the element-wise function.
            Value result = emitScalarOpFor<ElementwiseBinaryOp>(
                rewriter, loc, op, outputElementType, {lhs, rhs});

            result = opFusionHelper.emitFuseOps(result, alloc, loopInd);
            // Store result in the resulting array.
            createKrnl.store(result, alloc, loopInd);
          });
    } else {
      Value lhs = create.krnl.load(operands[0]);
      Value rhs = create.krnl.load(operands[1]);

      // Apply the element-wise function.
      Value result = emitScalarOpFor<ElementwiseBinaryOp>(
          rewriter, loc, op, outputElementType, {lhs, rhs});

      result = opFusionHelper.emitFuseOps(result, alloc);
      // Store result in the resulting array.
      create.krnl.store(result, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);

    return success();
  }
}; // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering
    : public OpConversionPattern<ElementwiseVariadicOp> {
  using OpAdaptor = typename ElementwiseVariadicOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXElementwiseVariadicOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD,
      bool enableParallel)
      : OpConversionPattern<ElementwiseVariadicOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseVariadicOp::getOperationName());
  }

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
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type outputElementType = outputMemRefType.getElementType();
    uint64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "Look at variadic elementwise op: " << op->getName()
                   << "\n";
      op->dump();
    });

    int64_t collapsedInnermostLoops, collapsedLiteralSize;
    IndexExpr collapsedDynamicSize;
    bool isScalar = hasAllScalarValues(operands);
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);
    bool hasManageableBroadcast =
        shapeHelper.hasManageableBroadcastForInnerDims(collapsedInnermostLoops,
            collapsedLiteralSize, collapsedDynamicSize, dimAnalysis);
    LLVM_DEBUG({
      if (hasManageableBroadcast)
        llvm::dbgs() << "  simd with manageable broadcast: inner dims "
                     << collapsedInnermostLoops << " of lit size "
                     << collapsedLiteralSize << "\n";
      else
        llvm::dbgs() << "  simd not possible, unmanageable broadcast\n";
      if (hasNoBroadcast)
        llvm::dbgs() << "  simd possible: hasNoBroadcast\n";
    });

    if (enableSIMD && !isScalar && hasManageableBroadcast &&
        !hasNonIdentityLayout(operands)) {
      // SIMD is enabled for this operation, test if desired and feasible
      int64_t estimatedSimdLoopTripCount;
      int64_t uVL =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseVariadicOp>(
              shapeHelper, create, op, outputMemRefType,
              collapsedInnermostLoops, estimatedSimdLoopTripCount);
      if (uVL > 0) {
        if (collapsedInnermostLoops == (int64_t)outputRank)
          onnxToKrnlSimdReport(op, /*successful*/ true, uVL,
              estimatedSimdLoopTripCount, "variadic fully flattened");
        else
          onnxToKrnlSimdReport(op, /*successful*/ true, uVL,
              estimatedSimdLoopTripCount, "variadic with manageable broadcast");
        return getPartiallyFlattenedSimdCode<ElementwiseVariadicOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            uVL, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0,
          estimatedSimdLoopTripCount,
          "no simd in variadic because no beneficial VL");
    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in variadic because no manageable broadcast/layout ");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        outputMemRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);

      if (enableParallel) {
        int64_t parId;
        if (findSuitableParallelDimension(
                lbs, ubs, 0, std::min((uint64_t)2, outputRank), parId)) {
          create.krnl.parallel(loopDef[parId]);
          onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
              "elementwise variadic not simdized");
        } else {
          onnxToKrnlParallelReport(op, false, -1, -1,
              "no dim with enough work in elementwise variadic not simdized");
        }
      }
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Fold over operands for each of their scalar values.
            // Obtain the first operand.
            SmallVector<IndexExpr, 4> oprdAccessExprs;
            LogicalResult res =
                shapeHelper.getAccessExprs(operands[0], 0, outputAccessExprs,
                    oprdAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value accumulated = createKrnl.loadIE(operands[0], oprdAccessExprs);

            // Iterate over the remaining operands.
            for (unsigned i = 1; i < numArgs; ++i) {
              // Obtain the next operand.
              SmallVector<IndexExpr, 4> oprdAccessExprs;
              LogicalResult res = shapeHelper.getAccessExprs(operands[i], i,
                  outputAccessExprs, oprdAccessExprs, /*flattened dims*/ false,
                  hasNoBroadcast);
              assert(succeeded(res) && "Could not compute access indices");
              Value next = createKrnl.loadIE(operands[i], oprdAccessExprs);
              // Fold.
              accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
                  rewriter, loc, op, outputElementType, {accumulated, next});
            }

            Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
                rewriter, loc, op, outputElementType, accumulated);
            finalResult =
                opFusionHelper.emitFuseOps(finalResult, alloc, loopInd);
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
      finalResult = opFusionHelper.emitFuseOps(finalResult, alloc);
      // Store result in the resulting array.
      create.krnl.store(finalResult, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);
    return success();
  }
}; // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// where op lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

struct ONNXWhereOpLowering : public ConversionPattern {
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXWhereOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD, bool enableParallel)
      : ConversionPattern(
            typeConverter, ONNXWhereOp::getOperationName(), 1, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXWhereOp::getOperationName());
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXWhereOp::getOperationName()),
        op->getLoc());

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    uint64_t outputRank = outputMemRefType.getRank();
    ONNXWhereOpAdaptor operandAdaptor(operands);

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      if (enableParallel) {
        int64_t parId;
        if (findSuitableParallelDimension(
                lbs, ubs, 0, std::min((uint64_t)2, outputRank), parId)) {
          create.krnl.parallel(loopDef[parId]);
          onnxToKrnlParallelReport(
              op, true, parId, lbs[parId], ubs[parId], "where op not simdized");
        } else {
          onnxToKrnlParallelReport(op, false, -1, -1,
              "no dim with enough work in where op not simdized");
        }
      }
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(&rewriter, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the condition value.
            SmallVector<IndexExpr, 4> condAccessExprs;
            LogicalResult res = shapeHelper.getAccessExprs(
                operandAdaptor.getCondition(), 0, outputAccessExprs,
                condAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value cond = createKrnl.loadIE(
                operandAdaptor.getCondition(), condAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getX(), 1,
                outputAccessExprs, lhsAccessExprs, /*flattened dims*/ false,
                hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs =
                createKrnl.loadIE(operandAdaptor.getX(), lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getY(), 2,
                outputAccessExprs, rhsAccessExprs, /*flattened dims*/ false,
                hasNoBroadcast);
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
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis,
    bool enableSIMD, bool enableParallel) {
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXAbsOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseAndOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseOrOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseXorOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCastOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXDequantizeLinearOp>,
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
      ONNXElementwiseUnaryOpLowering<mlir::ONNXGeluOp>,
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
      ONNXElementwiseUnaryOpLowering<mlir::ONNXClipOp>,
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
      typeConverter, ctx, dimAnalysis, enableSIMD, enableParallel);
  patterns.insert<ONNXElementwiseBinaryOpLowering<mlir::ONNXPReluOp>>(
      typeConverter, ctx, dimAnalysis, enableSIMD, /*isUniBroadcasting=*/true,
      enableParallel);
}

} // namespace onnx_mlir
