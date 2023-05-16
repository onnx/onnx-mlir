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

#include "llvm/Support/Debug.h"

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

// =============================================================================
// Template for SIMD analysis

// Helper for function that support SIMD.
static double simdAnalysis(ArrayRef<GenericOps> Gops, ArrayRef<int64_t> GopsNum,
    Type elementType, int64_t &vectorizedOpNum, int64_t &scalarOpNum) {
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();
  return vms->getAvgVectorLength(
      Gops, GopsNum, elementType, vectorizedOpNum, scalarOpNum);
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
double analyzeSimdFor(
    Type elementType, int64_t &vectorizedOpNum, int64_t &scalarOpNum) {
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
double analyzeSimdFor<ONNXTanhOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigHyperbolicGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
double analyzeSimdFor<ONNXAddOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXAbsOp> {
  using FOp = math::AbsFOp;
  using IOp = math::AbsIOp;
};
template <>
double analyzeSimdFor<ONNXAbsOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::AbsGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = arith::MulFOp;
  using IOp = arith::MulIOp;
};
template <>
double analyzeSimdFor<ONNXMulOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::MulGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = arith::DivFOp;
  using IOp = arith::DivSIOp;
};
template <>
double analyzeSimdFor<ONNXDivOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::DivGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = arith::SubFOp;
  using IOp = arith::SubIOp;
};
template <>
double analyzeSimdFor<ONNXSubOp>(Type t, int64_t &von, int64_t &son) {
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
  using FOp = arith::AndIOp; // Not used.
  using IOp = arith::AndIOp;
};

template <>
struct ScalarOp<ONNXBitwiseOrOp> {
  using FOp = arith::OrIOp; // Not used.
  using IOp = arith::OrIOp;
};

template <>
struct ScalarOp<ONNXBitwiseXorOp> {
  using FOp = arith::XOrIOp; // Not used.
  using IOp = arith::XOrIOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = math::ExpOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXExpOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ExpGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = arith::AddFOp;
  using IOp = arith::AddIOp;
};
template <>
double analyzeSimdFor<ONNXSumOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::ArithmeticGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = math::CosOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXCosOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = math::LogOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXLogOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::LogGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSqrtOp> {
  using FOp = math::SqrtOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXSqrtOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXCeilOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::CeilGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXFloorOp> {
  using FOp = math::FloorOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXFloorOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::FloorGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXSinOp> {
  using FOp = math::SinOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXSinOp>(Type t, int64_t &von, int64_t &son) {
  return simdAnalysis({GenericOps::TrigGop}, {1}, t, von, son);
}

template <>
struct ScalarOp<ONNXPowOp> {
  using FOp = math::PowFOp;
  using IOp = NotSuportedScalarOp;
};
template <>
double analyzeSimdFor<ONNXPowOp>(Type t, int64_t &von, int64_t &son) {
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
// Scalar unary ops for lowering ONNXIsInfOp
//===----------------------------------------------------------------------===//
template <>
struct ScalarOp<ONNXIsInfOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};

template <>
Value emitScalarOpFor<ONNXIsInfOp>(ConversionPatternRewriter &rewriter,
    Location loc, Operation *op, Type elementType,
    ArrayRef<Value> scalarOperands) {

  CheckIfCustomScalarOpIsSupported<ONNXIsInfOp>(elementType);
  Value operand = scalarOperands[0];
  Type inputElemType = getElementType(operand.getType());
  Value result;

  MultiDialectBuilder<MathBuilder> create(rewriter, loc);
  Value negInf = create.math.negativeInf(inputElemType);
  Value posInf = create.math.positiveInf(inputElemType);

  double detectNegAttribute = dyn_cast<ONNXIsInfOp>(op).getDetectNegative();
  double detectPosAttribute = dyn_cast<ONNXIsInfOp>(op).getDetectPositive();

  // Three different cases: Infinity, Negative Infinity and Positive Infinity
  bool detectInf = detectPosAttribute == 1 && detectNegAttribute == 1;
  bool detectNeg = detectNegAttribute == 1 && detectPosAttribute == 0;
  bool detectPos = detectPosAttribute == 1 && detectNegAttribute == 0;

  Value equPos = create.math.eq(operand, posInf);
  Value equNeg = create.math.eq(operand, negInf);

  if (detectInf) {
    // If infinity return true for both positive and negative infinity
    result = create.math.ori(equPos, equNeg);
  }
  if (detectPos) {
    // If positive infinity return true else false
    result = equPos;
  }
  if (detectNeg) {
    // If negative infinity return true else false
    result = equNeg;
  }
  return result;
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
double analyzeSimdFor<ONNXSinhOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXCoshOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXSigmoidOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXHardSigmoidOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXEluOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXReluOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXLeakyReluOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXPReluOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXSeluOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXReciprocalOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXSoftplusOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXSoftsignOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXSignOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXErfOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXMaxOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXMinOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXNegOp>(Type t, int64_t &von, int64_t &son) {
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
// Scalar unary ops for lowering ONNXLessOp
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
// Scalar unary ops for lowering ONNXLessOrEqualOp
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
// Scalar unary ops for lowering ONNXGreaterOp
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
// Scalar unary ops for lowering ONNXGreaterOrEqualOp
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
// Scalar unary ops for lowering ONNXEqualOp
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
};

template <>
double analyzeSimdFor<ONNXModOp>(Type t, int64_t &von, int64_t &son) {
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
};

template <>
double analyzeSimdFor<ONNXMeanOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXRoundOp>(Type t, int64_t &von, int64_t &son) {
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
double analyzeSimdFor<ONNXClipOp>(Type t, int64_t &von, int64_t &son) {
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
    Type t, int64_t &von, int64_t &son) {
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

// Return SIMD unroll; no simd -> return 0;
// collapsedLiteralSize is ignored when we can collapse every loop iterations as
// we then rely on padding of the allocated memory to enable arbitrary output
// array simdization. When partial simd is requested, then we must ensure that
// the collapsed loop cumulative static size is a multiple of the VL.
template <typename ShapeHelperType, typename ElementwiseOp>
int64_t canBeVectorized(ShapeHelperType &shapeHelper, MDBuilder &create,
    MemRefType memRefType, int64_t collapsedInnermostLoops,
    int64_t collapsedLiteralSize) {
  int64_t simdUnroll = 0;
  // SIMD is enabled for this operation, test if profitable.
  Type elementType = memRefType.getElementType();
  int64_t vectorizedOpNum, scalarOpNum;
  double avgSimdWidth =
      analyzeSimdFor<ElementwiseOp>(elementType, vectorizedOpNum, scalarOpNum);
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
  // Test if there is enough work.
  int64_t staticSize;
  IndexExpr dynSize;
  bool isStaticSize = create.mem.getStaticAndDynamicMemSize(
      memRefType, shapeHelper.getOutputDims(), staticSize, dynSize);
  if (isStaticSize && staticSize < simdUnroll) {
    LLVM_DEBUG(llvm::dbgs() << "  simd disabled: trip count " << staticSize
                            << " too short \n");
    return 0;
  }
  if (collapsedInnermostLoops > 0 &&
      collapsedInnermostLoops < (int64_t)memRefType.getRank()) {
    // We have a partially flattened operator. Since we do only simdize entire
    // loops (i.e. we don't support scalar epilogues at this time), make sure
    // the static size is a multiple of the VL. Get the VL of the store
    // (output's element type).
    int64_t VL = vms->getVectorLength(elementType);
    if (collapsedLiteralSize % VL != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  simd disabled: partial flattened dims "
                 << collapsedInnermostLoops << " with size "
                 << collapsedLiteralSize << " is not 0 mod VL " << VL << "\n");
      return 0;
    }
    // See if we can get a unroll factor.
    bool gotOne = false;
    for (int64_t u = simdUnroll; u > 0; --u) {
      if (collapsedLiteralSize % (u * VL) == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  partial flattened dims " << collapsedInnermostLoops
                   << " with size " << collapsedLiteralSize << " works with VL "
                   << VL << " and unroll " << u << "\n");
        simdUnroll = u;
        gotOne = true;
        break;
      }
    }
    // Since we passed the test collapsedLiteralSize % VL == 0 above, this
    // assert is expected to hold true.
    assert(gotOne && "expected at least 1 *VL to work");
  }
  LLVM_DEBUG(llvm::dbgs() << "  SIMD with avg width " << avgSimdWidth
                          << " and unroll " << simdUnroll << "\n");
  return simdUnroll;
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be partially or fully flattened.
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
static LogicalResult getPartiallyFlattenedSimdCode(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXBroadcastOpShapeHelper *shapeHelper, Operation *op,
    MemRefType outputMemRefType, ValueRange operands, int64_t alignment,
    int64_t simdUnroll, int64_t collapsedInnermostLoops, bool ruledOutBroadcast,
    bool isUnaryOp) {
  Type outputElementType = outputMemRefType.getElementType();
  unsigned numArgs = op->getNumOperands();
  LLVM_DEBUG(llvm::dbgs() << "  partial SIMD code for elementwise op "
                          << op->getName() << " flattening "
                          << collapsedInnermostLoops << " inner dims\n");

  // generate SIMD code of VL elements per vector.
  IndexExprScope allocScope(create.vec, shapeHelper->getScope());
  int64_t VL =
      create.vec.getMachineVectorLength(outputElementType) * simdUnroll;
  // Alloc memory with padding for SIMD.
  // For the moment, its ok to go here; if we truly have partial flattening of
  // the simd code, then we only do it with static memref size that are
  // multiples of VL * simdUnroll, so there should be no padding anyway. This
  // will change if we do partial flattening with non-multiple of VL *
  // simdUnroll.
  Value alloc = create.mem.alignedAllocWithSimdPadding(
      outputMemRefType, shapeHelper->getOutputDims(), simdUnroll, alignment);
  // Create flat inputs in the last innerDinNum dims.
  llvm::SmallVector<Value, 4> flatOperands;
  for (Value oper : operands) {
    if (isNoneValue(oper) || hasOneElement(oper)) {
      // If its a none / scalar, it is not meant to be flattened.
      flatOperands.emplace_back(oper);
      continue;
    }
    llvm::SmallVector<IndexExpr, 4> operDims;
    Value operSize;
    create.krnlIE.getShapeAsSymbols(oper, operDims);
    Value flatOper = create.mem.reshapeToFlat(
        oper, operDims, operSize, collapsedInnermostLoops);
    flatOperands.emplace_back(flatOper);
  }
  // Create flat output.
  Value flattenedOutputSize;
  DimsExpr outputDims = shapeHelper->getOutputDims();
  Value flatAlloc = create.mem.reshapeToFlat(
      alloc, outputDims, flattenedOutputSize, collapsedInnermostLoops);
  // Create loop iteration (flattened to output dim - inner dim + 1) with inner
  // one and blocked by mVL.
  int64_t rank = outputDims.size() - collapsedInnermostLoops + 1;
  LLVM_DEBUG(
      llvm::dbgs() << "SIMD partial flatten with loop rank " << rank << "\n");
  int64_t flattenedDim = rank - 1;
  ValueRange loopDef = create.krnl.defineLoops(rank);
  ValueRange blockedLoopDef = create.krnl.block(loopDef[flattenedDim], VL);
  SmallVector<Value, 4> optimizedLoopDef;
  SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
  SmallVector<IndexExpr, 4> ubs;
  for (int64_t r = 0; r < rank - 1; ++r) {
    optimizedLoopDef.emplace_back(loopDef[r]);
    ubs.emplace_back(SymbolIndexExpr(outputDims[r]));
  }
  optimizedLoopDef.emplace_back(blockedLoopDef[0]);
  ubs.emplace_back(SymbolIndexExpr(flattenedOutputSize));
  // Create the vector type to operate over.
  VectorType vecElementType = VectorType::get({VL}, outputElementType);
  // Iterate only over the blocks.
  create.krnl.iterateIE(loopDef, optimizedLoopDef, lbs, ubs,
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
          MemRefType memRefType = flatOper.getType().dyn_cast<MemRefType>();
          assert(memRefType && "expected memref");
          VectorType vecType =
              VectorType::get({VL}, memRefType.getElementType());
          if (hasOneElementInInnermostDims(flatOper, 1)) {
            // If its a scalar, do a scalar load and splat.
            llvm::SmallVector<IndexExpr, 4> scalarAccessFct;
            if (hasOneElement(flatOper)) {
              // Not flattened, with only 1 dims, just put zeros as needed.
              int64_t scalarRank =
                  flatOper.getType().dyn_cast<ShapedType>().getRank();
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
  OpFusionHelper(
      mlir::ConversionPatternRewriter &rewriter, mlir::Operation *rootOp)
      : rootOp(rootOp), rewriter(rewriter), fusibleOps(), fuseEmitFuctions() {}

  // Fusion should not break any control dependence
  static bool isControlFlowValidForFusion(Operation *useOp, Operation *defOp);

  // Check whether the inputs of the useOp are valid for useOp to be fused
  // with the defOp. The defOp defines one of useOp's inputs.
  static bool areInputsValidForFusion(Operation *useOp, Operation *defOp);

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

  Value emitFuseOps(Value producerResult, ValueRange loopInd = {});

  void replaceOrEraseONNXOps(Value alloc);

private:
  mlir::Operation *rootOp;
  mlir::ConversionPatternRewriter &rewriter;
  llvm::SmallVector<mlir::Operation *, 2> fusibleOps;
  llvm::SmallVector<EmitScalarFunc, 2> fuseEmitFuctions;
}; // End of OpFusionHelper Declaration

// Check a node with type T is fusible or not.
// If true, record the op to data structure
template <typename T>
bool enqueueFusibleOpImpl(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions) {
  if (isa<T>(useOp)) {
    if (OpFusionHelper::isControlFlowValidForFusion(useOp, defOp) &&
        OpFusionHelper::areInputsValidForFusion(useOp, defOp)) {
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
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions);

template <typename T, class... Ts>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions) {
  if (enqueueFusibleOpImpl<T>(useOp, defOp, fusibleOps, fuseEmitFunctions)) {
    return true;
  } else {
    return enqueueFusibleOp<Ts...>(useOp, defOp, fusibleOps, fuseEmitFunctions);
  }
}

template <>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions) {
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
      mlir::ONNXFloorOp, mlir::ONNXHardSigmoidOp, mlir::ONNXIsInfOp,
      mlir::ONNXIsNaNOp, mlir::ONNXLeakyReluOp, mlir::ONNXLogOp,
      mlir::ONNXNegOp, mlir::ONNXNotOp, mlir::ONNXReciprocalOp,
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
      useOp, defOp, fusibleOps, fuseEmitFunctions);
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
    Operation *useOp, Operation *defOp) {
  // Elementwise unary operation is always fusible
  if (useOp->getOperands().size() == 1)
    return true;

  // To fuse Elementwise op with more one operands with the producer,
  // the shape of the output the user Op has to have the same size
  // output as that of the producer Op. Here dimension expansion with size
  // 1 is allowed. Refer to hasNoBroadcast() definition.
  // ToFix: This PR simply check static shape and does not use symbolic
  // shape analysis and BroadcastShapeHelper
  // Some discussion can be found at
  // https://github.com/onnx/onnx-mlir/issues/2199

  if (!hasStaticShape(defOp->getResults()[0].getType()))
    return false;

  ArrayRef<int64_t> defShape = getShape(defOp->getResults()[0].getType());
  ArrayRef<int64_t> useShape = getShape(useOp->getResults()[0].getType());
  if (defShape != useShape) {
    return false;
  }

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

    // ToFix: This restriction can be relaxed if ShapeHelper utility is used
    // to generate load in future.
    if (!hasStaticShape(useOp->getOperand(i).getType()))
      return false;
    ArrayRef<int64_t> inputShape = getShape(useOp->getOperand(i).getType());
    if (inputShape != defShape)
      return false;
  }

  return true;
}

// The seach for fusible ops starts from the rootOp, an elementwise operation.
// A successor op (user) is fusible if it is the only user, it is in the
// fusible elementwise op list, and its inputs are valid for fusion.
void OpFusionHelper::findFusibleOps() {
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
Value OpFusionHelper::emitFuseOps(Value defOpResult, ValueRange loopInd) {
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

    // Prepare Values for EmitScalarOpFor<T>
    SmallVector<Value, 2> inputValues;
    // ToFix: expect to use new utility for this purpose
    // There is an issue to fix: cannot getRemappedValue for the Value that is
    // currently handling: the defOp.
    // Otherwise, runtime error: "null operand found" caused by
    // just calling the function without using the result!
#if 0
    SmallVector<Value, 4> useOperands;
    for (auto oper : useOp->getOperands()) {
      if (oper.getDefiningOp() != defOp)
        useOperands.emplace_back(rewriter.getRemappedValue(oper));
    }
    LogicalResult res =
        rewriter.getRemappedValues(useOp->getOperands(), useOperands);
    assert(succeeded(res) && "Could not remap value for rewriter");
    ONNXBroadcastOpShapeHelper shapeHelper(
        useOp, useOperands, &create.krnlIE, nullptr, false);
#endif
    for (size_t i = 0; i < useOp->getOperands().size(); i++) {
      Value inputValue = useOp->getOperand(i);
      Operation *inputOp = inputValue.getDefiningOp();
      if (inputOp == defOp) {
        inputValues.emplace_back(defOpResult);
      } else {
        // ToFix: expect to use new utility to handle any broadcast cases
#if 0
        IndexExprScope innerScope(create.krnl, shapeHelper.getScope());
        SmallVector<IndexExpr, 4> outputAccessExprs;
        getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);
        SmallVector<IndexExpr, 4> loadAccessExprs;
        LogicalResult res = shapeHelper.getAccessExprs(
            inputValue, i, outputAccessExprs, loadAccessExprs, true);
        assert(succeeded(res) && "Could not compute access indices");
        Value load = create.krnl.loadIE(useOperands[i], loadAccessExprs);
#endif
        // The shape is guaranteed to be the same.
        Value load =
            create.krnl.load(rewriter.getRemappedValue(inputValue), loopInd);
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

  ONNXElementwiseUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD)
      : OpConversionPattern<ElementwiseUnaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {}

  LogicalResult matchAndRewrite(ElementwiseUnaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    Location loc = ONNXLoc<ElementwiseUnaryOp>(op);
    Value X = operands[0];

    // If type is scalar or vector, there is no need to allocate a buffer.
    // Just call scalar computation and return the result. This is efficient
    // when elementwise ops are used as activations for ops like LSTM/GRU/RNN.
    if (!X.getType().isa<TensorType>() && !X.getType().isa<MemRefType>()) {
      SmallVector<Value> args;
      args.emplace_back(X);
      // Load the remaining (scalar) values.
      for (uint64_t i = 1; i < operands.size(); i++) {
        if (isNoneValue(operands[i])) {
          args.emplace_back(operands[i]);
          continue;
        }
        assert(!operands[i].getType().isa<TensorType>() &&
               !operands[i].getType().isa<MemRefType>() &&
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
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
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
      int64_t simdUnroll =
          canBeVectorized<ONNXUnaryOpShapeHelper, ElementwiseUnaryOp>(
              shapeHelper, create, outputMemRefType, outputRank,
              /*collapsedInnermostLoops, ignored*/ 1);
      if (simdUnroll > 0)
        return getPartiallyFlattenedSimdCode<ElementwiseUnaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            simdUnroll, /*collapsedInnermostLoop*/ outputRank,
            /*ruleOutBroadcast*/ true, /*unary*/ true);
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op);
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
                opFusionHelper.emitFuseOps(loweredOpResult, loopInd);
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
      loweredOpResult = opFusionHelper.emitFuseOps(loweredOpResult);
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

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD,
      bool isUniBroadcasting = false)
      : OpConversionPattern<ElementwiseBinaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD),
        isUniBroadcasting(isUniBroadcasting) {}

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
      int64_t simdUnroll =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseBinaryOp>(
              shapeHelper, create, outputMemRefType, collapsedInnermostLoops,
              collapsedLiteralSize);
      if (simdUnroll > 0)
        return getPartiallyFlattenedSimdCode<ElementwiseBinaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            simdUnroll, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false);
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op);
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

            result = opFusionHelper.emitFuseOps(result, loopInd);
            // Store result in the resulting array.
            createKrnl.store(result, alloc, loopInd);
          });
    } else {
      Value lhs = create.krnl.load(operands[0]);
      Value rhs = create.krnl.load(operands[1]);

      // Apply the element-wise function.
      Value result = emitScalarOpFor<ElementwiseBinaryOp>(
          rewriter, loc, op, outputElementType, {lhs, rhs});

      result = opFusionHelper.emitFuseOps(result);
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

  ONNXElementwiseVariadicOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD)
      : OpConversionPattern<ElementwiseVariadicOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {}

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
      int64_t simdUnroll =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseVariadicOp>(
              shapeHelper, create, outputMemRefType, collapsedInnermostLoops,
              collapsedLiteralSize);
      if (simdUnroll > 0)
        return getPartiallyFlattenedSimdCode<ElementwiseVariadicOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            simdUnroll, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false);
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op);
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
            finalResult = opFusionHelper.emitFuseOps(finalResult, loopInd);
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
      finalResult = opFusionHelper.emitFuseOps(finalResult);
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

  ONNXWhereOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD)
      : ConversionPattern(
            typeConverter, ONNXWhereOp::getOperationName(), 1, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {}

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
    bool enableSIMD) {
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
      typeConverter, ctx, dimAnalysis, enableSIMD);
  patterns.insert<ONNXElementwiseBinaryOpLowering<mlir::ONNXPReluOp>>(
      typeConverter, ctx, dimAnalysis, enableSIMD, /*isUniBroadcasting=*/true);
}

} // namespace onnx_mlir
