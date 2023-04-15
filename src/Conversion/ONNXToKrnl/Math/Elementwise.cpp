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
struct ScalarOp<ONNXErfOp> {
  using FOp = KrnlErfOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<ONNXIsInfOp> {
  using FOp = KrnlIsInfOp;
  using IOp = NotSuportedScalarOp;
};

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
template <typename ShapeHelperType, typename ElementwiseOp>
int64_t canBeVectorized(
    ShapeHelperType &shapeHelper, MDBuilder &create, MemRefType memRefType) {
  int64_t simdUnroll = 0;
  // SIMD is enabled for this operation, test if profitable.
  Type elementType = memRefType.getElementType();
  int64_t vectorizedOpNum, scalarOpNum;
  double avgSimdWidth =
      analyzeSimdFor<ElementwiseOp>(elementType, vectorizedOpNum, scalarOpNum);
  if (avgSimdWidth < 1.5) {
    if (DEBUG)
      llvm::errs() << "SIMD disabled: avg simd width  " << avgSimdWidth
                   << " too small\n";
    return 0;
  }
  // Determine empirical unroll factor.
  VectorMachineSupport *vms =
      VectorMachineSupport::getGlobalVectorMachineSupport();

  int64_t vrNum = vms->VectorRegisterNum();
  if (vectorizedOpNum >= vrNum / 2)
    simdUnroll = 1;
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
    if (DEBUG)
      llvm::errs() << "SIMD disabled: trip count " << staticSize
                   << " too short \n";
    return 0;
  }
  if (DEBUG)
    llvm::errs() << "SIMD with avg width " << avgSimdWidth << " and unroll "
                 << simdUnroll << "\n";
  return simdUnroll;
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be fully flattened.
//===----------------------------------------------------------------------===//

template <typename ElementwiseUnaryOp>
static LogicalResult getUnaryBinarySimdCodeFullyFlattened(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXOpShapeHelper *shapeHelper, Operation *op, MemRefType outputMemRefType,
    ValueRange operands, int64_t alignment, int64_t simdUnroll) {
  Type outputElementType = outputMemRefType.getElementType();

  if (DEBUG)
    llvm::errs() << "SIMD code for binary op " << op->getName() << "\n";

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
    if (isNoneValue(oper) || isScalarValue(oper)) {
      // If its a none / scalar, it is not meant to be flattened.
      flatOperands.emplace_back(oper);
      continue;
    }
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
          if (isNoneValue(flatOper)) {
            // None, just pass it on unmodified.
            loadedVals.emplace_back(flatOper);
            continue;
          }
          MemRefType memRefType = flatOper.getType().dyn_cast<MemRefType>();
          assert(memRefType && "expected memref");
          VectorType vecType =
              VectorType::get({VL}, memRefType.getElementType());
          if (isScalarValue(flatOper)) {
            // If its a scalar, do a scalar load and splat.
            Value loadedVal = create.krnl.load(flatOper);
            Value splatValue = create.vec.splat(vecType, loadedVal);
            loadedVals.emplace_back(splatValue);
          } else {
            Value loadedVal = create.vec.load(vecType, flatOper, loopInd);
            loadedVals.emplace_back(loadedVal);
          }
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
    llvm::errs() << "SIMD code for variadic op " << op->getName() << "\n";

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
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    bool isScalar = hasAllScalarValues(operands);
    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && !hasNonIdentityLayout(operands)) {
      int64_t simdUnroll =
          canBeVectorized<ONNXUnaryOpShapeHelper, ElementwiseUnaryOp>(
              shapeHelper, create, memRefType);
      if (simdUnroll > 0)
        return getUnaryBinarySimdCodeFullyFlattened<ElementwiseUnaryOp>(
            rewriter, create, &shapeHelper, op, memRefType, operands, alignment,
            simdUnroll);
    }

    // Insert an allocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(
        memRefType, shapeHelper.getOutputDims(), alignment);

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(memRefType.getRank());
      SmallVector<IndexExpr, 4> lbs(memRefType.getRank(), LiteralIndexExpr(0));
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
      // Store result in the resulting array.
      create.krnl.store(loweredOpResult, alloc);
    }

    rewriter.replaceOp(op, alloc);
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

    bool isScalar = hasAllScalarValues(operands);
    // Shape helper can determine if there is no static broadcast.
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);

    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && hasNoBroadcast &&
        !hasNonIdentityLayout(operands)) {
      int64_t simdUnroll =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseBinaryOp>(
              shapeHelper, create, outputMemRefType);
      if (simdUnroll > 0)
        return getUnaryBinarySimdCodeFullyFlattened<ElementwiseBinaryOp>(
            rewriter, create, &shapeHelper, op, outputMemRefType, operands,
            alignment, simdUnroll);
    }

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
            LogicalResult res = shapeHelper.getAccessExprs(operands[0], 0,
                outputAccessExprs, lhsAccessExprs, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs = createKrnl.loadIE(operands[0], lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operands[1], 1, outputAccessExprs,
                rhsAccessExprs, hasNoBroadcast);
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

    bool isScalar = hasAllScalarValues(operands);
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);
    if (enableSIMD && !isScalar && hasNoBroadcast &&
        !hasNonIdentityLayout(operands)) {
      // SIMD is enabled for this operation, test if desired and feasible
      int64_t simdUnroll =
          canBeVectorized<ONNXBroadcastOpShapeHelper, ElementwiseVariadicOp>(
              shapeHelper, create, outputMemRefType);
      if (simdUnroll > 0)
        return getVariadicSimdCodeFullyFlattened<ElementwiseVariadicOp>(
            rewriter, create, &shapeHelper, op, outputMemRefType, operands,
            alignment, simdUnroll);
    }

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
            LogicalResult res = shapeHelper.getAccessExprs(operands[0], 0,
                outputAccessExprs, oprdAccessExprs, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value accumulated = createKrnl.loadIE(operands[0], oprdAccessExprs);

            // Iterate over the remaining operands.
            for (unsigned i = 1; i < numArgs; ++i) {
              // Obtain the next operand.
              SmallVector<IndexExpr, 4> oprdAccessExprs;
              LogicalResult res = shapeHelper.getAccessExprs(operands[i], i,
                  outputAccessExprs, oprdAccessExprs, hasNoBroadcast);
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
            LogicalResult res =
                shapeHelper.getAccessExprs(operandAdaptor.getCondition(), 0,
                    outputAccessExprs, condAccessExprs, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value cond = createKrnl.loadIE(
                operandAdaptor.getCondition(), condAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getX(), 1,
                outputAccessExprs, lhsAccessExprs, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs =
                createKrnl.loadIE(operandAdaptor.getX(), lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getY(), 2,
                outputAccessExprs, rhsAccessExprs, hasNoBroadcast);
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
