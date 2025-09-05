/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.hpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to mlir::Krnl dialect.
//
// There are 3 steps here:
// 1) Add the type of the operations that use the standard code gen pattern
// below.
//
// 2) Add the "struct ScalarOp<mlir::ONNX OP>" for each op so that the
// code gen pattern knows how to process each ONNX elementwise operation.
// Operations that use CustomScalarOp must also define a emitScalarOpFor<> which
// can be done using the DECL_EMIT_SCALAR_OP_FOR macro.
//
// 3) Add custom code generation (and simd cost) in the Elementwise.cpp file.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

// Usage: to reuse the list of all op type that are handled by Elementwise code,
// define either the generic ELEMENTWISE_ALL(OP_TYPE) macro (targeting all ops),
// or the 3 specific macros (ELEMENTWISE_BINARY, _UNARY, or _VARIADIC) to
// indicate how to process the type list. For example of usages, grep for this
// pattern ` #include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp" `.

// Define ELEMENTWISE_ALL(OP_TYPE) to apply the same patterns to all elementwise
// types regardless of unary, binary, or variadic. Apply only if specific macro
// is not already defined.

#ifdef ELEMENTWISE_ALL
#ifndef ELEMENTWISE_BINARY
#define ELEMENTWISE_BINARY(_OP_TYPE) ELEMENTWISE_ALL(_OP_TYPE)
#endif
#ifndef ELEMENTWISE_UNARY
#define ELEMENTWISE_UNARY(_OP_TYPE) ELEMENTWISE_ALL(_OP_TYPE)
#endif
#ifndef ELEMENTWISE_VARIADIC
#define ELEMENTWISE_VARIADIC(_OP_TYPE) ELEMENTWISE_ALL(_OP_TYPE)
#endif
#endif

//===----------------------------------------------------------------------===//
// STEP 1: Add new op in the type list below, in the right category: Unary,
// binary, or variadic. This list is used when we need to "know" all the
// elementwise ops for which we have standard code generation scheme.
//===----------------------------------------------------------------------===//

#ifdef ELEMENTWISE_BINARY
// Binary elementwise, alphabetical
ELEMENTWISE_BINARY(mlir::ONNXBitShiftOp)
ELEMENTWISE_BINARY(mlir::ONNXBitwiseAndOp)
ELEMENTWISE_BINARY(mlir::ONNXBitwiseOrOp)
ELEMENTWISE_BINARY(mlir::ONNXBitwiseXorOp)
ELEMENTWISE_BINARY(mlir::ONNXEqualOp)
ELEMENTWISE_BINARY(mlir::ONNXGreaterOp)
ELEMENTWISE_BINARY(mlir::ONNXGreaterOrEqualOp)
ELEMENTWISE_BINARY(mlir::ONNXLessOp)
ELEMENTWISE_BINARY(mlir::ONNXLessOrEqualOp)
ELEMENTWISE_BINARY(mlir::ONNXModOp)
ELEMENTWISE_BINARY(mlir::ONNXPowOp)
ELEMENTWISE_BINARY(mlir::ONNXPReluOp)
#endif

#ifdef ELEMENTWISE_UNARY
// Unary elementwise, alphabetical
ELEMENTWISE_UNARY(mlir::ONNXAbsOp)
ELEMENTWISE_UNARY(mlir::ONNXAcosOp)
ELEMENTWISE_UNARY(mlir::ONNXAcoshOp)
ELEMENTWISE_UNARY(mlir::ONNXAsinOp)
ELEMENTWISE_UNARY(mlir::ONNXAsinhOp)
ELEMENTWISE_UNARY(mlir::ONNXAtanOp)
ELEMENTWISE_UNARY(mlir::ONNXAtanhOp)
ELEMENTWISE_UNARY(mlir::ONNXBinarizerOp)
ELEMENTWISE_UNARY(mlir::ONNXBitwiseNotOp)
ELEMENTWISE_UNARY(mlir::ONNXCastOp)
ELEMENTWISE_UNARY(mlir::ONNXCeilOp)
ELEMENTWISE_UNARY(mlir::ONNXCeluOp)
ELEMENTWISE_UNARY(mlir::ONNXClipOp)
ELEMENTWISE_UNARY(mlir::ONNXCosOp)
ELEMENTWISE_UNARY(mlir::ONNXCoshOp)
ELEMENTWISE_UNARY(mlir::ONNXDequantizeLinearOp)
ELEMENTWISE_UNARY(mlir::ONNXEluOp)
ELEMENTWISE_UNARY(mlir::ONNXErfOp)
ELEMENTWISE_UNARY(mlir::ONNXExpOp)
ELEMENTWISE_UNARY(mlir::ONNXFloorOp)
ELEMENTWISE_UNARY(mlir::ONNXGeluOp)
ELEMENTWISE_UNARY(mlir::ONNXHardSigmoidOp)
ELEMENTWISE_UNARY(mlir::ONNXHardSwishOp)
ELEMENTWISE_UNARY(mlir::ONNXIsInfOp)
ELEMENTWISE_UNARY(mlir::ONNXIsNaNOp)
ELEMENTWISE_UNARY(mlir::ONNXLeakyReluOp)
ELEMENTWISE_UNARY(mlir::ONNXLogOp)
ELEMENTWISE_UNARY(mlir::ONNXMishOp)
ELEMENTWISE_UNARY(mlir::ONNXNegOp)
ELEMENTWISE_UNARY(mlir::ONNXNotOp)
ELEMENTWISE_UNARY(mlir::ONNXReciprocalOp)
ELEMENTWISE_UNARY(mlir::ONNXReluOp)
ELEMENTWISE_UNARY(mlir::ONNXRoundOp)
ELEMENTWISE_UNARY(mlir::ONNXSeluOp)
ELEMENTWISE_UNARY(mlir::ONNXShrinkOp)
ELEMENTWISE_UNARY(mlir::ONNXSigmoidOp)
ELEMENTWISE_UNARY(mlir::ONNXSignOp)
ELEMENTWISE_UNARY(mlir::ONNXSinOp)
ELEMENTWISE_UNARY(mlir::ONNXSinhOp)
ELEMENTWISE_UNARY(mlir::ONNXSoftplusOp)
ELEMENTWISE_UNARY(mlir::ONNXSoftsignOp)
ELEMENTWISE_UNARY(mlir::ONNXSqrtOp)
ELEMENTWISE_UNARY(mlir::ONNXTanOp)
ELEMENTWISE_UNARY(mlir::ONNXTanhOp)
ELEMENTWISE_UNARY(mlir::ONNXThresholdedReluOp)
#endif

#ifdef ELEMENTWISE_VARIADIC
// Variadic elementwise, alphabetical
ELEMENTWISE_VARIADIC(mlir::ONNXAddOp)
ELEMENTWISE_VARIADIC(mlir::ONNXAndOp)
ELEMENTWISE_VARIADIC(mlir::ONNXDivOp)
ELEMENTWISE_VARIADIC(mlir::ONNXMaxOp)
ELEMENTWISE_VARIADIC(mlir::ONNXMeanOp)
ELEMENTWISE_VARIADIC(mlir::ONNXMinOp)
ELEMENTWISE_VARIADIC(mlir::ONNXMulOp)
ELEMENTWISE_VARIADIC(mlir::ONNXOrOp)
ELEMENTWISE_VARIADIC(mlir::ONNXSubOp)
ELEMENTWISE_VARIADIC(mlir::ONNXSumOp)
ELEMENTWISE_VARIADIC(mlir::ONNXXorOp)
#endif

//===----------------------------------------------------------------------===//
// STEP 2: Add the declaration for each op, in alphabetical order.
//===----------------------------------------------------------------------===//

#if !defined(ONNX_MLIR_ELEMENTWISE_H) && !defined(ELEMENTWISE_BINARY) &&       \
    !defined(ELEMENTWISE_UNARY) && !defined(ELEMENTWISE_VARIADIC)
#define ONNX_MLIR_ELEMENTWISE_H

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

namespace onnx_mlir {

// =============================================================================
// Scalar ops handling
// add a DECL_EMIT_SCALAR_OP_FOR(_XXX) for any type that uses CustomScalarOp.

#define DECL_EMIT_SCALAR_OP_FOR(_OP_TYPE)                                      \
  template <>                                                                  \
  mlir::Value emitScalarOpFor<_OP_TYPE>(                                       \
      mlir::ConversionPatternRewriter & rewriter, mlir::Location loc,          \
      mlir::Operation * op, mlir::Type elementType,                            \
      mlir::ArrayRef<mlir::Value> scalarOperands);

// A
template <>
struct ScalarOp<mlir::ONNXAddOp> {
  using FOp = mlir::arith::AddFOp;
  using IOp = mlir::arith::AddIOp;
};

template <>
struct ScalarOp<mlir::ONNXAbsOp> {
  using FOp = mlir::math::AbsFOp;
  using IOp = mlir::math::AbsIOp;
};

template <>
struct ScalarOp<mlir::ONNXAcosOp> {
  using FOp = mlir::KrnlAcosOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXAcoshOp> {
  using FOp = mlir::KrnlAcoshOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXAndOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::AndIOp;
};

template <>
struct ScalarOp<mlir::ONNXAsinOp> {
  using FOp = mlir::KrnlAsinOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXAsinhOp> {
  using FOp = mlir::KrnlAsinhOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXAtanOp> {
  using FOp = mlir::KrnlAtanOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXAtanhOp> {
  using FOp = mlir::KrnlAtanhOp;
  using IOp = NotSuportedScalarOp;
};

// B
template <>
struct ScalarOp<mlir::ONNXBinarizerOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXBinarizerOp)

template <>
struct ScalarOp<mlir::ONNXBitShiftOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXBitShiftOp)

template <>
struct ScalarOp<mlir::ONNXBitwiseAndOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::AndIOp;
};

template <>
struct ScalarOp<mlir::ONNXBitwiseNotOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXBitwiseNotOp)

template <>
struct ScalarOp<mlir::ONNXBitwiseOrOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::OrIOp;
};

template <>
struct ScalarOp<mlir::ONNXBitwiseXorOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::XOrIOp;
};

// C
template <>
struct ScalarOp<mlir::ONNXCastOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXCastOp)

template <>
struct ScalarOp<mlir::ONNXCeilOp> {
  using FOp = mlir::math::CeilOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXCeluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXCeluOp)

template <>
struct ScalarOp<mlir::ONNXClipOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXClipOp)

template <>
struct ScalarOp<mlir::ONNXCosOp> {
  using FOp = mlir::math::CosOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXCoshOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXCoshOp)

// D
template <>
struct ScalarOp<mlir::ONNXDequantizeLinearOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXDequantizeLinearOp)

template <>
struct ScalarOp<mlir::ONNXDivOp> {
  using FOp = mlir::arith::DivFOp;
  using IOp = mlir::arith::DivSIOp;
};

// E
template <>
struct ScalarOp<mlir::ONNXEluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXEluOp)

template <>
struct ScalarOp<mlir::ONNXEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXEqualOp)

template <>
struct ScalarOp<mlir::ONNXErfOp> {
  using FOp = mlir::math::ErfOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXExpOp> {
  using FOp = mlir::math::ExpOp;
  using IOp = NotSuportedScalarOp;
};

// F
template <>
struct ScalarOp<mlir::ONNXFloorOp> {
  using FOp = mlir::math::FloorOp;
  using IOp = NotSuportedScalarOp;
};

// G
template <>
struct ScalarOp<mlir::ONNXGeluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXGeluOp)

template <>
struct ScalarOp<mlir::ONNXGreaterOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXGreaterOp)

template <>
struct ScalarOp<mlir::ONNXGreaterOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXGreaterOrEqualOp)

// H
template <>
struct ScalarOp<mlir::ONNXHardSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXHardSigmoidOp)

template <>
struct ScalarOp<mlir::ONNXHardSwishOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXHardSwishOp)

// I
template <>
struct ScalarOp<mlir::ONNXIsInfOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXIsInfOp)

template <>
struct ScalarOp<mlir::ONNXIsNaNOp> {
  using FOp = mlir::KrnlIsNaNOp;
  using IOp = NotSuportedScalarOp;
};

// L
template <>
struct ScalarOp<mlir::ONNXLeakyReluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXLeakyReluOp)

template <>
struct ScalarOp<mlir::ONNXLessOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXLessOp)

template <>
struct ScalarOp<mlir::ONNXLessOrEqualOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXLessOrEqualOp)

template <>
struct ScalarOp<mlir::ONNXLogOp> {
  using FOp = mlir::math::LogOp;
  using IOp = NotSuportedScalarOp;
};

// M
template <>
struct ScalarOp<mlir::ONNXMaxOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXMaxOp)

template <>
struct ScalarOp<mlir::ONNXMeanOp> {
  using FOp = mlir::arith::AddFOp;
  using IOp = mlir::arith::AddIOp;
};

template <>
struct ScalarOp<mlir::ONNXMinOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXMinOp)

template <>
struct ScalarOp<mlir::ONNXMishOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXMishOp)

template <>
struct ScalarOp<mlir::ONNXModOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXModOp)

template <>
struct ScalarOp<mlir::ONNXMulOp> {
  using FOp = mlir::arith::MulFOp;
  using IOp = mlir::arith::MulIOp;
};

// N
template <>
struct ScalarOp<mlir::ONNXNegOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXNegOp)

template <>
struct ScalarOp<mlir::ONNXNotOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXNotOp)

// O
template <>
struct ScalarOp<mlir::ONNXOrOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::OrIOp;
};

// P
template <>
struct ScalarOp<mlir::ONNXPowOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXPowOp)

template <>
struct ScalarOp<mlir::ONNXPReluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXPReluOp)

// R
template <>
struct ScalarOp<mlir::ONNXReciprocalOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXReciprocalOp)

template <>
struct ScalarOp<mlir::ONNXReluOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXReluOp)

template <>
struct ScalarOp<mlir::ONNXRoundOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXRoundOp)

// S
template <>
struct ScalarOp<mlir::ONNXSeluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSeluOp)

template <>
struct ScalarOp<mlir::ONNXShrinkOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXShrinkOp)

template <>
struct ScalarOp<mlir::ONNXSqrtOp> {
  using FOp = mlir::math::SqrtOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXSignOp> {
  using FOp = CustomScalarOp;
  using IOp = CustomScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSignOp)

template <>
struct ScalarOp<mlir::ONNXSigmoidOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSigmoidOp)

template <>
struct ScalarOp<mlir::ONNXSinOp> {
  using FOp = mlir::math::SinOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXSinhOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSinhOp)

template <>
struct ScalarOp<mlir::ONNXSoftplusOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSoftplusOp)

template <>
struct ScalarOp<mlir::ONNXSoftsignOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXSoftsignOp)

template <>
struct ScalarOp<mlir::ONNXSubOp> {
  using FOp = mlir::arith::SubFOp;
  using IOp = mlir::arith::SubIOp;
};

template <>
struct ScalarOp<mlir::ONNXSumOp> {
  using FOp = mlir::arith::AddFOp;
  using IOp = mlir::arith::AddIOp;
};

// T
template <>
struct ScalarOp<mlir::ONNXTanOp> {
  using FOp = mlir::KrnlTanOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXTanhOp> {
  using FOp = mlir::math::TanhOp;
  using IOp = NotSuportedScalarOp;
};

template <>
struct ScalarOp<mlir::ONNXThresholdedReluOp> {
  using FOp = CustomScalarOp;
  using IOp = NotSuportedScalarOp;
};
DECL_EMIT_SCALAR_OP_FOR(mlir::ONNXThresholdedReluOp)

// X
template <>
struct ScalarOp<mlir::ONNXXorOp> {
  using FOp = NotSuportedScalarOp;
  using IOp = mlir::arith::XOrIOp;
};

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// STEP 3: Add code gen for CustomScalarOps in Elementwise.cpp
//===----------------------------------------------------------------------===//

#endif

// Undefine all the marcos for the types.
#undef ELEMENTWISE_BINARY
#undef ELEMENTWISE_UNARY
#undef ELEMENTWISE_VARIADIC
#undef ELEMENTWISE_ALL
