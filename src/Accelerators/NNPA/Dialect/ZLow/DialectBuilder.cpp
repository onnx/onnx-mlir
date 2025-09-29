/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// ZLow Builder for building ZLow operations
// =============================================================================

void ZLowBuilder::stick(
    Value x, Value out, StringAttr layout, IntegerAttr noSaturation) const {
  b().create<zlow::ZLowStickOp>(loc(), x, out, layout, noSaturation);
}

void ZLowBuilder::convertDLF16ToF32(
    Value dlf16, Value &highF32, Value &lowF32) {
  assert(mlir::dyn_cast<VectorType>(dlf16.getType()) && "expect vector");
  auto op = b().create<zlow::ZLowConvertDLF16ToF32VectorOp>(loc(), dlf16);
  highF32 = op.getResult(0);
  lowF32 = op.getResult(1);
}

Value ZLowBuilder::convertDLF16ToF32(Value dlf16) {
  assert(!mlir::dyn_cast<VectorType>(dlf16.getType()) && "expect scalar");
  auto op = b().create<zlow::ZLowConvertDLF16ToF32Op>(loc(), dlf16);
  return op.getResult();
}

Value ZLowBuilder::convertF32ToDLF16(
    Value highF32, Value lowF32, bool disableSaturation) {
  assert(mlir::dyn_cast<VectorType>(highF32.getType()) && "expect vector");
  assert(mlir::dyn_cast<VectorType>(lowF32.getType()) && "expect vector");
  if (!disableSaturation) {
    // Saturation is requested
    MultiDialectBuilder<MathBuilder> create(*this);
    Type f32Type = b().getF32Type();
    Value minInF32 = create.math.constant(f32Type, DLF16_MIN);
    Value maxInF32 = create.math.constant(f32Type, DLF16_MAX);
    highF32 = create.math.min(highF32, maxInF32);
    lowF32 = create.math.min(lowF32, maxInF32);
    highF32 = create.math.max(highF32, minInF32);
    lowF32 = create.math.max(lowF32, minInF32);
  }
  return b().create<zlow::ZLowConvertF32ToDLF16VectorOp>(
      loc(), highF32, lowF32);
}

Value ZLowBuilder::convertF32ToDLF16(Value f32) {
  assert(!mlir::dyn_cast<VectorType>(f32.getType()) && "expect scalar");
  return b().create<zlow::ZLowConvertF32ToDLF16Op>(loc(), f32);
}

void ZLowBuilder::quantizedStick(Value x, Value recScale, Value offset,
    Value out, StringAttr layout, StringAttr qType) const {
  b().create<zlow::ZLowQuantizedStickOp>(
      loc(), x, recScale, offset, out, layout, qType);
}

void ZLowBuilder::quantizedMatMul(Value x, Value xRecScale, Value xOffset,
    Value y, Value yRecScale, Value yOffset, Value bias, Value biasRecScale,
    Value biasOffset, Value workArea, Value shape, Value out, Value outRecScale,
    Value outOffset, StringAttr xQType, StringAttr yQType, StringAttr biasQType,
    StringAttr outQType, IntegerAttr isBcast, IntegerAttr isStacked,
    IntegerAttr preComputedBias, IntegerAttr disableClipping,
    IntegerAttr dequantizeOutput) const {
  b().create<zlow::ZLowQuantizedMatMulOp>(loc(), x, xRecScale, xOffset, y,
      yRecScale, yOffset, bias, biasRecScale, biasOffset, workArea, shape, out,
      outRecScale, outOffset, xQType, yQType, biasQType, outQType, isBcast,
      isStacked, preComputedBias, disableClipping, dequantizeOutput);
}

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForZLow::getConst(Value value) { return nullptr; }

Value IndexExprBuilderForZLow::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  uint64_t rank = getShapedTypeRank(intArrayVal);
  if (rank == 0)
    return create.affine.load(intArrayVal);
  uint64_t size = getArraySize(intArrayVal);
  assert(i < size && "out of bound reference");
  Value iVal = create.math.constantIndex(i);
  return create.affine.load(intArrayVal, {iVal});
}

Value IndexExprBuilderForZLow::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  MemRefBuilder createMemRef(*this);
  return createMemRef.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
