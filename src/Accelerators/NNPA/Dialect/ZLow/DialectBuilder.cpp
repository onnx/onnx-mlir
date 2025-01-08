/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// ZLow Builder for building ZLow operations
// =============================================================================

void ZLowBuilder::stick(
    Value x, Value out, StringAttr layout, IntegerAttr saturation) const {
  b().create<zlow::ZLowStickOp>(loc(), x, out, layout, saturation);
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
