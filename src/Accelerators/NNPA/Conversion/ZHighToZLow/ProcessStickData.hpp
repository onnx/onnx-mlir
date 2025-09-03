/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickData.cpp - Process Stick data ----------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to Krnl/Affine/SCF
// operations that operates on stickified input/output data.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PROCESS_STICK_DATA_H
#define ONNX_MLIR_PROCESS_STICK_DATA_H

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

namespace onnx_mlir {

// By definition of the conversion from dlf16 to f32, vecOfF32Vals should always
// contain pairs of vectors.
using ContiguousVectorOfF32IterateBodyFn = std::function<void(
    const KrnlBuilder &b, mlir::SmallVectorImpl<mlir::Value> &vecOfF32Vals,
    DimsExpr &loopIndices)>;

using ScalarF32IterateBodyFn = std::function<void(
    const KrnlBuilder &b, mlir::Value scalarF32Val, DimsExpr &loopIndices)>;

// Iterate over each values in the input's sticks, processing vectors (of 4 F32)
// with processVectorOfF32Vals, and scalars (1 F32) with processScalarF32Val, By
// definition, processVectorOfF32Vals contains either 2 or 2*unrollVL vectors.
// And processScalarF32Val process only 1 scalar value. Output is only used for
// prefetching. If output is null, skip output prefetching. In general, we
// expects lbs={0...0} and ubs=dims. WHen parallelized outside of this loop,
// then lbs and ubs can reflect the subset of iterations assigned to this
// thread. Iterations cannot be partitioned on the innermost dim.
template <class BUILDER>
void IterateOverStickInputData(const BUILDER &b, mlir::Operation *op,
    DimsExpr &lbs, DimsExpr &ubs, DimsExpr &dims, mlir::StringAttr layout,
    mlir::Value input, mlir::Value output, int64_t unrollVL,
    bool enableParallel, bool enablePrefetch,
    ContiguousVectorOfF32IterateBodyFn processVectorOfF32Vals,
    ScalarF32IterateBodyFn processScalarF32Val);

// Compute min/max from stickified input. Currently support 2DS, 3D, 3DS,
// 4D formats.
void emitDynamicQuantizationLinearMinMaxFromStickifiedInput(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *op, mlir::Value input, mlir::StringAttr inputLayout,
    mlir::Value &inputMin, mlir::Value &inputMax, bool enableSIMD,
    bool enableParallel);

} // namespace onnx_mlir

// Include template code.
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp.inc"

#endif
