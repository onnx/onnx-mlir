/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickData.cpp - Process Stick data ----------------===//
//
// Copyright 2024-25 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to Krnl/Affine/SCF
// operations that operates on stickified input/output data.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PROCESS_STICK_DATA_H
#define ONNX_MLIR_PROCESS_STICK_DATA_H

#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

namespace onnx_mlir {
// By definition of the conversion from dlf16 to f32, vecOfF32Vals should always
// contain pairs of vectors.
using ContiguousVectorOfF32IterateBodyFn = std::function<void(
    const KrnlBuilder &b, mlir::SmallVectorImpl<mlir::Value> &vecOfF32Vals,
    DimsExpr &loopIndices)>;

using ScalarF32IterateBodyFn = std::function<void(
    const KrnlBuilder &b, mlir::Value scalarF32Val, DimsExpr &loopIndices)>;

// TODO: eventually this method can be replaced by the one used by elementwise
// operations.

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

class StickComputeSupport {
public:
  using MultiValuesOfF32IterateBodyFn =
      std::function<mlir::Value(const KrnlBuilder &b,
          mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals)>;

  // Used to compute ahead of time the re-interpreted memref for stick data.
  // If not stick reference, return nullptr.
  static mlir::Value getMemRefForStick(
      KrnlBuilder &kb, mlir ::Value originalVal, mlir ::Value originalMemRef);

  StickComputeSupport(KrnlBuilder &kb,
      /* op inputs */ mlir::ValueRange originalInput,
      /* op inputs */ mlir::ValueRange originalInputMemRef,
      /* optional memref for stick */ mlir::ValueRange optionalMemRefForStick,
      /* op output */ mlir::Value originalOutput,
      /* op output */ mlir::Value originalOutputMemRef, 
      /* optional memref for stick */ mlir::Value optionalOutputMemRefForStick, 
      bool disableSaturation = false);
  void init(KrnlBuilder &kb,
      /* op inputs */ mlir::ValueRange originalInput,
      /* op inputs */ mlir::ValueRange originalInputMemRef,
      /* optional memref for stick */ mlir::ValueRange optionalMemRefForStick,
      /* op output */ mlir::Value originalOutput,
      /* op output */ mlir::Value originalOutputMemRef, 
      /* optional memref for stick */ mlir::Value optionalOutputMemRefForStick, 
      bool disableSaturation = false);

  bool isStickifiedOutput() { return ioIsStick[inputNum]; }
  void prepareInsideTiledLoop(
      KrnlBuilder &kb, DimsExpr &tiledOuterIndices, IndexExpr E1);

  void loadComputeStore(KrnlBuilder &kb,
      MultiValuesOfF32IterateBodyFn processVectorOfF32Vals, IndexExpr l,
      int64_t u, mlir::Value tempBufferMemRef = nullptr);

  static const int64_t archVL = 8;
  static const int64_t stickLen = 64;

private:
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, VectorBuilder, SCFBuilder, MathBuilder, ZLowBuilder>;

  DimsExpr computeAccessFct(
      mlir::Value val, DimsExpr &loopIndices, IndexExpr additionalInnerOffset);
  void loadVector(MDBuilder &create, DimsExpr &localOuterIndices, IndexExpr l,
      int64_t u, int64_t i);
  void storeVector(MDBuilder &create, DimsExpr &localOuterIndices, IndexExpr l,
      int64_t u, int64_t o, mlir::Value tempBufferMemRef,
      mlir::Value outputHigh, mlir::Value outputLow);

  // Inputs.
  mlir::SmallVector<mlir::Value, 4> ioOriginalOper;   // Untransformed opers.
  mlir::SmallVector<mlir::Value, 4> ioOriginalMemRef; // Memref opers.
  bool disableSaturation;

  // Computed values outside the loop.
  mlir::SmallVector<mlir::Value, 4> ioMemRef; // Memrefs possibly reinterpreted.
  mlir::BitVector ioIsStick;     // True if the operand has a stick data layout.
  mlir::BitVector ioIsBroadcast; // True if the last dimension is a broadcast
  mlir::BitVector ioIsBuffer;    // True if oper has <1, 8> static shapes.
  int64_t inputNum, ioNum;

  // Computed value inside the tiled loop, preparing for a stick loop.
  DimsExpr ioStickOffsets; // Offset pointing to current stick.
  mlir::SmallVector<mlir::Value, 4> inputHigh, inputLow; // High/low values.
  DimsExpr outerIndices; // Indices of the outer loop (orig, not tiles)

  // Helper values.
  IndexExpr litZero, lit2, litStickLen;
  mlir::VectorType vecF16Type, vecF32Type;
};

} // namespace onnx_mlir

// Include template code.
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp.inc"

#endif
