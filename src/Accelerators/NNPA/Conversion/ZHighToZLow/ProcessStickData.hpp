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

class UnifiedStickSupport {
public:
  // For stickified values, OriginalVal must be the tensor version (prior to
  // MemRef conversions); otherwise it can be a MemRef. E1 is the innermost
  // (possibly stickified) dimension. At least one of isRead or isWrite must be
  // true. Same holds for init.
  UnifiedStickSupport(KrnlBuilder &kb, mlir::Value originalVal,
      mlir::Value originalMemRef, IndexExpr E1, bool isRead, bool isWrite,
      bool disableSaturation)  {
    init(kb, originalVal, originalMemRef, E1, isRead, isWrite,
        disableSaturation);
  }
  void init(KrnlBuilder &kb, mlir::Value originalVal,
      mlir::Value originalMemRef, IndexExpr E1, bool isRead, bool isWrite,
      bool disableSaturation);

  // Run before the stickified inner loop. This will load read values that are
  // broadcasted. For stickified references, the stick offset will also be
  // computed.  Outer indices should iterate over sticks (be blocked by
  // stickLen). E.g. the innermost index goes from 0, 64, 128,...
  void beforeStickLoop(KrnlBuilder &kb, DimsExpr &outerIndices, IndexExpr E1);
  // Perform the read and the write operations as needed.
  void beforeCompute(KrnlBuilder &kb, IndexExpr l, int64_t u);
  void afterCompute(
      KrnlBuilder &kb, IndexExpr l, int64_t u, mlir::Value tempBufferMemRef);

  // Get the values to perform the computation, and must set the values for the
  // values that will be stored.
  void get4xF32Vals(mlir::Value &highVal, mlir::Value &lowVal);
  void set4xF32Vals(mlir::Value highVal, mlir::Value lowVal);

  // Getters
  bool hasRead() { return isRead; }
  bool hasWrite() { return isWrite; }
  bool hasStick() { return isStick; }
  bool isInitialized() { return isRead || isWrite; }
  mlir::Value getOriginalVal() { return originalVal; }
  mlir::Value getOriginalMemRef() { return originalMemRef; }

  static const int64_t archVL = 8;
  static const int64_t stickLen = 64;

private:
  static DimsExpr computeAccessFct(
      mlir::Value val, DimsExpr &loopIndices, IndexExpr additionalInnerOffset);

  // Input and characterization.
  mlir::Value originalVal, originalMemRef, memRef;
  bool isRead = false;  // Detect data is uninitialized if both isRead and
  bool isWrite = false; // isWrite are false.
  bool disableSaturation;
  bool isStick, isBroadcast, isBuffer;
  // Computed values outside of the stick loop.
  DimsExpr outerIndices;
  IndexExpr stickOffset; // For stick values.
  mlir::Value highVal, lowVal;
};

struct UnifiedStickMemSupportForKernels {

  UnifiedStickMemSupportForKernels(KrnlBuilder &kb,
      mlir::ValueRange originalVals, mlir::ValueRange originalMemRefs,
      IndexExpr E1, mlir::BitVector isReads, mlir::BitVector isWrites,
      bool disableSaturation);
  void init(KrnlBuilder &kb, mlir::ValueRange originalVals,
      mlir::ValueRange originalMemRefs, IndexExpr E1, mlir::BitVector isReads,
      mlir::BitVector isWrites, bool disableSaturation);

  // Outer indices should iterate over sticks (be blocked by stickLen). E.g. the
  // innermost index goes from 0, 64, 128,...
  void beforeStickLoop(KrnlBuilder &kb, DimsExpr &outerIndices, IndexExpr E1);

  void beforeCompute(KrnlBuilder &kb, IndexExpr l, int64_t u);
  void afterCompute(KrnlBuilder &kb, IndexExpr l, int64_t u,
      mlir::ValueRange tempBufferMemRefs);

  // Function that receives vectors of 4xf32 values and compute one resulting
  // 4xF32 value. Used in loadComputeStore below.
  using IterateFctOver4xF32 = std::function<mlir::Value(const KrnlBuilder &b,
      mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals)>;
  // This function works only when initialized with originVals and originMemRefs
  // to have only one output. Typically used for elementwise operations. All
  // UnifiedStickSupport in list are expected to be initialized.
  void loadComputeStore(KrnlBuilder &kb,
      IterateFctOver4xF32 processVectorOfF32Vals, IndexExpr l, int64_t u,
      mlir::Value tempBufferMemRef = nullptr);

  mlir::SmallVector<UnifiedStickSupport, 4> list;
};

} // namespace onnx_mlir

// Include template code.
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp.inc"

#endif
