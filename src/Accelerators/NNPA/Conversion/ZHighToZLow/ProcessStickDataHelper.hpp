/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickDataHelper.cpp - Process Stick data ----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the helper class to support Stick handling.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PROCESS_STICK_DATA_HELPER_H
#define ONNX_MLIR_PROCESS_STICK_DATA_HELPER_H

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

namespace onnx_mlir {

// Class that supports memref that are either to normal reference or stick
// references, and that either fully defined in their innermost dimensions (no
// broadcast) or that are broadcasted in their innermost dimensions. Broadcast
// can also occur in higher dimensions (either compile time or runtime), it does
// not matter here. But broadcasting or not in the innermost dimension must be
// known at compile time.
//
// Most important: this class expect an external loop that iterates over chunks
// of 64 values in the innermost dimension (called outer loops), and one loop
// that iterates by chunks of TotVL within the group of 64 innermost dimension
// value (the stick).
//
// There are 4 basic operations: initialization before any loops (init method),
// initialization before the stick loop (beforeStickLoop), and actions inside
// the stick loop (beforeCompute, afterCompute, get/set values).
//
// There is also  methods (named loadComputeStore) that assist computing with
// the data. The unit of work is a vector of 4 float (f32).
class UnifiedStickSupport {
public:
  // Initialize:
  //
  // OriginalVal must be the tensor version (prior to MemRef conversions);
  // otherwise it can be a MemRef.
  //
  // OriginalMemRef is the original translated tensor in memref.
  //
  // At least one of isRead or isWrite must be
  //
  // DisableSaturation indicates if saturation should not occur during
  // conversions from f32 to dlf16.
  //
  // During init, reference is classified and if the reference is to a stick
  // data, then we perform a reinterpret cast to facilitate stick accesses.
  // Most values are stride through using the n-dimensional loop iterations; the
  // exception is for "buffers", which are expected to have a [1,8] shape. They
  // will are used without access functions (typically (0,0..3) and (0, 4..7)).
  UnifiedStickSupport(KrnlBuilder &kb, mlir::Value originalVal,
      mlir::Value originalMemRef, bool isRead, bool isWrite,
      bool disableSaturation) {
    init(kb, originalVal, originalMemRef, isRead, isWrite, disableSaturation);
  }
  void init(KrnlBuilder &kb, mlir::Value originalVal,
      mlir::Value originalMemRef, bool isRead, bool isWrite,
      bool disableSaturation);
  UnifiedStickSupport() = default;

  // Run before the stickified inner loop. This will load read values that are
  // broadcasted. For stickified references, the stick offset will also be
  // computed using the specified outer indices.  Outer indices should iterate
  // over sticks (be blocked by stickLen). E.g. the innermost index goes from 0,
  // 64, 128,... So this call "locks in" the outer indices for all but the
  // innermost index which will be allow to go from 0 to up to 63 (within that
  // stick).
  void beforeStickLoop(KrnlBuilder &kb, DimsExpr &outerIndices);

  // Perform the read and the write operations as needed. Index
  // offsetWithinStick is the offset into the current stick. processing totVL =
  // archVL * unrollVL values at a time. Because vector registers cannot be
  // virtualized (due to the asm support of f32 <-> df16 conversions), we must
  // iterates by chunks of 8 values. The offsetWithinVector is that index
  // (between 0 and UnrollVL) pointing to the current ArchVL vector being
  // processed.
  void beforeCompute(
      KrnlBuilder &kb, IndexExpr offsetWithinStick, int64_t offsetWithinVector);
  void afterCompute(KrnlBuilder &kb, IndexExpr offsetWithinStick,
      int64_t offsetWithinVector, mlir::Value tempBufferMemRef);

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

  // Globals that are to be used for this interface.
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
  // Computed values outside of the stick loop for broadcast, inside the stick
  // loop for non-broadcast.
  mlir::Value highVal, lowVal;
};

// Higher level of interface. It allows the grouping of several
// UnifiedStickSupport (or USS) to be grouped as a single unit. This allows to
// gather all the references used in a kernel and process them in one single
// call.
//
// It also provide a higher level interface to compute values in the innermost
// loop (the one computing archVL=8 values at a time).
//
// This struct basically defined a (externally accessible) list, and operations
// that can be performed on them.
struct UnifiedStickSupportList {

  // Init, mimic the init of a single USS.
  UnifiedStickSupportList(KrnlBuilder &kb, mlir::ValueRange originalVals,
      mlir::ValueRange originalMemRefs, mlir::BitVector isReads,
      mlir::BitVector isWrites, bool disableSaturation);
  void init(KrnlBuilder &kb, mlir::ValueRange originalVals,
      mlir::ValueRange originalMemRefs, mlir::BitVector isReads,
      mlir::BitVector isWrites, bool disableSaturation);
  UnifiedStickSupportList() = default;

  // Mimic the beforeStickLoop of a single USS.
  // Outer indices should iterate over sticks (be blocked by stickLen). E.g. the
  // innermost index goes from 0, 64, 128,... Different actions are performed
  // for read vs write reference, stick vs broadcast///
  void beforeStickLoop(KrnlBuilder &kb, DimsExpr &outerIndices);

  void beforeCompute(
      KrnlBuilder &kb, IndexExpr offsetWithinStick, int64_t offsetWithinVector);
  void afterCompute(KrnlBuilder &kb, IndexExpr offsetWithinStick,
      int64_t offsetWithinVector, mlir::ValueRange tempBufferMemRefs);

  // Function that receives vectors of 4xf32 values (one per USS that is a read
  // in the list, in the same list order) and compute one resulting 4xF32 value.
  // That value must be returned. Used in loadComputeStore below.
  using IterateFctOver4xF32 = std::function<mlir::Value(const KrnlBuilder &b,
      mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals)>;
  // This function works only when initialized with originVals and originMemRefs
  // to have only one output. Typically used for elementwise operations. All
  // UnifiedStickSupport in list are expected to be initialized. A more generic
  // version with no restriction is listed below.
  void loadComputeStore(KrnlBuilder &kb,
      IterateFctOver4xF32 processVectorOfF32Vals, IndexExpr offsetWithinStick,
      int64_t offsetWithinVector, mlir::Value tempBufferMemRef = nullptr);

  // Function that receives vectors of 4xf32 values, some of which may be input
  // values (to be used by the function), some of which may be output are are
  // expected to be modified by the function, and some that may be undefined
  // (maybe not needed for this version of the operation). It is the
  // responsibility of the function to only use values defined as read, and any
  // modified values that were not defined as write will be lost.
  using GenericIterateFctOver4xF32M = std::function<void(
      const KrnlBuilder &b, mlir::SmallVectorImpl<mlir::Value> &listOfF32Vals)>;
  void genericLoadComputeStore(KrnlBuilder &kb,
      GenericIterateFctOver4xF32M processVectorOfF32Vals,
      IndexExpr offsetWithinStick, int64_t offsetWithinVector);

  mlir::SmallVector<UnifiedStickSupport, 4> list;
};

} // namespace onnx_mlir

#endif
