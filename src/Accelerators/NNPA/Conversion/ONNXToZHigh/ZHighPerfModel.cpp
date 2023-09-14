/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighPerfModel.cpp - Deciding ONNX vs ZHigh for ops -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains model info to help decide for the relevant NNPA ops if
// they are faster / slower than their equivalent CPU versions.
//
//===----------------------------------------------------------------------===//

// hi alex: determine which one are really needed
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ZHighPerfModel.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "zhigh-perf-model"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// This is the implementation of Excel.ceiling, which result the smallest value
// that is greater or equal to A and is a multiple of B.
inline int64_t roundToNextMultiple(int64_t a, int64_t b) {
  // Ceil only works with unsigned number, but shapes are given as signed
  // numbers. Do the necessary checks/conversions here.
  assert(a >= 0 && "expected nonnegative");
  assert(b > 0 && "expected strictly positive");
  uint64_t aa = a;
  uint64_t bb = b;
  uint64_t ceilDiv = (aa + bb - 1) / bb;
  return ceilDiv * bb;
}

// Return true with a debug message reporting reason for success on NNPA.
inline bool fasterOnNNPA(Operation *op, std::string msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "Faster on NNPA: " << msg << " for op: ";
    op->dump();
  });
  return true;
}

// Return false with a debug message reporting reason for failure on NNPA.
inline bool fasterOnCPU(Operation *op, std::string msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "Faster on CPU: " << msg << " for op: ";
    op->dump();
  });
  return false;
}

} // namespace

bool isElementwiseFasterOnNNPA(Operation *op, Value lhs, Value rhs,
    const DimAnalysis *dimAnalysis, double relativeNNPASpeedup) {
  // At this time, use only 1 of the two
  ShapedType lhsType = lhs.getType().dyn_cast_or_null<ShapedType>();
  assert(lhsType && lhsType.hasRank() && "expected shaped type with rank");
  int64_t lhsRank = lhsType.getRank();
  assert(lhsRank <= 4 && "expected rank <= 4");
  llvm::ArrayRef<int64_t> shape = lhsType.getShape();
  int64_t e4 = lhsRank >= 4 ? shape[lhsRank - 4] : 1;
  int64_t e3 = lhsRank >= 3 ? shape[lhsRank - 3] : 1;
  int64_t e2 = lhsRank >= 2 ? shape[lhsRank - 2] : 1;
  int64_t e1 = lhsRank >= 1 ? shape[lhsRank - 1] : 1;

  // Disqualify if e1 is too small (full is 64, so shoot for half full).
  if (e1 > 0 && e1 < 32)
    return fasterOnCPU(op, "elementwise has too small e1");
  // If e1 or e2 are runtime, assume they will be large enough.
  if (e1 == ShapedType::kDynamic || e2 == ShapedType::kDynamic)
    return fasterOnNNPA(op, "elementwise has runtime e1 or e2");
  // If larger dims are runtime, assume it might just be size 1.
  if (e3 == ShapedType::kDynamic)
    e3 = 1;
  if (e4 == ShapedType::kDynamic)
    e4 = 1;
  // Cedric's spreadsheet calculations.
  int64_t computed2dFMA =
      e4 * e3 * roundToNextMultiple(e2, 2) * roundToNextMultiple(e1, 64);
  computed2dFMA = (double)computed2dFMA * relativeNNPASpeedup;
  assert(computed2dFMA > 0 && "dyn size should have been removed");
  // Cedric's model show still significant benefits for 16 full tiles,
  // arbitrarily assume cross over at 8. Will need new measurements on this.
  if (computed2dFMA < 8 * 2048)
    return fasterOnCPU(
        op, "elementwise computed 2D FMA is too small (<8 full tiles)");
  return fasterOnNNPA(op, "elementwise has enough computations");
}

bool isMatMulFasterOnNNPA(Operation *op, Value a, Value b, bool aTransposed,
    bool bTransposed, const DimAnalysis *dimAnalysis) {
  // Scanning A.
  ShapedType aType = a.getType().dyn_cast_or_null<ShapedType>();
  assert(aType && aType.hasRank() && "expected shaped type with A rank");
  int64_t aRank = aType.getRank();
  assert(aRank >= 2 && aRank <= 3 && "expected A rank 2..3");
  llvm::ArrayRef<int64_t> aShape = aType.getShape();
  int64_t aB = aRank >= 3 ? aShape[aRank - 3] : 1;
  int64_t aNIndex = aTransposed ? aRank - 1 : aRank - 2;
  int64_t aMIndex = aTransposed ? aRank - 2 : aRank - 1;
  int64_t aN = aShape[aNIndex];
  int64_t aM = aShape[aMIndex];
  // Scanning B.
  ShapedType bType = b.getType().dyn_cast_or_null<ShapedType>();
  assert(bType && bType.hasRank() && "expected shaped type with B rank");
  int64_t bRank = bType.getRank();
  assert(bRank >= 2 && bRank <= 3 && "expected B rank 2..3");
  llvm::ArrayRef<int64_t> bShape = bType.getShape();
  int64_t bB = bRank >= 3 ? bShape[bRank - 3] : 1;
  int64_t bMIndex = bTransposed ? bRank - 1 : bRank - 2;
  int64_t bKIndex = bTransposed ? bRank - 2 : bRank - 1;
  int64_t bM = bShape[bMIndex];
  int64_t bK = bShape[bKIndex];
  assert(aM == bM && "expected M dims to be identical");
  // Rules common to matmul with/without broadcast.
  // Make sure the constant lower dim of the matmul are large enough.
  if (aM > 0 && aM < 32)
    return fasterOnCPU(op, "matmul no-broadcast M dim too small)");
  if (bK > 0 && bK < 32)
    return fasterOnCPU(op, "matmul no-broadcast K dim too small)");
  // Assume the dynamic lower dim of the matmul will be large enough.
  if (aM == ShapedType::kDynamic || bK == ShapedType::kDynamic)
    return fasterOnNNPA(op, "matmul no-broadcast has runtime M or K");
  // Assume the N dim of the matmul will be small.
  if (aN == ShapedType::kDynamic)
    aN = 1;
  // Determine if we have a broadcast (will change cost calculations).
  bool hasBroadcast = true;
  if (aRank == 2 && bRank == 2) // No broadcast dim.
    hasBroadcast = false;
  else if (aB == 1 && bB == 1) // No broadcast because both 1.
    hasBroadcast = false;
  else if (aRank == 3 && bRank == 3 &&
           dimAnalysis->sameDim(a, aRank - 3, b, bRank - 3))
    hasBroadcast = false;
  // Assume the B dim of the matmul will be small.
  if (aB == ShapedType::kDynamic)
    aB = 1;
  if (bB == ShapedType::kDynamic)
    bB = 1;

  // Handle case without broadcast.
  if (!hasBroadcast) {
    int64_t computed2dFMA = aB * roundToNextMultiple(aN, 2) *
                            roundToNextMultiple(aM, 64) *
                            roundToNextMultiple(bK, 64);
    assert(computed2dFMA > 0 && "dyn size should have been removed");
    // Cedric's model show still  benefits for 64^3 == 128 full tiles.
    if (computed2dFMA < 64 * 64 * 64)
      return fasterOnCPU(
          op, "matmul no-broadcast computed 2D FMA is too small (<64^3 flops)");
    return fasterOnNNPA(op, "matmul no-broadcast has enough computations");
  }
  // Else we have broadcast.
  // Virtual E2/E4 from Cedric's spreadsheet: TODO, need refinement.
  int B = bB == 1 ? aB : bB;
  int64_t virtualNB;
  if (aN >= 128)
    virtualNB = aN; // No B?
  else if (aM >= 64)
    virtualNB = aN * std::min(2, B);
  else if (aM >= 32)
    virtualNB = aN * std::min(4, B);
  else
    virtualNB = aN * std::min(8, B);
  int64_t computed2dFMA =
      virtualNB * roundToNextMultiple(aM, 64) * roundToNextMultiple(bK, 64);
  if (computed2dFMA < 64 * 64 * 64)
    return fasterOnCPU(
        op, "matmul broadcast computed 2D FMA is too small (<64^3 flops)");
  return fasterOnNNPA(op, "matmul broadcast has enough computations");
}
