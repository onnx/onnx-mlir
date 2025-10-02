/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickData.cpp - Process Stick data ----------------===//
//
// Copyright 2024-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to Krnl/Affine/SCF
// operations that operates on stickified input/output data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickDataHelper.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
// #include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
// #include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
// #include "src/Support/SmallVectorHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// UnifiedStickSupport.

void UnifiedStickSupport::init(KrnlBuilder &kb, mlir::Value originalVal,
    mlir::Value originalMemRef, IndexExpr E1, bool isRead, bool isWrite,
    bool disableSaturation) {
  // Check it was not already initialized.
  assert(!isInitialized() && "should not initialize twice");
  // Save values to class.
  this->originalVal = originalVal;
  this->originalMemRef = originalMemRef;
  this->memRef = originalMemRef;
  this->isRead = isRead;
  this->isWrite = isWrite;
  this->disableSaturation = disableSaturation;
  assert((isRead || isWrite) && "must be at least read or write");
  // Classify: get info and classify.
  assert(E1.isLiteral());
  Type originalType = originalVal.getType();
  auto originalShape = getShape(originalType);
  int64_t innermostShape = getShape(originalType, -1);
  isBroadcast =
      innermostShape == 1 && !isWrite; // Shape=1, there will be some broadcast.
  isStick = zhigh::isZTensor(originalType);
  isBuffer = originalShape.size() == 2 && originalShape[0] == 1 &&
             originalShape[1] == archVL;
  if (isStick && !isBroadcast) {
    // Overwrite memRefValue as a flattened [2, 64] view.
    assert(
        zhigh::supportedLayoutForCompilerGeneratedStickUnstick(originalVal) &&
        "unsupported layout");
    MultiDialectBuilder<MemRefBuilder> create(kb);
    IndexExpr lit2 = LitIE(2);
    IndexExpr litStickLen = LitIE(stickLen);
    DimsExpr castShape = {lit2, litStickLen};
    this->memRef = create.mem.reinterpretCast(originalMemRef, castShape);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "  Operand: ";
    originalVal.dump();
    llvm::dbgs() << "    " << (isStick ? "is stick " : "")
                 << (isBroadcast ? "is broadcast " : "")
                 << (isBuffer ? "is buffer " : "") << "operand\n";
  });
}

void UnifiedStickSupport::beforeStickLoop(
    KrnlBuilder &kb, DimsExpr &outerIndices, IndexExpr E1) {
  if (!isInitialized())
    return;
  MultiDialectBuilder<KrnlBuilder, VectorBuilder, ZLowBuilder> create(kb);
  this->outerIndices = outerIndices;
  // Initialize data that will hold data and stick offsets.
  stickOffset = nullptr;
  highVal = lowVal = nullptr;
  // Handling for broadcast or stick
  if (isBroadcast) {
    IndexExpr lit0 = LitIE(0);
    DimsExpr accessFct = computeAccessFct(memRef, outerIndices, lit0);
    Value scalar = create.krnl.loadIE(memRef, accessFct);
    if (isStick) {
      Type f16Type = create.getBuilder().getF16Type();
      VectorType vecF16Type = VectorType::get({archVL}, f16Type);
      Value vecF16 = create.vec.splat(vecF16Type, scalar);
      Value vecHigh, vecLow;
      create.zlow.convertDLF16ToF32(vecF16, vecHigh, vecLow);
      highVal = lowVal = vecHigh; // Splatted, low and high are the same.
    } else {
      Type f32Type = create.getBuilder().getF32Type();
      VectorType vecF32Type = VectorType::get({archVL / 2}, f32Type);
      highVal = lowVal = create.vec.splat(vecF32Type, scalar);
    }
  } else if (isStick) {
    // Stick data that is not broadcasted: need SIMD support.
    // Translate the tile index to the stick offset. Have to
    // give the actual indices, not the tiled ones.
    IndexExpr lit0 = LitIE(0);
    DimsExpr accessFct = computeAccessFct(originalMemRef, outerIndices, lit0);
    Value offset =
        create.krnl.getLinearOffsetIndexIE(originalMemRef, accessFct);
    stickOffset = DimIE(offset).floorDiv(stickLen);
  }
}

// For read references.
void UnifiedStickSupport::beforeCompute(
    KrnlBuilder &kb, IndexExpr offsetWithinStick, int64_t offsetWithinVector) {
  if (!isRead)
    return;
  // Broadcast loaded inside the outer loops.
  if (isBroadcast) {
    assert(highVal && lowVal && "should be read in beforeStickLoop");
    return;
  }
  // Compute innermost offset (offsetWithinStick: index of loop,
  // offsetWithinVector:  unrolling by archVL).
  MultiDialectBuilder<MathBuilder, VectorBuilder, ZLowBuilder> create(kb);
  DimsExpr currIndices;
  IndexExpr currStickOffset, currOffset;
  if (isBuffer) {
    // For buffer, all accesses are zero.
    IndexExpr lit0 = LitIE(0);
    currIndices = {lit0, lit0};
    currStickOffset = lit0;
    currOffset = lit0;
  } else {
    currIndices = outerIndices;
    currStickOffset = stickOffset;
    currOffset = offsetWithinStick + (archVL * offsetWithinVector);
  }
  if (isStick) {
    DimsExpr accessFct = {DimIE(currStickOffset), currOffset};
    Type f16Type = create.getBuilder().getF16Type();
    VectorType vecF16Type = VectorType::get({archVL}, f16Type);
    Value vecOfDLF16 = create.vec.loadIE(vecF16Type, memRef, accessFct);
    create.zlow.convertDLF16ToF32(vecOfDLF16, highVal, lowVal);
  } else {
    DimsExpr localIndices = DimListIE(currIndices);
    DimsExpr accessFct = computeAccessFct(memRef, localIndices, currOffset);
    Type f32Type = create.getBuilder().getF32Type();
    VectorType vecF32Type = VectorType::get({archVL / 2}, f32Type);
    highVal = create.vec.loadIE(vecF32Type, memRef, accessFct);
    Value lowOffset = create.math.constantIndex(archVL / 2);
    lowVal = create.vec.loadIE(vecF32Type, memRef, accessFct, {lowOffset});
  }
  assert(highVal && lowVal && "expected high/low val to be defined");
}

// For write references.
void UnifiedStickSupport::afterCompute(KrnlBuilder &kb,
    IndexExpr offsetWithinStick, int64_t offsetWithinVector,
    Value tempBufferMemRef) {
  if (!isWrite)
    return;
  assert(!isBroadcast && "output should not be broadcast");
  assert(highVal && lowVal && "expected high/low val to be defined");
  // Compute innermost offset (offsetWithinStick: index of loop,
  // offsetWithinVector:  unrolling by archVL).
  MultiDialectBuilder<MathBuilder, VectorBuilder, ZLowBuilder> create(kb);
  DimsExpr currIndices;
  IndexExpr currOffset, currStickOffset;
  Value currMemref;
  if (isBuffer || tempBufferMemRef) {
    // For buffer, all accesses are zero.
    IndexExpr lit0 = LitIE(0);
    currIndices = {lit0, lit0};
    currStickOffset = lit0;
    currOffset = lit0;
    currMemref = tempBufferMemRef ? tempBufferMemRef : memRef;
  } else {
    currIndices = outerIndices;
    currStickOffset = stickOffset;
    currOffset = offsetWithinStick + (archVL * offsetWithinVector);
    currMemref = memRef;
  }
  if (isStick) {
    Value dlf16 =
        create.zlow.convertF32ToDLF16(highVal, lowVal, disableSaturation);
    DimsExpr accessFct = {DimIE(currStickOffset), currOffset};
    create.vec.storeIE(dlf16, currMemref, accessFct);
  } else {
    DimsExpr localIndices = DimListIE(currIndices);
    DimsExpr accessFct = computeAccessFct(currMemref, localIndices, currOffset);
    create.vec.storeIE(highVal, currMemref, accessFct);
    Value lowOffset = create.math.constantIndex(archVL / 2);
    create.vec.storeIE(lowVal, currMemref, accessFct, {lowOffset});
  }
}

void UnifiedStickSupport::get4xF32Vals(Value &highVal, Value &lowVal) {
  highVal = this->highVal;
  lowVal = this->lowVal;
  assert(highVal && lowVal && "expected high/low val to be defined");
}

void UnifiedStickSupport::set4xF32Vals(Value highVal, Value lowVal) {
  assert(highVal && lowVal && "expected high/low val to be defined");
  this->highVal = highVal;
  this->lowVal = lowVal;
}

/* static */ DimsExpr UnifiedStickSupport::computeAccessFct(
    Value val, DimsExpr &loopIndices, IndexExpr additionalInnerOffset) {
  DimsExpr accessFct;
  int64_t valRank = getRank(val.getType());
  int64_t loopRank = loopIndices.size();
  accessFct.clear();
  for (int i = 0; i < valRank; ++i) {
    if (getShape(val.getType(), i) == 1) {
      // Possible broadcast situation, ignore loop index, just use 0.
      accessFct.emplace_back(LitIE(0));
    } else {
      // Use innermost loop indices.
      IndexExpr index = loopIndices[loopRank - valRank + i];
      if (i == valRank - 1)
        index = index + additionalInnerOffset; // Add innermost offset.
      accessFct.emplace_back(index);
    }
  }
  return accessFct;
}

//===----------------------------------------------------------------------===//
// UnifiedStickSupportList: higher level support for collection of
// UnifiedStickSupport.

UnifiedStickSupportList::UnifiedStickSupportList(KrnlBuilder &kb,
    ValueRange originalVals, ValueRange originalMemRefs, IndexExpr E1,
    mlir::BitVector isReads, mlir::BitVector isWrites, bool disableSaturation) {
  init(kb, originalVals, originalMemRefs, E1, isReads, isWrites,
      disableSaturation);
}

void UnifiedStickSupportList::init(KrnlBuilder &kb, ValueRange originalVals,
    ValueRange originalMemRefs, IndexExpr E1, mlir::BitVector isReads,
    mlir::BitVector isWrites, bool disableSaturation) {
  int64_t size = originalVals.size();
  assert((int)originalMemRefs.size() == size && "bad memref size");
  assert((int)isReads.size() == size && "bad isRead size");
  assert((int)isWrites.size() == size && "bad isWrite size");
  list.clear();
  for (int64_t i = 0; i < size; ++i) {
    UnifiedStickSupport uss(kb, originalVals[i], originalMemRefs[i], E1,
        isReads[i], isWrites[i], disableSaturation);
    list.emplace_back(uss);
  }
}

void UnifiedStickSupportList::beforeStickLoop(
    KrnlBuilder &kb, DimsExpr &outerIndices, IndexExpr E1) {
  int64_t size = list.size();
  for (int64_t i = 0; i < size; ++i)
    list[i].beforeStickLoop(kb, outerIndices, E1);
}

void UnifiedStickSupportList::beforeCompute(
    KrnlBuilder &kb, IndexExpr offsetWithinStick, int64_t offsetWithinVector) {
  int64_t size = list.size();
  for (int64_t i = 0; i < size; ++i)
    list[i].beforeCompute(kb, offsetWithinStick, offsetWithinVector);
}

void UnifiedStickSupportList::afterCompute(KrnlBuilder &kb,
    IndexExpr offsetWithinStick, int64_t offsetWithinVector,
    ValueRange tempBufferMemRefs) {
  int64_t size = list.size();
  int64_t tempSize = tempBufferMemRefs.size();
  for (int64_t i = 0; i < size; ++i) {
    Value tempMemRef = i < tempSize ? tempBufferMemRefs[i] : nullptr;
    list[i].afterCompute(kb, offsetWithinStick, offsetWithinVector, tempMemRef);
  }
}

void UnifiedStickSupportList::loadComputeStore(KrnlBuilder &kb,
    IterateFctOver4xF32 processVectorOfF32Vals, IndexExpr offsetWithinStick,
    int64_t offsetWithinVector, Value tempBufferMemRef) {
  // Load values;
  beforeCompute(kb, offsetWithinStick, offsetWithinVector);
  // Gather input value, and usms that hold the store.
  int64_t size = list.size();
  mlir::SmallVector<Value, 4> highInputVals, lowInputVals;
  UnifiedStickSupport *storeUSMS = nullptr;
  for (int64_t i = 0; i < size; ++i) {
    assert(list[i].isInitialized() && "expect all to be initialized");
    if (list[i].hasRead()) {
      Value highVal, lowVal;
      list[i].get4xF32Vals(highVal, lowVal);
      highInputVals.emplace_back(highVal);
      lowInputVals.emplace_back(lowVal);
    }
    if (list[i].hasWrite()) {
      assert(storeUSMS == nullptr && "only one store allowed");
      storeUSMS = &(list[i]);
    }
  }
  // Compute output values.
  Value highOutputVal = processVectorOfF32Vals(kb, highInputVals);
  Value lowOutputVal = processVectorOfF32Vals(kb, lowInputVals);
  // Store values.
  storeUSMS->set4xF32Vals(highOutputVal, lowOutputVal);
  storeUSMS->afterCompute(
      kb, offsetWithinStick, offsetWithinVector, tempBufferMemRef);
}

void UnifiedStickSupportList::genericLoadComputeStore(KrnlBuilder &kb,
    GenericIterateFctOver4xF32M processVectorOfF32Vals,
    IndexExpr offsetWithinStick, int64_t offsetWithinVector) {
  beforeCompute(kb, offsetWithinStick, offsetWithinVector);
  int64_t size = list.size();
  // Set the low/high values for all USS on the list, setting the values that
  // should be read to the proper values.
  mlir::SmallVector<Value, 4> highVals(size, nullptr), lowVals(size, nullptr);
  for (int64_t i = 0; i < size; ++i) {
    if (list[i].isInitialized() && list[i].hasRead())
      list[i].get4xF32Vals(highVals[i], lowVals[i]);
  }
  // This function is responsible for updating the values that need to be
  // stored.
  processVectorOfF32Vals(kb, highVals);
  processVectorOfF32Vals(kb, lowVals);
  // Update the USS with the low/high values that were calculated above.
  for (int64_t i = 0; i < size; ++i) {
    if (list[i].isInitialized() && list[i].hasWrite())
      list[i].set4xF32Vals(highVals[i], lowVals[i]);
  }
  afterCompute(kb, offsetWithinStick, offsetWithinVector, {});
}

} // namespace onnx_mlir
