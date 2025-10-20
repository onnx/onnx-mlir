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

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickDataHelper.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

// True: optimize with write of 8 values for the last iter of a stick,
// regardless if we need each of the results. False, conservatively only write
// the "allowed" values in the output for the last couple of values.
#define STICK_OUTPUT_WRITE_PAST_BOUNDS true

// Include necessary info from elementwise so as to gen code here.
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"

//===----------------------------------------------------------------------===//
// Handle quantization on stickified inputs
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
// Implementation of quantize helper function.
void emitDynamicQuantizationLinearMinMaxFromStickifiedInput(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Value input, StringAttr inputLayout, Value &inputMin, Value &inputMax,
    bool enableSIMD, bool enableParallel) {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MathBuilder, MemRefBuilder, VectorBuilder>;
  MDBuilder create(rewriter, loc);

  // Extract dims from input, set lbs/ubs.
  DimsExpr dims;
  create.krnlIE.getShapeAsSymbols(input, dims);
  int64_t rank = dims.size();
  IndexExpr zero = LitIE(0);
  DimsExpr lbs(rank, zero);
  DimsExpr ubs = dims;

  // Decide parameters.
  // UnrollVL decides how many vectors of 8 DLF16 will be processed at once.
  int64_t unrollVL = 4; // Experimentally good unroll factor.
  int64_t archVL = 8;   // DLF16.
  int64_t totVL = unrollVL * archVL;

  // If not parallel, threadNum = 1, forExplicitParallelLoopIE will simply pass
  // through the lb/ub, so ok to have parID = 0 for the sequential cases.
  int64_t parId = 0;
  int64_t threadNum = 1;
  if (enableParallel) {
    int64_t parId = tryCreateKrnlParallel(create.krnl, op,
        "simd min/max for DQL in parallel", {}, lbs, ubs, 0, rank - 1, {},
        /*min iter for going parallel*/ 8, /*createKrnlParallel=*/false);
    if (parId == -1) {
      enableParallel = false;
    } else {
      threadNum = 8; // TODO use more flexible value.
    }
  }

  // Alloc temp buffers (more when using parallel).
  Type f32Type = rewriter.getF32Type();
  // For each thread, we can use totVL temp values for the current min/max.
  // But to increase the compute ratio over mem, we will reuse the same tmp
  // memory location for a pair of totVL values being processed.
  int64_t tmpSizePerThread = totVL / 2; // Reduce pair in same tmp.
  int64_t tmpSize = threadNum * tmpSizePerThread;
  MemRefType redType = MemRefType::get({tmpSize}, f32Type);
  VectorType vec8xF32Type = VectorType::get({archVL}, f32Type);
  VectorType vec4xF32Type = VectorType::get({archVL / 2}, f32Type);

  Value minTmp = create.mem.alignedAlloc(redType);
  Value maxTmp = create.mem.alignedAlloc(redType);

  // Init min and max.
  Value minInit = create.math.positiveInf(f32Type);
  Value splatMinInit = create.vec.splat(vec8xF32Type, minInit);
  Value maxInit = create.math.negativeInf(f32Type);
  Value splatMaxInit = create.vec.splat(vec8xF32Type, maxInit);
  // Could parallelize init, here main thread do it all. Use SIMD of 8x.
  for (int64_t offsetWithinVector = 0; offsetWithinVector < tmpSize;
       offsetWithinVector += 8) {
    IndexExpr offset = LitIE(offsetWithinVector);
    create.vec.storeIE(splatMinInit, minTmp, {offset});
    create.vec.storeIE(splatMaxInit, maxTmp, {offset});
  }

  // Reduction into these temps.
  IndexExpr tNum = LitIE(threadNum);
  create.krnl.forExplicitParallelLoopIE(lbs[parId], ubs[parId], tNum,
      [&](const KrnlBuilder &ck, ValueRange loopInd) {
        IndexExprScope scope(ck);
        IndexExpr t = DimIE(loopInd[0]);
        DimsExpr currDims = DimListIE(dims);
        // Reduce lbs, ubs for parallel region, if any.
        DimsExpr currLbs = DimListIE(lbs);
        DimsExpr currUbs = DimListIE(ubs);
        // In sequential cases (threadNum ==1, loopInd[1,2]== orig lb,ub).
        currLbs[parId] = SymIE(loopInd[1]);
        currUbs[parId] = SymIE(loopInd[2]);
        // Cannot use krnl because we may not have affine bounds.
        SCFBuilder sb(ck);
        IterateOverStickInputData<SCFBuilder>(
            sb, op, currLbs, currUbs, currDims, inputLayout, input, nullptr,
            unrollVL, /*enableParallel*/ false,
            /*prefetch, disable as it causes issue with affine*/ false,
            [&](const KrnlBuilder &b, SmallVectorImpl<Value> &vecOf4xF32Vals,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              int64_t size = vecOf4xF32Vals.size();
              assert((size == 2 || size == 2 * unrollVL) && "unexpected size");
              // Since all threads share the same tmpMin/Max, needs to offset by
              // t * <size for one thread>.
              IndexExpr threadOffset = SymIE(t) * tmpSizePerThread;
              size = size / 2; // handle pairs of 2, so size=1 or unrollVL.
              for (int i = 0; i < size; ++i) {
                Value val0 = vecOf4xF32Vals[2 * i];
                Value val1 = vecOf4xF32Vals[2 * i + 1];
                // Load appropriate tmp, compute min/max, store in tmp.
                IndexExpr offset = threadOffset + LitIE(4 * i);
                Value currMin =
                    create.vec.loadIE(vec4xF32Type, minTmp, {offset});
                Value currMax =
                    create.vec.loadIE(vec4xF32Type, maxTmp, {offset});
                currMin = create.math.min(currMin, val0);
                currMax = create.math.max(currMax, val0);
                currMin = create.math.min(currMin, val1);
                currMax = create.math.max(currMax, val1);
                create.vec.storeIE(currMin, minTmp, {offset});
                create.vec.storeIE(currMax, maxTmp, {offset});
              }
            },
            [&](const KrnlBuilder &b, Value scalarF32Val,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              Value currMin = create.krnl.loadIE(minTmp, {zero});
              Value currMax = create.krnl.loadIE(maxTmp, {zero});
              currMin = create.math.min(currMin, scalarF32Val);
              currMax = create.math.max(currMax, scalarF32Val);
              create.krnl.storeIE(currMin, minTmp, {zero});
              create.krnl.storeIE(currMax, maxTmp, {zero});
            }); // Iterate over stick.
      });       // Explicit parallel loop (sequential if threadNum==1).

  // Now we have all the partial min/max inside the minTmp/maxTmp: reduce each
  // vectors with each others. Main thread reduces all the values. Use SIMD of
  // 8x.
  Value finalVecMin = create.vec.loadIE(vec8xF32Type, minTmp, {zero});
  Value finalVecMax = create.vec.loadIE(vec8xF32Type, maxTmp, {zero});
  for (int offsetWithinVector = 8; offsetWithinVector < tmpSize;
       offsetWithinVector += 8) {
    IndexExpr offset = LitIE(offsetWithinVector);
    Value currMin = create.vec.loadIE(vec8xF32Type, minTmp, {offset});
    Value currMax = create.vec.loadIE(vec8xF32Type, maxTmp, {offset});
    finalVecMin = create.math.min(finalVecMin, currMin);
    finalVecMax = create.math.max(finalVecMax, currMax);
  }

  // Horizontal reduction of the vectors into a scalar.
  inputMin = create.vec.reduction(VectorBuilder::MIN, finalVecMin);
  inputMax = create.vec.reduction(VectorBuilder::MAX, finalVecMax);
}

//===----------------------------------------------------------------------===//
// Handle elementwise operations with stickified inputs/outputs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Process operation with stick inputs/output.

static void IterateOverStickInputOutput(const KrnlBuilder &kb, Operation *op,
    ValueRange operands /*converted*/, Value alloc, DimsExpr &outputDims,
    int64_t unrollVL, bool enableParallel, bool disableSaturation,
    bool enablePrefetch,
    UnifiedStickSupportList::IterateFctOver4xF32 processVectorOfF32Vals) {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, VectorBuilder, SCFBuilder, MathBuilder, ZLowBuilder>;

  // Init builder and scopes.
  MDBuilder create(kb);
  // IndexExprScope initialScope(b);
  //  Get info and check some inputs.
  int64_t rank = outputDims.size();
  int64_t d1 = rank - 1;
  IndexExpr E1 = outputDims[d1];
  assert(op->getNumResults() == 1 && "handle only 1 output ops");

  int64_t archVL = UnifiedStickSupport::archVL;
  int64_t stickLen = UnifiedStickSupport::stickLen;
  int64_t totVL = archVL * unrollVL;
  assert(stickLen % totVL == 0 && "bad unrollVL factor");
  IndexExpr litZero = LitIE(0);
  IndexExpr litStickLen = LitIE(stickLen);

  // Create loop iterations. We iterate over E1 as sticks of 64 elements. Lbs
  // and ubs reflect the iteration over the sticks (tiled data points).
  DimsExpr tiledLbs(rank, litZero);
  DimsExpr tiledUbs = outputDims;
  tiledUbs[d1] = E1.ceilDiv(litStickLen);

  // Parallel... Should not be turned on when parallelized in the outside.
  int64_t parId = 0;
  if (enableParallel) {
    // TODO: may want to check if ub of rank makes sense here.
    // Its ok here even to partition rank-1, included in (0..rank(, because
    // rank-1 is tiled. So we are still dealing with multiple of sticks.
    parId = tryCreateKrnlParallel(create.krnl, op,
        "compiler-generated stickify", {}, tiledLbs, tiledUbs, 0, rank, {},
        /*min iter for going parallel*/ 8, /*createKrnlParallel=*/false);
    if (parId == -1)
      enableParallel = false;
  }

  int64_t inputNum = op->getNumOperands();
  mlir::SmallVector<Value, 4> originalVals = op->getOperands();
  originalVals.emplace_back(op->getResult(0)); // Output is at index inputNum.
  mlir::SmallVector<Value, 4> originalMemRefs = operands;
  originalMemRefs.emplace_back(alloc); // Output is at index inputNum.
  mlir::BitVector isReads(inputNum + 1, true), isWrites(inputNum + 1, false);
  isReads[inputNum] = false; // Output is at index inputNum.
  isWrites[inputNum] = true; // Output is at index inputNum.
  UnifiedStickSupportList stickCS(create.krnl, originalVals, originalMemRefs,
      isReads, isWrites, disableSaturation);
  bool isStickifiedOutput = stickCS.list[inputNum].hasStick();

  // Predicates used to avoid creating code that is never used.
  bool neverHas64 = E1.isLiteralAndSmallerThan(stickLen);
  bool neverHas8 = E1.isLiteralAndSmallerThan(archVL);
  bool hasOnly64 = E1.isLiteral() && (E1.getLiteral() % stickLen == 0);
  bool hasOnly8 = E1.isLiteral() && (E1.getLiteral() % archVL == 0);

  if (STICK_OUTPUT_WRITE_PAST_BOUNDS && isStickifiedOutput) {
    // Output is stickified, we can write 8 values for the last iteration no
    // mater what, possibly over-writing values that we are not supposed to
    // write, but we know the memory exists as we always allocate a stick of
    // 64 values.
    neverHas8 = false; // Force at least one iteration into the 8-way simd loop.
    hasOnly8 = true;   // Skip the scalar loop with the custom buffer.
  }
  LLVM_DEBUG(
      llvm::dbgs() << "  Predicates: " << (neverHas64 ? "never-has-64 " : "")
                   << (neverHas8 ? "never-has-8 " : "")
                   << (hasOnly64 ? "has-only-64 " : "")
                   << (hasOnly8 ? "has-only-8\n" : "\n"));

  // Iterates over sticks.
  llvm::SmallVector<int64_t, 4> steps(rank, 1);
  llvm::SmallVector<bool, 4> useParallel(rank, false);
  if (enableParallel)
    useParallel[parId] = true;
  create.krnl.forLoopsIE(tiledLbs, tiledUbs, steps, useParallel,
      [&](const KrnlBuilder &b, mlir::ValueRange tiledLoopInd) {
        IndexExprScope outerScope(b);
        MDBuilder create(b);
        DimsExpr tiledOuterIndices = DimListIE(tiledLoopInd);
        // Computation for accessing data (not tiled, actual indices).
        DimsExpr outerIndices = tiledOuterIndices;
        IndexExpr E1 = DimIE(outputDims[d1]); // Original upper bound in d1.
        IndexExpr e1 = outerIndices[d1] = tiledOuterIndices[d1] * litStickLen;

        stickCS.beforeStickLoop(create.krnl, outerIndices);

        if (enablePrefetch) {
          // TODO: enable prefetch
          // Prefetch all in ioMemRefValues
          // create.krnl.prefetchIE(input, outerIndices, /*write*/ false,
          //    /*locality*/ 1);
        }
        // Check if we have a full stick (aka end of stick is not beyond UB).
        IndexExpr hasFullStick;
        if (hasOnly64) {
          hasFullStick = PredIE(true); // Has only full sicks.
        } else if (neverHas64) {
          hasFullStick = PredIE(false); // Doesn't even has 1 stick.
        } else {
          IndexExpr isFull = create.krnlIE.isTileFull(e1, litStickLen, E1);
          hasFullStick = (isFull >= 0);
        }
        create.scf.ifThenElse(
            hasFullStick.getValue(),
            // If is full, process all 64 values by iterating over totVL
            // values at a time.
            [&](const SCFBuilder b) {
              if (neverHas64)
                return; // Nothing to do here. Avoid generating dead code.
              MDBuilder create(b);
              // Iterate through stick by totVL (aka archVL==8 * unrollVL).
              create.scf.forLoopIE(litZero, litStickLen, totVL, /*par*/ false,
                  [&](const SCFBuilder b, mlir::ValueRange loopInd) {
                    IndexExprScope innerScope(b, &outerScope);
                    MDBuilder create(b);
                    IndexExpr offsetWithinStick = DimIE(loopInd[0]);
                    for (int64_t offsetWithinVector = 0;
                         offsetWithinVector < unrollVL; ++offsetWithinVector)
                      stickCS.loadComputeStore(create.krnl,
                          processVectorOfF32Vals, offsetWithinStick,
                          offsetWithinVector);
                  });
            },
            // Else, we don't have a full (64 e1) tile;
            [&](SCFBuilder b) {
              if (hasOnly64)
                return; // Do not generate dead code.
              MDBuilder create(b);
              IndexExprScope middleScope(b, &outerScope);
              IndexExpr tripCount = DimIE(E1) - DimIE(e1);
              if (!neverHas8) {
                // Not full 64, process archVL (8) values at a time instead of
                // TotVL. Note: if we only have multiple of archVL, loop below
                // will handle all archVL-full as we subtract (archVL-1). Aka
                // if VL=8 and tripCount = 16, tripCountSimdByVL is 16 - 7
                // = 9. Thus we iterate over i=0 (val 0..7) & i=8 (val 8..15)
                // as both are < 9.
                int64_t correction = archVL - 1;
                if (STICK_OUTPUT_WRITE_PAST_BOUNDS && isStickifiedOutput) {
                  // Overwrite is allowed, so if VL=8 and trip count = 16:
                  // will execute i=0 and i=8 (both full). But if trip count =
                  // 17, then will execute i=0 (val 0..7) & 8 (val 8..15), and
                  // i=16 (to compute/save the stick[x,16] single value, but
                  // overriding stick[x, 17..23] with garbage values).
                  correction = 0;
                }
                IndexExpr tripCountSimdByVL = tripCount - correction;
                create.scf.forLoopIE(litZero, tripCountSimdByVL, archVL,
                    /*par*/ false, [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      IndexExprScope innerScope(b, &middleScope);
                      MDBuilder create(b);
                      IndexExpr offsetWithinStick = DimIE(loopInd[0]);
                      stickCS.loadComputeStore(create.krnl,
                          processVectorOfF32Vals, offsetWithinStick,
                          /*no unroll here*/ 0);
                    });
              }
              if (!hasOnly8) {
                // Deal with the last < ArchVL=8 values: compute f32 using
                // simd. IndexExpr remainingScalarValues = tripCount % archVL;

                // Can use E1 instead of trip count as trip count substract
                // multiple of 64 to E1, and 64 % (archVal=8) = 0.
                IndexExpr remainingScalarValues = DimIE(E1) % archVL;
                IndexExpr lastL = tripCount - remainingScalarValues;
                IndexExpr innerIndexPlusLastL = DimIE(e1) + lastL;
                // Need a buffer to store partial results (less than 8) for
                // last iterations. Use a type of [1][8] so that it can
                // contain up to 7 partial results. Use a unit first dim to
                // match the rank of the reinterpreted casts.
                Type outputElementType =
                    getElementType(op->getResult(0).getType());
                MemRefType bufferType =
                    mlir::MemRefType::get({1, archVL}, outputElementType);
                Value outputBuffer = create.mem.alignedAlloc(bufferType);

                // Compute results and store into output buffer.
                // Buffer holds original or stickified (normalized) results
                // depending on the output type.
                stickCS.loadComputeStore(create.krnl, processVectorOfF32Vals,
                    lastL, /* unroll */ 0, outputBuffer);
                // Scalar store of buffer values.
                create.scf.forLoopIE(litZero, remainingScalarValues, 1,
                    /*par*/ false, [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      IndexExprScope innerScope(b, &middleScope);
                      MDBuilder create(b);
                      IndexExpr offsetWithinStick = DimIE(loopInd[0]);
                      Value bufferVal = create.krnl.loadIE(
                          outputBuffer, {litZero, offsetWithinStick});
                      // Even if stickified, we don't need simd store, and
                      // thus we can use the memref format without the view.
                      DimsExpr innerIndices = DimListIE(outerIndices);
                      innerIndices[d1] =
                          DimIE(innerIndexPlusLastL) + offsetWithinStick;
                      create.krnl.storeIE(bufferVal, alloc, innerIndices);
                    });
              }
            });
      });
}

// Check that all the input/outputs are float32 without ztensor, or dlf16 with
// tensor. Must have at least one dlf16 to return true.
static bool isZTensorOfF32AndDLF16(Operation *op) {
  bool hasDLF16 = false;
  for (Value val : op->getOperands()) {
    Type elementType = getElementType(val.getType());
    if (mlir::isa<NoneType>(elementType)) // Ignore none types.
      continue;
    if (zhigh::isZTensor(val.getType())) {
      if (!elementType.isF16()) {
        return false;
      }
      hasDLF16 = true;
    } else {
      if (!elementType.isF32()) {
        return false;
      }
    }
  }
  for (Value val : op->getResults()) {
    Type elementType = getElementType(val.getType());
    if (mlir::isa<NoneType>(elementType)) // Ignore none types.
      continue;
    if (zhigh::isZTensor(val.getType())) {
      if (!elementType.isF16()) {
        return false;
      }
      hasDLF16 = true;
    } else {
      if (!elementType.isF32()) {
        return false;
      }
    }
  }
  return hasDLF16;
}

// This use a wide range of distinct interfaces, keep it as is for the moment.
Value allocateTraditionalOrZtensor(ConversionPatternRewriter &rewriter,
    const TypeConverter *typeConverter, Operation *op, ValueRange operands,
    Value outputTensor, DimsExpr &dims, int64_t VL) {
  Type outputTensorType = outputTensor.getType();
  if (zhigh::isZTensor(outputTensorType)) {
    // Alloc for Z MemRefs
    zhigh::ZMemRefType zMemRefType =
        zhigh::convertZTensorToMemRefType(outputTensorType);
    // Allocate a buffer for the result MemRef.
    return zhigh::insertAllocForZMemRef(zMemRefType, dims, op, rewriter);
  }
  // Normal tensor.
  // Convert the output type to MemRefType.
  Type convertedType = typeConverter->convertType(outputTensorType);
  int64_t alignment =
      KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
  assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
         "Failed to convert type to MemRefType");
  MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
  // Insert an allocation and deallocation for the result of this
  // operation.
  MemRefBuilder b(rewriter, op->getLoc());
  return allocOrReuse(b, op, operands, outputMemRefType, dims, alignment, VL);
}

//===----------------------------------------------------------------------===//
// Elementwise patterns.

template <typename ElementwiseOp>
struct ONNXElementwiseOpLoweringWithNNPALayout
    : public OpConversionPattern<ElementwiseOp> {
  using OpAdaptor = typename ElementwiseOp::Adaptor;
  bool enableParallel = false;
  bool disableSaturation = false;

  ONNXElementwiseOpLoweringWithNNPALayout(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableParallel, bool disableSaturation)
      : OpConversionPattern<ElementwiseOp>(typeConverter, ctx,
            PatternBenefit(
                10)), // Benefit must be high so that we come here first.
        disableSaturation(disableSaturation) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    Location loc = ONNXLoc<ElementwiseOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Test if operation is suitable for processing here. If not, will be
    // handled by the normal elementwise operations.
    if (!isZTensorOfF32AndDLF16(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Reject elementwise op (" << op->getName()
                              << ") because no NNPA layout\n");
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Process elementwise op " << op->getName()
                   << " with NNPA layout:\n  ";
      op->dump();
    });
    assert(op->getNumResults() == 1 && "expect exactly 1 output");

    // Shape helper.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value alloc = allocateTraditionalOrZtensor(rewriter, this->typeConverter,
        op, operands, elmsOp.getResult(), shapeHelper.getOutputDims(),
        /*does not collapse and write past boundaries, ok to set VL=1 here*/ 1);

    UnifiedStickSupportList::IterateFctOver4xF32 fct =
        [&](const KrnlBuilder &b,
            mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals) {
          return emitScalarOpFor<ElementwiseOp>(rewriter, b.getLoc(), op,
              inputOfF32Vals[0].getType(), inputOfF32Vals);
        };
    // Unroll: can unroll up to 8 (for 8 * simd of 8 = 1 stick of 64.)
    int64_t unrollFactor = 8;
    IterateOverStickInputOutput(create.krnl, op, adaptor.getOperands(), alloc,
        shapeHelper.getOutputDims(), unrollFactor, enableParallel,
        disableSaturation, true /*prefetch*/, fct);

    // replace op.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generate code for (well behaving) Layer Norm, namely axis = -1, inner dim
// of static size and multiple of 64.

template <typename OP_TYPE, typename SHAPE_HELPER_TYPE>
struct FuzedStickUnstickGenericLayerNormaOpLowering
    : public OpConversionPattern<OP_TYPE> {
  FuzedStickUnstickGenericLayerNormaOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableParallel, bool disableSaturation)
      : OpConversionPattern<OP_TYPE>(typeConverter, ctx),
        disableSaturation(disableSaturation) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            OP_TYPE::getOperationName());
  }

  bool disableSaturation, enableParallel;

  using ADAPTOR_TYPE = typename OP_TYPE::Adaptor;
  using MDBuilder =
      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
          VectorBuilder, MemRefBuilder, AffineBuilderKrnlMem, SCFBuilder>;

  LogicalResult matchAndRewrite(OP_TYPE lnOp, ADAPTOR_TYPE adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Blocking. VL is for number of dlf16 in vector (8). B is for number of
    // parallel reductions.
    const int64_t archVL = UnifiedStickSupport::archVL;
    const int64_t stickLen = UnifiedStickSupport::stickLen;
    const int64_t B = 4;
    bool isTraditionalLayerNorm = false;
    if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value)
      isTraditionalLayerNorm = true;

    // Get generic info.
    Operation *op = lnOp.getOperation();
    Location loc = ONNXLoc<OP_TYPE>(op);
    ValueRange operands = adaptor.getOperands();
    Value xMemRef = adaptor.getX();
    MemRefType xMemRefType = mlir::cast<MemRefType>(xMemRef.getType());
    // Cannot rely on type of X or Y as they might be f16. Use here the type as
    // the computation type.
    Type elementType = rewriter.getF32Type();

    // Test if operation is suitable for processing here. If not, will be
    // handled by the normal elementwise operations.
    if (!isZTensorOfF32AndDLF16(op))
      return failure();

    int64_t XRank = xMemRefType.getRank();
    int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
    assert(XRank >= 2 && "expected 2+ for rank of X and Y");
    assert(axis == XRank - 1 && "fused Stick/Unstick/LN only with axis = -1");

    // Create builder and shape helper
    MDBuilder create(rewriter, loc);

    SHAPE_HELPER_TYPE shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    IndexExpr E1 = shapeHelper.getOutputDims(0)[XRank - 1];
    assert(E1.isLiteral() && E1.getLiteral() % stickLen == 0 &&
           "expected E1 mod 64 == 0");

    // Create Stick mem support for X.
    UnifiedStickSupport xUSS(create.krnl, lnOp.getX(), xMemRef,
        /*read only*/ true, false, disableSaturation);

    // Get other info: 1) epsilon as a scalar, 2) scale, and 3) optional bias.
    Value epsilon =
        create.math.constant(elementType, lnOp.getEpsilon().convertToDouble());
    UnifiedStickSupport scaleUSS(create.krnl, lnOp.getScale(),
        adaptor.getScale(), /*read only*/ true, false, disableSaturation);
    UnifiedStickSupport biasUSS;
    // TODO: current additional ONNX op ONNXRMSLayerNormalizationOp has bias;
    // but in opset 24, RMSNormalization is introduced without biased. We
    // should remove the additional version and remove it below too.
    if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value ||
                  std::is_same<OP_TYPE, ONNXRMSLayerNormalizationOp>::value) {
      // Handle optional bias.
      if (!isNoneValue(lnOp.getB()))
        biasUSS.init(create.krnl, lnOp.getB(), adaptor.getB(),
            /*read only*/ true, false, disableSaturation);
    }

    // Allocate output: convert and allocate
    Value yMemRef = allocateTraditionalOrZtensor(rewriter, this->typeConverter,
        op, operands, lnOp.getY(), shapeHelper.getOutputDims(0),
        /*does not collapse and write past boundaries, ok to set VL=1 here*/ 1);
    UnifiedStickSupport yUSS(create.krnl, lnOp.getY(), yMemRef,
        /*write only*/ false, true, disableSaturation);

    // This pass does not support mean or inv std dev.
    if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value)
      assert(isNoneValue(lnOp.getMean()) &&
             "Mean not supported in fused Stick/Unstick/LN");
    if constexpr (std::is_same<OP_TYPE, ONNXLayerNormalizationOp>::value ||
                  std::is_same<OP_TYPE, ONNXRMSLayerNormalizationOp>::value)
      assert(isNoneValue(lnOp.getInvStdDev()) &&
             "InvStdDev not supported in fused Stick/Unstick/LN");

    // Outerloops (all but E1), with blocked E2 blocked by B. No E1 in lbs/ubs.
    DimsExpr ubs(shapeHelper.getOutputDims(0));
    ubs.pop_back();
    DimsExpr lbs(XRank - 1, LitIE(0));
    ValueRange loopDefs = create.krnl.defineLoops(XRank - 1);
    SmallVector<Value, 4> outerOptLoops, innerOptLoops;
    create.krnl.blockAndPermute(loopDefs, {B}, outerOptLoops, innerOptLoops);
    // Handle Parallel
    bool useParallel = false;
    if (enableParallel) {
      // Parallelize one loop from 0 to (exclusively) min(2, outer loop nums).
      int parRank = ubs.size();
      if (parRank > 2)
        parRank = 2;
      SmallVector<IndexExpr, 2> parLbs(parRank, LitIE(0));
      SmallVector<IndexExpr, 2> parUbs =
          firstFew<IndexExpr, 2>(ubs, parRank - 1 /*inclusive*/);
      if (tryCreateKrnlParallel(create.krnl, op, "layer-norm", outerOptLoops,
              parLbs, parUbs, 0, parRank, {}, 4,
              /*createKrnlParallel=*/true) != -1)
        useParallel = true;
    }

    // Temp reduction buffers
    MemRefType redType = MemRefType::get({B, archVL}, elementType);
    Value redMemRef1 = nullptr, redMemRef2 = nullptr;
    if (!useParallel) {
      // Sequential, alloc before loop.
      if (isTraditionalLayerNorm)
        redMemRef1 = create.mem.alignedAlloc(redType);
      redMemRef2 = create.mem.alignedAlloc(redType);
    }

    create.krnl.iterateIE(loopDefs, outerOptLoops, lbs, ubs,
        [&](const KrnlBuilder &ck, ValueRange outerLoopInd) {
          MDBuilder create(ck);
          IndexExprScope middleScope(ck);
          DimsExpr outerIndices = DimListIE(outerLoopInd); // no e1.
          int64_t d2 = outerIndices.size() - 1;
          if (useParallel) {
            // Parallel, alloc inside parallel loop.
            if (isTraditionalLayerNorm)
              redMemRef1 = create.mem.alignedAlloc(redType);
            redMemRef2 = create.mem.alignedAlloc(redType);
          }

          // Determine full tile.
          IndexExpr blockedCurrIndex = DimIE(outerIndices[d2]);
          IndexExpr blockedUB = DimIE(ubs[d2]);
          IndexExpr isFull =
              create.krnlIE.isTileFull(blockedCurrIndex, LitIE(B), blockedUB);
          Value zero = create.math.constantIndex(0);
          Value isFullVal = create.math.ge(isFull.getValue(), zero);
          create.scf.ifThenElse(
              isFullVal,
              [&](const SCFBuilder &scf) {
                MDBuilder create(scf);
                IndexExprScope innerScopes(scf, &middleScope);
                // create.krnl.printf("full tile\n");
                //  Compute a full tile of B
                generateIter<B>(create, lnOp, elementType,
                    isTraditionalLayerNorm, xUSS, biasUSS, scaleUSS, redMemRef1,
                    redMemRef2, yUSS, outerIndices, E1, epsilon, archVL,
                    stickLen);
              },
              [&](const SCFBuilder &scf) {
                MDBuilder create(scf);
                IndexExprScope innerScope(scf, &middleScope);
                // create.krnl.printf("partial tile\n");
                Value startOfLastBlockVal = blockedCurrIndex.getValue();
                Value blockedUBVal = blockedUB.getValue();
                // Iterate over the last few values of b.
                create.scf.forLoop(startOfLastBlockVal, blockedUBVal, 1,
                    [&](const SCFBuilder &scf, ValueRange loopInd) {
                      IndexExprScope innermostScope(scf, &innerScope);
                      MDBuilder create(scf);
                      // Reflect b inot current indices.
                      DimsExpr currOuterIndices = DimListIE(outerIndices);
                      currOuterIndices[d2] = DimIE(loopInd[0]);
                      generateIter<1>(create, lnOp, elementType,
                          isTraditionalLayerNorm, xUSS, biasUSS, scaleUSS,
                          redMemRef1, redMemRef2, yUSS, currOuterIndices, E1,
                          epsilon, archVL, stickLen);
                    }); // Last values of b.
              });       // If full then else.
        });             // Blocked outer loop.

    // Replace the op (here only if we have a single output.)
    Value noneValue;
    llvm::SmallVector<Value, 3> outputs;
    if (isTraditionalLayerNorm)
      outputs = {yMemRef, noneValue, noneValue};
    else
      outputs = {yMemRef, noneValue};
    rewriter.replaceOp(lnOp, outputs);
    return success();
  }

  template <int64_t B>
  void generateIter(MDBuilder &create, OP_TYPE lnOp, Type elementType,
      bool isTraditionalLayerNorm, UnifiedStickSupport &xUSS,
      UnifiedStickSupport &biasUSS, UnifiedStickSupport &scaleUSS,
      Value redMemRef1, Value redMemRef2, UnifiedStickSupport &yUSS,
      /* index expr param */ DimsExpr outerLoopIndices, IndexExpr E1,
      /* value params */ Value epsilon,
      /* int params */ int64_t archVL, int64_t stickLen) const {

    // Init the reductions, compute subviews for [1, archVL] views, init
    // USS.
    VectorType vecType = VectorType::get({archVL}, elementType);
    Value init = create.math.constant(elementType, 0.0);
    Value initVec = create.vec.splat(vecType, init);
    Value zero = create.math.constantIndex(0);
    UnifiedStickSupportList blockedMeanUSSList[B];
    int64_t xRank = getRank(xUSS.getOriginalVal().getType());
    inlineFor(create, B, [&](int64_t b, Value bb) {
      // Init tmpRed1 as second parameter.
      UnifiedStickSupport redUSS1;
      if (isTraditionalLayerNorm) {
        create.vec.store(initVec, redMemRef1, {bb, zero});
        Value redSubview1 =
            create.mem.subview(redMemRef1, {b, 0}, {1, archVL}, {1, 1});
        redUSS1.init(create.krnl, redSubview1, redSubview1, /*read/write*/ true,
            true, disableSaturation);
      }
      // Init tmpRed2 as third parameter.
      create.vec.store(initVec, redMemRef2, {bb, zero});
      Value redSubview2 =
          create.mem.subview(redMemRef2, {b, 0}, {1, archVL}, {1, 1});
      UnifiedStickSupport redUSS2(create.krnl, redSubview2, redSubview2,
          /*read/write*/ true, true, disableSaturation);
      // Set list.
      blockedMeanUSSList[b].list = {xUSS, redUSS1, redUSS2};
    });

    // Compute parallel reductions to compute red1 and red2, iterating over
    // each chunk of 8 values for B sticks at a time.
    IndexExpr lit0 = LitIE(0);
    IndexExpr litStickLen = LitIE(stickLen);

    // Iterate over sticks.
    create.affineKMem.forLoopIE(lit0, E1, stickLen,
        [&](const onnx_mlir::AffineBuilderKrnlMem &ck, ValueRange loopInd) {
          MDBuilder create(ck);
          IndexExprScope outerScope(ck);

          IndexExpr e1 = DimIE(loopInd[0]);

          // Init before stick loop with index reflecting the b-blocking factor
          // (2nd to last innermost dim).
          for (int64_t b = 0; b < B; ++b) {
            DimsExpr loopIndices = DimListIE(outerLoopIndices);
            loopIndices.emplace_back(e1); // Add E1 dim.
            assert(((int64_t)loopIndices.size()) == xRank && "size mismatch");
            loopIndices[xRank - 2] = loopIndices[xRank - 2] + b; // Account for
            blockedMeanUSSList[b].beforeStickLoop(create.krnl, loopIndices);
          }

          // Iterate within the stick
          create.affineKMem.forLoopIE(lit0, litStickLen, archVL,
              [&](const onnx_mlir::AffineBuilderKrnlMem &ck,
                  ValueRange loopInd) {
                MDBuilder create(ck);
                IndexExprScope innerScope(ck, &outerScope);

                IndexExpr offsetWithinStick = DimIE(loopInd[0]);
                // load X, compute X**2, sum into reductions for a vector of
                // ArchVL.
                for (int64_t b = 0; b < B; ++b) {
                  // Define function to apply to the inputs.
                  UnifiedStickSupportList::GenericIterateFctOver4xF32M fct =
                      [&](const KrnlBuilder &kb,
                          mlir::SmallVectorImpl<mlir::Value> &listOfF32Vals) {
                        MDBuilder create(ck);
                        Value x = listOfF32Vals[0];     // Input.
                        Value &red1 = listOfF32Vals[1]; // Input and output.
                        Value &red2 = listOfF32Vals[2]; // Input and output.
                        // Compute x^2.
                        Value xSquare = create.math.mul(x, x);
                        // Perform reduction on x values.
                        if (isTraditionalLayerNorm)
                          red1 = create.math.add(red1, x);
                        // Perform reduction of x^2 values.
                        red2 = create.math.add(red2, xSquare);
                      };
                  blockedMeanUSSList[b].genericLoadComputeStore(
                      create.krnl, fct, offsetWithinStick, 0);
                } // over each b
              }); // over one stick archVL
        });       // over each sticks by stickLen

    // Sum across, compute mean, var, standard deviation and its inverse.
    llvm::SmallVector<Value, 4> mean(B), invStdDev(B);
    Value redDimFloat = create.math.cast(elementType, E1.getValue());
    Value oneFloat = create.math.constant(elementType, 1.0);
    inlineFor(create, B, [&](int64_t b, Value bb) {
      Value finalRed1, finalRed2, currSum1, currSum2, mean2, meanSquare, var;
      // Load reductions.
      if (isTraditionalLayerNorm)
        finalRed1 = create.vec.load(vecType, redMemRef1, {bb, zero});
      finalRed2 = create.vec.load(vecType, redMemRef2, {bb, zero});
      // Horizontal reductions.
      if (isTraditionalLayerNorm)
        currSum1 =
            create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed1);
      currSum2 =
          create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed2);
      // Compute means.
      if (isTraditionalLayerNorm)
        mean[b] = create.math.div(currSum1, redDimFloat);
      mean2 = create.math.div(currSum2, redDimFloat);
      // Compute standard deviation (with epsilon) and its inverse.
      if (isTraditionalLayerNorm) {
        meanSquare = create.math.mul(mean[b], mean[b]);
        var = create.math.sub(mean2, meanSquare);
      } else {
        var = mean2;
      }
      Value varEps = create.math.add(var, epsilon);
      Value stdDev = create.math.sqrt(varEps);
      invStdDev[b] = create.math.div(oneFloat, stdDev);
    });

    // Normalize of entire vectors.
    UnifiedStickSupportList blockedNormUSSList[B];
    for (int64_t b = 0; b < B; ++b)
      blockedNormUSSList[b].list = {xUSS, scaleUSS, biasUSS, yUSS};

    // Iterates over sticks
    create.affineKMem.forLoopIE(lit0, E1, stickLen,
        [&](const onnx_mlir::AffineBuilderKrnlMem &ck, ValueRange loopInd) {
          MDBuilder create(ck);
          IndexExprScope outerScope(ck);
          IndexExpr e1 = DimIE(loopInd[0]);

          // Init before stick loop with index reflecting the b-blocking factor
          // (2nd to last innermost dim).
          for (int64_t b = 0; b < B; ++b) {
            DimsExpr loopIndices = DimListIE(outerLoopIndices);
            loopIndices.emplace_back(e1);
            loopIndices[xRank - 2] = loopIndices[xRank - 2] + b;
            blockedNormUSSList[b].beforeStickLoop(create.krnl, loopIndices);
          }

          // Iterate within the stick
          create.affineKMem.forLoopIE(lit0, litStickLen, archVL,
              [&](const onnx_mlir::AffineBuilderKrnlMem &ck,
                  ValueRange loopInd) {
                MDBuilder create(ck);
                IndexExprScope innerScope(ck, &outerScope);

                IndexExpr offsetWithinStick = DimIE(loopInd[0]);
                for (int64_t b = 0; b < B; ++b) {
                  // Function to apply to the inputs.
                  UnifiedStickSupportList::GenericIterateFctOver4xF32M fct =
                      [&](const KrnlBuilder &kb,
                          mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals) {
                        MDBuilder create(ck);
                        Value x = inputOfF32Vals[0];     // Input.
                        Value scale = inputOfF32Vals[1]; // Input.
                        Value bias = inputOfF32Vals[2];  // Input.
                        Value &y = inputOfF32Vals[3];    // Output.
                        Value XMinusMean;
                        if (isTraditionalLayerNorm)
                          XMinusMean = create.math.sub(x, mean[b]);
                        else
                          XMinusMean = x;
                        Value normalizedX =
                            create.math.mul(XMinusMean, invStdDev[b]);
                        // Process with multiplying by scale (scalar or 1D
                        // vector).
                        y = create.math.mul(normalizedX, scale);
                        if (bias)
                          y = create.math.add(y, bias);
                      };
                  blockedNormUSSList[b].genericLoadComputeStore(
                      create.krnl, fct, offsetWithinStick, 0);
                } // over each b
              }); // over one stick by archVL
        });       // over all sticks by stickLen
  }

  using F1 = std::function<void(int64_t offsetInt, Value offsetVal)>;
  void inlineFor(MDBuilder &create, int64_t B, F1 genCode) const {
    for (int64_t offsetInt = 0; offsetInt < B; ++offsetInt) {
      Value offsetVal = create.math.constantIndex(offsetInt);
      genCode(offsetInt, offsetVal);
    }
  }
};

using ONNXFuzedStickUnstickLayerNormalizationOpLowering =
    FuzedStickUnstickGenericLayerNormaOpLowering<ONNXLayerNormalizationOp,
        ONNXLayerNormalizationOpShapeHelper>;
using ONNXFuzedStickUnstickRMSLayerNormalizationOpLowering =
    FuzedStickUnstickGenericLayerNormaOpLowering<ONNXRMSLayerNormalizationOp,
        ONNXRMSLayerNormalizationOpShapeHelper>;

//===----------------------------------------------------------------------===//
// Pass
namespace zhigh {
void populateONNXWithNNPALayoutToKrnlConversionPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    bool enableParallel, bool disableSaturation) {
// Add the insert patterns for all elementwise types regardless of unary, binary
// or variadic.
#define ELEMENTWISE_ALL(_OP_TYPE)                                              \
  patterns.insert<ONNXElementwiseOpLoweringWithNNPALayout<_OP_TYPE>>(          \
      typeConverter, ctx, enableParallel, disableSaturation);
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"

  // Add normalization patterns.
  patterns.insert<ONNXFuzedStickUnstickLayerNormalizationOpLowering>(
      typeConverter, ctx, enableParallel, disableSaturation);
  patterns.insert<ONNXFuzedStickUnstickRMSLayerNormalizationOpLowering>(
      typeConverter, ctx, enableParallel, disableSaturation);
}

} // namespace zhigh
} // namespace onnx_mlir
