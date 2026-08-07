/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ FusedSplitOpGather.cpp - Lower simd-split-op-gather ----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Lowers onnx.Fused(kind="simd-split-op-gather") directly to one or two
// simdIterateIE calls that read/write disjoint offset ranges of the same
// buffers -- instead of materializing the two Slice outputs and the Concat
// output as three separate allocations, which is what happens when this
// kind has no dedicated lowering and FusedOpInlineFallback inlines it back
// to plain Slice.cpp/Concat.cpp/elementwise codegen.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Dispatch to the same per-op scalar/SIMD codegen the regular elementwise
// lowering uses (emitScalarOpFor<T>), over the exact same canonical
// op-type list SplitOpGatherFusionHelper's allow-list check pulls in (see
// ONNXFusionOpHelper.cpp) -- keeps detection and lowering permanently in
// sync: adding an op to Elementwise.hpp's list makes it both matchable and
// lowerable with no second list to touch.
Value emitPerHalfOp(ConversionPatternRewriter &rewriter, Location loc,
    Operation *opNode, Type elemType, ArrayRef<Value> scalarOperands) {
#define ELEMENTWISE_ALL(_OP_TYPE)                                             \
  if (isa<_OP_TYPE>(opNode))                                                  \
    return emitScalarOpFor<_OP_TYPE>(                                         \
        rewriter, loc, opNode, elemType, scalarOperands);
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"
  llvm_unreachable(
      "op type not in allow-list; verify() should have rejected it");
}

} // namespace

struct ONNXFusedSplitOpGatherLowering
    : public FusedOpKindLowering<SplitOpGatherFusionHelper> {
  using Base = FusedOpKindLowering<SplitOpGatherFusionHelper>;
  bool enableSIMD;

  ONNXFusedSplitOpGatherLowering(
      TypeConverter &tc, MLIRContext *ctx, bool enableSIMD)
      : Base(tc, ctx), enableSIMD(enableSIMD) {}

  FailureOr<SmallVector<Value>> lowerVerified(ONNXFusedOp fusedOp,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      SplitOpGatherFusionHelper &fusion) const override {
    Location loc = fusedOp.getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // ---- Reconstruct the fixed op order & locate external operands. ------
    // See ONNXFusionOpHelper.hpp's "Chain-op order convention": ops is
    // always [sliceLow, opLow?, sliceHigh, opHigh?, concatOp]; the accessors
    // below encode that same layout (the Concat itself isn't needed here).
    ONNXSliceOp sliceLowOp = fusion.getSliceLowOp();
    Operation *opLowNode = fusion.getOpLowNode();
    ONNXSliceOp sliceHighOp = fusion.getSliceHighOp();
    Operation *opHighNode = fusion.getOpHighNode();

    // adaptor.getInputs()[0] is always the shared source tensor: sliceLowOp
    // is always ops[0], and its `data` operand is the first external value
    // FusionOpKindHelper::computeInputsAndInsertionPoint()'s collectExternals
    // walk ever encounters. Each per-half op's external operand (if binary),
    // when present, follows in ops order -- low's before high's.
    Value dataMemref = adaptor.getInputs()[0];
    size_t inputIdx = 1;
    Value externalLowMemref, externalHighMemref;
    bool lowHalfIsFirstOperand = true, highHalfIsFirstOperand = true;
    if (opLowNode && opLowNode->getNumOperands() == 2) {
      externalLowMemref = adaptor.getInputs()[inputIdx++];
      lowHalfIsFirstOperand =
          (opLowNode->getOperand(0) == sliceLowOp.getResult());
    }
    if (opHighNode && opHighNode->getNumOperands() == 2) {
      externalHighMemref = adaptor.getInputs()[inputIdx++];
      highHalfIsFirstOperand =
          (opHighNode->getOperand(0) == sliceHighOp.getResult());
    }

    // ---- Shapes. -----------------------------------------------------------
    MemRefType dataMemRefType = cast<MemRefType>(dataMemref.getType());
    int64_t rank = dataMemRefType.getRank();
    int64_t axis = fusion.axis;
    Type elemType = dataMemRefType.getElementType();
    int64_t D = dataMemRefType.getShape()[axis];
    // Should always be static: Part 1 unconditionally requires this at
    // detection time. Defensive re-check; falls through to
    // FusedOpInlineFallback via pattern competition if it somehow isn't.
    if (D == ShapedType::kDynamic)
      return failure();
    int64_t lenLow = fusion.splitPoint;
    int64_t lenHigh = D - fusion.splitPoint;

    // ---- Allocate the output. -----------------------------------------------
    // Output shares every dim with `data`, including the axis dim's total
    // size -- only its internal partitioning changes.
    Type outputTensorType = fusedOp.getOutputs()[0].getType();
    Type convertedType = typeConverter->convertType(outputTensorType);
    MemRefType outputMemRefType = cast<MemRefType>(convertedType);
    DimsExpr outputDims;
    for (int64_t d = 0; d < rank; ++d)
      outputDims.emplace_back(create.krnlIE.getShapeAsDim(dataMemref, d));
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value outputMemref =
        create.mem.alignedAlloc(outputMemRefType, outputDims, alignment);

    // ---- Emit one half's simdIterateIE call, given the outer (non-axis)
    // loop induction variables (empty when rank==1). ------------------------
    auto lowerHalf = [&](ValueRange outerInd, int64_t start, int64_t len,
                          int64_t outputOffset, bool hasOp, Operation *opNode,
                          Value externalMemref, bool halfIsFirstOperand) {
      int64_t VL = enableSIMD
          ? VectorMachineSupport::getArchVectorLength(elemType)
          : 1;
      bool fullySimd = (len % VL == 0);

      DimsExpr dataAF, outAF;
      for (Value v : outerInd) {
        dataAF.emplace_back(DimIE(v));
        outAF.emplace_back(DimIE(v));
      }
      dataAF.emplace_back(LitIE(start));
      outAF.emplace_back(LitIE(outputOffset));

      SmallVector<Value, 2> inputs{dataMemref};
      SmallVector<DimsExpr, 2> inputAFs{dataAF};
      if (externalMemref) {
        // Already sized to exactly `len`; indexed from its own 0, not
        // offset by `start` (that offset is specific to the shared `data`
        // buffer, not to this already-half-sized external operand).
        DimsExpr extAF;
        for (Value v : outerInd)
          extAF.emplace_back(DimIE(v));
        extAF.emplace_back(LitIE(0));
        inputs.push_back(externalMemref);
        inputAFs.push_back(extAF);
      }

      KrnlBuilder::KrnlSimdIterateBodyFn bodyFn =
          [&](const KrnlBuilder &, ArrayRef<Value> inputVals,
              int64_t currVL) -> Value {
        if (!hasOp)
          return inputVals[0]; // zero op == copy.
        SmallVector<Value, 2> scalarOperands;
        if (externalMemref)
          scalarOperands = halfIsFirstOperand
              ? SmallVector<Value, 2>{inputVals[0], inputVals[1]}
              : SmallVector<Value, 2>{inputVals[1], inputVals[0]};
        else
          scalarOperands = {inputVals[0]};
        // emitScalarOpFor uses this type as the RESULT type of the op it
        // builds (not just to type-dispatch), so it must reflect the
        // *current* invocation's mode -- SIMD (currVL > 1, e.g. the main
        // loop) or scalar (currVL == 1, e.g. simdIterateIE's own remainder
        // loop) -- not the outer, requested VL. Mirrors Elementwise.cpp's
        // own SIMD body construction of `currElementType` exactly.
        Type currElemType =
            currVL > 1 ? VectorType::get({currVL}, elemType) : elemType;
        return emitPerHalfOp(rewriter, loc, opNode, currElemType, scalarOperands);
      };

      create.krnl.simdIterateIE(LitIE(0), LitIE(len), VL, fullySimd,
          /*useParallel=*/false, inputs, inputAFs, {outputMemref}, {outAF},
          {bodyFn});
    };

    if (rank > 1) {
      ValueRange loopDef = create.krnl.defineLoops(rank - 1);
      SmallVector<IndexExpr, 4> lbs(rank - 1, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      for (int64_t d = 0; d < rank - 1; ++d)
        ubs.emplace_back(create.krnlIE.getShapeAsDim(dataMemref, d));
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &, ValueRange outerInd) {
            lowerHalf(outerInd, /*start=*/0, lenLow,
                fusion.outputOffsetForSplitLow, fusion.hasOpForSplitLow,
                opLowNode, externalLowMemref, lowHalfIsFirstOperand);
            lowerHalf(outerInd, /*start=*/fusion.splitPoint, lenHigh,
                fusion.outputOffsetForSplitHigh, fusion.hasOpForSplitHigh,
                opHighNode, externalHighMemref, highHalfIsFirstOperand);
          });
    } else {
      lowerHalf({}, /*start=*/0, lenLow, fusion.outputOffsetForSplitLow,
          fusion.hasOpForSplitLow, opLowNode, externalLowMemref,
          lowHalfIsFirstOperand);
      lowerHalf({}, /*start=*/fusion.splitPoint, lenHigh,
          fusion.outputOffsetForSplitHigh, fusion.hasOpForSplitHigh,
          opHighNode, externalHighMemref, highHalfIsFirstOperand);
    }

    return SmallVector<Value>{outputMemref};
  }
};

void populateLoweringONNXFusedSplitOpGatherOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<ONNXFusedSplitOpGatherLowering>(
      typeConverter, ctx, enableSIMD);
}

} // namespace onnx_mlir
