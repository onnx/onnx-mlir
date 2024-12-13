
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutTransform.cpp - Lowering Layout Transform Op --------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Layout Transform Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "layout-transform"
#define PREFETCH_ON 0 /* did not seem to help at this stage */

using namespace mlir;

namespace onnx_mlir {

struct ONNXLayoutTransformOpLowering
    : public OpConversionPattern<ONNXLayoutTransformOp> {
  bool enableParallel = false;

  using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
      MathBuilder, MemRefBuilder, VectorBuilder, AffineBuilder, SCFBuilder>;

  ONNXLayoutTransformOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXLayoutTransformOp::getOperationName());
  }

  // Look for layout pattern of the type "dx" or "dx mod c" where dx is the last
  // dim of the map and c is a constant literal. Return true on success with
  // mod set to the proper value (-1 if no modulo).
  bool inspectMappedLowestDim(MemRefType type, int64_t &modVal) const {
    modVal = -1; // Set to undefined value.
    AffineMap affineMap = type.getLayout().getAffineMap();
    int64_t numDims = affineMap.getNumDims();
    int64_t numResults = affineMap.getNumResults();
    AffineExpr innerAffineExpr = affineMap.getResult(numResults - 1);
    LLVM_DEBUG({
      llvm::dbgs() << "Investigate Layout transform\n";
      affineMap.dump();
    });

    // Check if we have a "d2" pattern.
    if (innerAffineExpr.getKind() == AffineExprKind::DimId) {
      AffineDimExpr dimExpr = mlir::cast<AffineDimExpr>(innerAffineExpr);
      int64_t dimId = dimExpr.getPosition();
      if (dimId != numDims - 1)
        return false;
      LLVM_DEBUG(llvm::dbgs() << "  found d" << dimId << "\n");
      return true;
    }
    // Check if we have a "d2 mod 64" pattern.
    if (innerAffineExpr.getKind() == AffineExprKind::Mod) {
      AffineBinaryOpExpr modExpr =
          mlir::cast<AffineBinaryOpExpr>(innerAffineExpr);
      // Expect dim on the LHS.
      AffineExpr expectedDimExpr = modExpr.getLHS();
      if (expectedDimExpr.getKind() != AffineExprKind::DimId)
        return false;
      AffineDimExpr dimExpr = mlir::cast<AffineDimExpr>(expectedDimExpr);
      int64_t dimId = dimExpr.getPosition();
      if (dimId != numDims - 1)
        return false;
      // Expect literal on the RHS.
      AffineExpr expectedConstExpr = modExpr.getRHS();
      if (expectedConstExpr.getKind() != AffineExprKind::Constant)
        return false;
      AffineConstantExpr valExpr =
          mlir::cast<AffineConstantExpr>(expectedConstExpr);
      modVal = valExpr.getValue();
      LLVM_DEBUG(
          llvm::dbgs() << "  found d" << dimId << " mod " << modVal << "\n");
      return modVal > 0;
    }
    LLVM_DEBUG(llvm::dbgs() << "  did not find d2 or d2 mod 64 type pattern\n");
    return false;
  }

  // If both input and output have defined modVals (!=-1), then they must have
  // the save value.
  LogicalResult generateLayoutWithMod(ConversionPatternRewriter &rewriter,
      MDBuilder create, Operation *op, Value alloc, Value input,
      SmallVector<IndexExpr, 4> &lbs, SmallVector<IndexExpr, 4> &ubs,
      int64_t inModVal, int64_t outModVal) const {
    int64_t rank = lbs.size();
    assert((inModVal == -1 || outModVal == -1 || inModVal == outModVal) &&
           "bad mods");
    LLVM_DEBUG(llvm::dbgs() << "use fast pattern with in/out mod vals"
                            << inModVal << "/" << outModVal << "\n");
    int64_t modVal = (inModVal > outModVal) ? inModVal : outModVal;
    // Create loop iterations. Note that we iterate over E1 as tiles of modVal
    // elements.
    ValueRange loopDefs = create.krnl.defineLoops(rank);
    int64_t E1 = rank - 1; // Innermost dim is referred to E1 here.
    IndexExpr ub1 = ubs[E1];
    IndexExpr T1 = ub1.ceilDiv(modVal);
    ubs[E1] = T1;

    // Parallel...
    if (enableParallel) {
      int64_t parId;
      // TODO: may want to check if ub of rank makes sense here.
      if (findSuitableParallelDimension(lbs, ubs, 0, rank, parId, 8)) {
        create.krnl.parallel(loopDefs[parId]);
        onnxToKrnlParallelReport(op, true, parId, lbs[parId], ubs[parId],
            "layout transform fast pattern");
      } else {
        onnxToKrnlParallelReport(op, false, -1, -1,
            "no dim with enough work in layout transform fast pattern");
      }
    }

    //  Outer loop (E1 iterates over tiles of 64 elements).
    create.krnl.iterateIE(loopDefs, loopDefs, lbs, ubs,
        [&](const KrnlBuilder &b, ValueRange loopInd) {
          MDBuilder create(b);
          IndexExprScope outerScope(create.krnl);
          DimsExpr outerIndices;
          getIndexExprList<SymbolIndexExpr>(loopInd, outerIndices);
          DimsExpr memAF = outerIndices;
          memAF[E1] =
              memAF[E1] * modVal; // Loop index for E1 is in tiles of modVal.
          Value allocOffset = create.krnl.getLinearOffsetIndexIE(alloc, memAF);
          Value inputOffset = create.krnl.getLinearOffsetIndexIE(input, memAF);
          Value len = create.math.constant(rewriter.getI64Type(), modVal);
#if PREFETCH_ON
          DimsExpr prefetchAF = memAF;
          // Prefetch current line
          create.krnl.prefetchIE(input, prefetchAF, /*isWrite*/ false,
              /*locality*/ 1);
          create.krnl.prefetchIE(alloc, prefetchAF, /*isWrite*/ true,
              /*locality*/ 1);
#endif
          // Now if we copy into a modVal, and ub1 was not a multiple of modVal,
          // we may read and write a few values that should not be read/written
          // but since the output data physical memory will have the extra space
          // for it, we simply write it all.
          if (outModVal != -1) {
            create.krnl.memcpy(alloc, input, len, allocOffset, inputOffset);
          } else {
            // Compute if we have a last tile.
            IndexExpr modLit = LitIE(modVal);
            IndexExpr isFull =
                create.krnlIE.isTileFull(memAF[E1], modLit, SymIE(ub1));
            IndexExpr isFullLogical = isFull >= 0;
            create.scf.ifThenElse(
                // Condition
                isFullLogical.getValue(),
                // Then (is full).
                [&](const SCFBuilder b) {
                  MDBuilder create(b);
                  create.krnl.memcpy(
                      alloc, input, len, allocOffset, inputOffset);
                },
                // Else, we don't have a full tile.
                [&](const SCFBuilder b) {
                  MDBuilder create(b);
                  IndexExprScope middleScope(b, &outerScope);
                  IndexExpr tripCount = SymIE(ub1) - SymIE(memAF[E1]);
                  Value len = create.math.cast(
                      rewriter.getI64Type(), tripCount.getValue());
                  create.krnl.memcpy(
                      alloc, input, len, allocOffset, inputOffset);
                });
          }
        });
    rewriter.replaceOp(op, alloc);
    return success();
  }

  LogicalResult matchAndRewrite(ONNXLayoutTransformOp layoutOp,
      ONNXLayoutTransformOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = layoutOp.getOperation();
    Location loc = ONNXLoc<ONNXLayoutTransformOp>(op);

    // Operands and attributes.
    Value data = adaptor.getData();

    // Convert the input type to MemRefType.
    Type inConvertedType = typeConverter->convertType(data.getType());
    assert(inConvertedType && mlir::isa<MemRefType>(inConvertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType inMemRefType = mlir::cast<MemRefType>(inConvertedType);
    // Convert the output type to MemRefType.
    Type outputTensorType = *op->result_type_begin();
    Type outConvertedType = typeConverter->convertType(outputTensorType);
    assert(outConvertedType && mlir::isa<MemRefType>(outConvertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = mlir::cast<MemRefType>(outConvertedType);

    // Note that by definition the input and output of LayoutTransformOp have
    // the same logical rank. The only difference between them should be their
    // layout. Currently defined layout may increase the dimensionality of the
    // mapped data by 1 or 2 dimensions compared to the original layout. Note
    // that these higher dimensionality will only manifest themselves once the
    // memref are normalized.
    uint64_t rank = inMemRefType.getShape().size();

    // Transform simply copy the input data to the output data. Both must have
    // the same logical size so use the input ones (arbitrary).
    MDBuilder create(rewriter, loc);
    IndexExprScope outerScope(create.krnl);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(data, ubs);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));

    // Insert an allocation and deallocation for the result of this
    // operation.
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    Value alloc = create.mem.alignedAlloc(outMemRefType, ubs, alignment);

    // Inspect input and output layout to see if we can optimize the
    // transformation.
    int64_t inMod, outMod;
    bool inValid = inspectMappedLowestDim(inMemRefType, inMod);
    bool outValid = inspectMappedLowestDim(outMemRefType, outMod);
    if (inValid && outValid && rank >= 2) {
      // For the moment, support only a mod in the one or the other direction.
      if ((inMod == -1 && outMod >= 16) || (inMod >= 16 && outMod == -1)) {
        return generateLayoutWithMod(
            rewriter, create, op, alloc, data, lbs, ubs, inMod, outMod);
      }
      if (inMod == outMod && inMod != -1) {
        // We have 2 identical mods, do it too.
        // TODO: this scenario may need to be tested thoroughly once we generate
        // this pattern.
        return generateLayoutWithMod(
            rewriter, create, op, alloc, data, lbs, ubs, inMod, outMod);
      }
    }

    // Insert loop over all inputs.
    ValueRange loopDef = create.krnl.defineLoops(rank);

    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(lbs, ubs, 0, 1, parId,
              /*min iter for going parallel*/ 128)) {
        onnxToKrnlParallelReport(op, /*successful*/ true, 0, lbs[0], ubs[0],
            "LayoutTransform op fully parallelized with perfectly nested "
            "loops");
        create.krnl.parallel(loopDef[parId]);
      } else {
        onnxToKrnlParallelReport(op, /*successful*/ false, 0, lbs[0], ubs[0],
            "not enough work for LayoutTransform op");
      }
    }
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          // Simply copy the input into the output.
          Value val = createKrnl.load(data, indices);
          createKrnl.store(val, alloc, indices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXLayoutTransformOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXLayoutTransformOpLowering>(
      typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
