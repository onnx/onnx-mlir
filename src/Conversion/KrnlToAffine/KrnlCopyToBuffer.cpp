/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- KrnlCopyToBuffer.cpp - Lower KrnlCopyToBufferOp -----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnCopyToBufferOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToAffine/KrnlToAffineHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlCopyToBufferLowering : public ConversionPattern {
public:
  explicit KrnlCopyToBufferLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, KrnlCopyToBufferOp::getOperationName(),
            1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Get info from operands.
    auto copyToBufferOp = cast<KrnlCopyToBufferOp>(op);
    KrnlCopyToBufferOpAdaptor operandAdaptor(copyToBufferOp);
    Value buffMemref(operandAdaptor.buffer());
    Value sourceMemref(operandAdaptor.source());
    ValueRange startVals(operandAdaptor.starts());
    Value padVal(operandAdaptor.padValue());
    int64_t srcRank =
        sourceMemref.getType().cast<MemRefType>().getShape().size();
    int64_t buffRank =
        buffMemref.getType().cast<MemRefType>().getShape().size();
    int64_t srcOffset = srcRank - buffRank;
    assert(srcOffset >= 0 && "offset expected non negative");
    Location loc = copyToBufferOp.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);
    SmallVector<IndexExpr, 4> starts, bufferReadUBs, bufferPadUBs;
    MemRefBoundsIndexCapture buffBounds(buffMemref);
    MemRefBoundsIndexCapture sourceBounds(sourceMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    ArrayAttributeIndexCapture padCapture(copyToBufferOp.padToNextAttr(), 1);
    ArrayAttributeIndexCapture readSizeCapture(copyToBufferOp.tileSizeAttr());
    // Handle possible transpose by having an indirect array for indices
    // used in conjunction with source.
    SmallVector<int64_t, 4> srcIndexMap, srcLoopMap;
    generateIndexMap(srcIndexMap, srcRank, copyToBufferOp.transpose());
    generateIndexMap(srcLoopMap, buffRank, copyToBufferOp.transpose());

    // Overread not currently used, will if we simdize reads or
    // unroll and jam loops.
    // ArrayAttributeIndexCapture overCapture(op.overreadToNextAttr(), 1);

    // Determine here bufferReadUBs, which determine how many values of source
    // memeref to copy into the buffer. Also determine bufferPadUBs, which is
    // the upper bound past bufferReadUBs that must be padded.
    // This is only done on the dimensions shared between src memref and buffer.
    LiteralIndexExpr zero(0);
    for (long buffIndex = 0; buffIndex < buffRank; ++buffIndex) {
      long srcIndex = srcIndexMap[srcOffset + buffIndex];
      // Compute how many values to read.
      IndexExpr sourceBound =
          sourceBounds.getSymbol(srcIndex); // Source memref size.
      IndexExpr blockSize =
          buffBounds.getSymbol(buffIndex); // Buffer memref size.
      if (readSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = readSizeCapture.getLiteral(buffIndex); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "readTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI =
          starts[srcIndex]; // Global index in source memref of tile.
      IndexExpr bufferRead = trip(sourceBound, blockSize, startGI);
      bufferRead.debugPrint("buffer read");
      bufferReadUBs.emplace_back(bufferRead);
      // Determine the UB until which to pad
      IndexExpr padToNext = padCapture.getLiteral(buffIndex);
      int64_t padToNextLit =
          padToNext.getLiteral(); // Will assert if undefined.
      int64_t blockSizeLit = blockSize.getLiteral(); // Will assert if not lit.
      if (bufferRead.isLiteralAndIdenticalTo(blockSizeLit)) {
        // Read the full buffer already, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (bufferRead.isLiteral() &&
                 bufferRead.getLiteral() % padToNextLit == 0) {
        // We are already reading to the end of a line.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == 1) {
        // Add pad % 1... namely no pad, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == blockSizeLit) {
        // Pad to end.
        bufferPadUBs.emplace_back(blockSize);
      } else {
        assert(padToNextLit > 1 && padToNextLit < blockSizeLit &&
               "out of range padToLit");
        IndexExpr newPadUB = (bufferRead.ceilDiv(padToNext)) * padToNext;
        bufferPadUBs.emplace_back(newPadUB);
      }
    }
    SmallVector<Value, 4> loopIndices;
    genCopyLoops(createAffine, &indexScope, buffMemref, sourceMemref,
        srcLoopMap, padVal, zero, starts, bufferReadUBs, bufferPadUBs,
        loopIndices, 0, buffRank, false);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(AffineBuilderKrnlMem &createAffine,
      IndexExprScope *enclosingScope, Value buffMemref, Value sourceMemref,
      SmallVectorImpl<int64_t> &srcLoopMap, Value padVal, IndexExpr zero,
      SmallVectorImpl<IndexExpr> &starts, SmallVectorImpl<IndexExpr> &readUBs,
      SmallVectorImpl<IndexExpr> &padUBs, SmallVectorImpl<Value> &loopIndices,
      int64_t i, int64_t buffRank, bool padPhase) const {
    if (i == buffRank) {
      // create new scope and import index expressions
      IndexExprScope currScope(createAffine, enclosingScope);
      KrnlBuilder createKrnl(createAffine);
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      if (!padPhase) {
        SmallVector<IndexExpr, 4> currLoadIndices;
        getIndexExprList<DimIndexExpr>(starts, currStarts);
        int64_t srcRank = starts.size();
        int64_t srcOffset = srcRank - buffRank;
        for (long srcIndex = 0; srcIndex < srcRank; ++srcIndex) {
          if (srcIndex < srcOffset) {
            // Dimensions that are unique to source memref, just use starts.
            currLoadIndices.emplace_back(currStarts[srcIndex]);
          } else {
            // Dimensions that are shared by source memref & buffer, add loop
            // indices to starts.
            int64_t buffIndex = srcIndex - srcOffset;
            currLoadIndices.emplace_back(
                currLoopIndices[srcLoopMap[buffIndex]] + currStarts[srcIndex]);
          }
        }
        Value sourceVal = createKrnl.loadIE(sourceMemref, currLoadIndices);
        createKrnl.storeIE(sourceVal, buffMemref, currLoopIndices);
      } else {
        createKrnl.storeIE(padVal, buffMemref, currLoopIndices);
      }
    } else {
      readUBs[i].getValue();
      if (readUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to read, skip.
      } else {
        createAffine.forIE(zero, readUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref,
                  sourceMemref, srcLoopMap, padVal, zero, starts, readUBs,
                  padUBs, loopIndices, i + 1, buffRank,
                  /*no pad phase*/ false);
              loopIndices.pop_back_n(1);
            });
      }
      if (padUBs[i].isLiteralAndIdenticalTo(0)) {
        // No padding needed.
      } else {
        createAffine.forIE(readUBs[i], padUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref,
                  sourceMemref, srcLoopMap, padVal, zero, starts, readUBs,
                  padUBs, loopIndices, i + 1, buffRank,
                  /*pad phase*/ true);
              loopIndices.pop_back_n(1);
            });
      }
      // For next level up of padding, if any, will not copy data anymore
      readUBs[i] = zero;
    }
  }
};

void populateLoweringKrnlCopyToBufferOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlCopyToBufferLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
