/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- KrnlCopyFromBuffer.cpp - Lower KrnlCopyFromBufferOp ---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnCopyFromBufferOp operator.
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

class KrnlCopyFromBufferLowering : public ConversionPattern {
public:
  explicit KrnlCopyFromBufferLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter,
            KrnlCopyFromBufferOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto copyFromBufferOp = cast<KrnlCopyFromBufferOp>(op);
    KrnlCopyFromBufferOpAdaptor operandAdaptor(copyFromBufferOp);
    Value buffMemref(operandAdaptor.buffer());
    Value destMemref(operandAdaptor.dest());
    ValueRange startVals(operandAdaptor.starts());
    int64_t destRank =
        destMemref.getType().cast<MemRefType>().getShape().size();
    int64_t buffRank =
        buffMemref.getType().cast<MemRefType>().getShape().size();
    int64_t destOffset = destRank - buffRank;
    assert(destOffset >= 0 && "offset expected non negative");
    ArrayAttributeIndexCapture writeSizeCapture(
        copyFromBufferOp.tileSizeAttr());

    Location loc = copyFromBufferOp.getLoc();
    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);

    SmallVector<IndexExpr, 4> starts, bufferWriteUBs;
    MemRefBoundsIndexCapture buffBounds(buffMemref);
    MemRefBoundsIndexCapture destBounds(destMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zero(0);

    for (long buffIndex = 0; buffIndex < buffRank; ++buffIndex) {
      long destIndex = destOffset + buffIndex;
      // Compute how many values to read.
      IndexExpr destBound =
          destBounds.getSymbol(destIndex); // Source memref size.
      IndexExpr blockSize =
          buffBounds.getSymbol(buffIndex); // Buffer memref size.
      if (writeSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = writeSizeCapture.getLiteral(buffIndex); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "writeTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI =
          starts[destIndex]; // Global index in dest memref of tile.
      IndexExpr bufferWrite = trip(destBound, blockSize, startGI);
      bufferWrite.debugPrint("buffer wrote");
      bufferWriteUBs.emplace_back(bufferWrite);
    }
    genCopyLoops(createAffine, &indexScope, buffMemref, destMemref, zero,
        starts, bufferWriteUBs, loopIndices, 0, buffRank);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(AffineBuilderKrnlMem &createAffine,
      IndexExprScope *enclosingScope, Value buffMemref, Value destMemref,
      IndexExpr zero, SmallVectorImpl<IndexExpr> &starts,
      SmallVectorImpl<IndexExpr> &writeUBs, SmallVectorImpl<Value> &loopIndices,
      int64_t i, int64_t buffRank) const {
    if (i == buffRank) {
      // create new scope and import index expressions
      IndexExprScope currScope(createAffine, enclosingScope);
      KrnlBuilder createKrnl(createAffine);
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      getIndexExprList<SymbolIndexExpr>(starts, currStarts);
      int64_t destRank = starts.size();
      int64_t destOffset = destRank - buffRank;
      SmallVector<IndexExpr, 4> currStoreIndices;
      for (long destIndex = 0; destIndex < destRank; ++destIndex) {
        if (destIndex < destOffset) {
          // Dimensions that are unique to source memref, just use starts.
          currStoreIndices.emplace_back(currStarts[destIndex]);
        } else {
          // Dimensions that are shared by source memref & buffer, add loop
          // indices to starts.
          int64_t buffIndex = destIndex - destOffset;
          currStoreIndices.emplace_back(
              currLoopIndices[buffIndex] + currStarts[destIndex]);
        }
      }
      Value destVal = createKrnl.loadIE(buffMemref, currLoopIndices);
      createKrnl.storeIE(destVal, destMemref, currStoreIndices);
    } else {
      if (writeUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to write.
      } else {
        // Loop to copy the data.
        createAffine.forIE(zero, writeUBs[i], 1,
            [&](AffineBuilderKrnlMem &createAffine, Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(createAffine, enclosingScope, buffMemref, destMemref,
                  zero, starts, writeUBs, loopIndices, i + 1, buffRank);
              loopIndices.pop_back_n(1);
            });
      }
    }
  }
};

void populateLoweringKrnlCopyFromBufferOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlCopyFromBufferLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
