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
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToAffine/KrnlToAffineHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

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
    KrnlCopyFromBufferOp copyFromBufferOp =
        mlir::cast<KrnlCopyFromBufferOp>(op);
    Location loc = copyFromBufferOp.getLoc();
    MultiDialectBuilder<AffineBuilderKrnlMem, IndexExprBuilderForKrnl> create(
        rewriter, loc);
    IndexExprScope indexScope(create.affineKMem);

    KrnlCopyFromBufferOpAdaptor operandAdaptor(copyFromBufferOp);
    Value buffMemref(operandAdaptor.getBuffer());
    Value destMemref(operandAdaptor.getDest());
    ValueRange startVals(operandAdaptor.getStarts());
    int64_t destRank =
        mlir::cast<MemRefType>(destMemref.getType()).getShape().size();
    int64_t buffRank =
        mlir::cast<MemRefType>(buffMemref.getType()).getShape().size();
    int64_t destOffset = destRank - buffRank;
    assert(destOffset >= 0 && "offset expected non negative");

    auto writeSizeAttr = copyFromBufferOp.getTileSizeAttr();
    SmallVector<IndexExpr, 4> starts, bufferWriteUBs;
    getIndexExprList<DimIndexExpr>(startVals, starts);
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zeroIE(0);

    for (long buffIndex = 0; buffIndex < buffRank; ++buffIndex) {
      long destIndex = destOffset + buffIndex;
      // Compute how many values to read.
      IndexExpr destBound = create.krnlIE.getShapeAsSymbol(
          destMemref, destIndex); // Source memref size.
      IndexExpr blockSize = create.krnlIE.getShapeAsSymbol(
          buffMemref, buffIndex); // Buffer memref size.
      if (create.krnlIE.getArraySize(writeSizeAttr)) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = create.krnlIE.getIntFromArrayAsLiteral(
            writeSizeAttr, buffIndex); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "writeTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI =
          starts[destIndex]; // Global index in dest memref of tile.
      IndexExpr bufferWrite =
          create.krnlIE.tileSize(startGI, blockSize, destBound);
      bufferWrite.debugPrint("buffer wrote");
      bufferWriteUBs.emplace_back(bufferWrite);
    }
    genCopyLoops(create.affineKMem, &indexScope, buffMemref, destMemref, zeroIE,
        starts, bufferWriteUBs, loopIndices, 0, buffRank);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(const AffineBuilderKrnlMem &createAffine,
      IndexExprScope *enclosingScope, Value buffMemref, Value destMemref,
      IndexExpr zeroIE, SmallVectorImpl<IndexExpr> &starts,
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
        createAffine.forLoopIE(zeroIE, writeUBs[i], 1, false /*parallel*/,
            [&](const AffineBuilderKrnlMem &createAffine, ValueRange loopInd) {
              loopIndices.emplace_back(loopInd[0]);
              genCopyLoops(createAffine, enclosingScope, buffMemref, destMemref,
                  zeroIE, starts, writeUBs, loopIndices, i + 1, buffRank);
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
