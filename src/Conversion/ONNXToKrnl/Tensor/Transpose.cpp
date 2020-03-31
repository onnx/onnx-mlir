//===---------------- Transpose.cpp - Lowering Transpose Op ---------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Transpose Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXTransposeOpLowering : public ConversionPattern {
  ONNXTransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTransposeOpOperandAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    Value data = operandAdaptor.data();

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {data});

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock =
        defineLoops(rewriter, loc, originalLoops, optimizedLoops, rank);

    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    // Iterate over the loop nest using the input shape.
    for (int i = 0; i < rank; ++i)
      addDimensionToPack(rewriter, loc, pack, data, i);

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation.

    // Read perm attribute.
    SmallVector<int, 4> perm;
    auto permAttribute = llvm::dyn_cast<ONNXTransposeOp>(op).permAttr();
    if (permAttribute) {
      for (auto permVal : permAttribute.getValue())
        perm.emplace_back(permVal.cast<IntegerAttr>().getInt());
    } else {
      // TODO: Remove when perm is guaranteed to be present (even for
      // the default case). This means that perm was added by shape
      // inference or another pass to contain the values corresponding
      // to the default behavior of Transpose.
      for (int i = iterationBlock.getArguments().size() - 1; i >= 0; i--)
        perm.emplace_back(i);
    }

    SmallVector<Value, 4> inLoopIVs;
    for (auto arg : iterationBlock.getArguments())
      inLoopIVs.emplace_back(arg);

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < iterationBlock.getArguments().size(); ++i)
      outLoopIVs.emplace_back(iterationBlock.getArguments()[perm[i]]);

    auto inVal = rewriter.create<LoadOp>(loc, data, inLoopIVs);
    rewriter.create<StoreOp>(loc, inVal, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTransposeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(ctx);
}
