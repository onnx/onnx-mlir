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
    ONNXTransposeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    Value data = operandAdaptor.data();

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, op);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, op, {data});

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, rank);

    KrnlIterateOperandPack pack(rewriter, originalLoops);
    // Iterate over the loop nest using the input shape.
    for (int i = 0; i < rank; ++i)
      addDimensionToPack(rewriter, loc, pack, data, i);

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation.

    // Read perm attribute.
    SmallVector<int, 4> perm;
    auto permAttribute = llvm::dyn_cast<ONNXTransposeOp>(op).permAttr();
    assert(permAttribute && "permute attribute expected to be defined here");
    for (auto permVal : permAttribute.getValue())
      perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

    SmallVector<Value, 4> inLoopIVs;
    for (auto arg : iterationBlock.getArguments())
      inLoopIVs.emplace_back(arg);

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < iterationBlock.getArguments().size(); ++i)
      outLoopIVs.emplace_back(iterationBlock.getArguments()[perm[i]]);

    auto inVal = rewriter.create<AffineLoadOp>(loc, data, inLoopIVs);
    rewriter.create<AffineStoreOp>(loc, inVal, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTransposeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLowering>(ctx);
}
