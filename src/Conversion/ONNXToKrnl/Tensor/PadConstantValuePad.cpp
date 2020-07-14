//===------PadConstantValuePad.cpp - Lowering PadConstantValuePad Op ------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX PadConstantValuePad  Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXPadConstantValuePadOpLowering : public ConversionPattern {
  ONNXPadConstantValuePadOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXPadConstantValuePadOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin());
    ONNXPadConstantValuePadOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Only constant padding is supported now.
    auto padMode = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).mode();
    if (padMode != "constant")
      return emitError(loc, "unsupported mode for PadConstantValuePad");
    auto constantValAttr =
        llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).constant_valueAttr();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(tensorType);

    // Create init block if this is the first operation in the function.
    createInitState(rewriter, loc, op);

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, op);
    else
      return emitError(loc, "unexpected output has non-Constant shape");

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Iterate over the loop nest using the output shape.
    BuildKrnlLoop padLoops(rewriter, loc, rank);
    padLoops.createDefineOp();
    for (int i = 0; i < rank; ++i)
      padLoops.pushBounds(0, alloc, i);
    padLoops.createIterateOp();

    // Iterate over the loop nest using the input shape.
    BuildKrnlLoop valueLoops(rewriter, loc, rank);
    valueLoops.createDefineOp();
    for (int i = 0; i < rank; ++i)
      valueLoops.pushBounds(0, operandAdaptor.data(), i);
    valueLoops.createIterateOp();

    // Copy the input data into the output.
    rewriter.setInsertionPointToStart(valueLoops.getIterateBlock());

    SmallVector<Value, 4> inLoopIVs;
    for (int i = 0; i < rank; ++i)
      inLoopIVs.emplace_back(valueLoops.getInductionVar(i));

    auto pads = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).pads();
    SmallVector<int64_t, 4> pad_begin;
    for (int i = 0; i < pads.size() / 2; ++i) {
      pad_begin.emplace_back(pads.getValue()[i].cast<IntegerAttr>().getInt());
    }

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < rank; ++i) {
      // Calculate the index for the load and store.
      if (pad_begin[i] == 0) {
        outLoopIVs.emplace_back(valueLoops.getInductionVar(i));
      } else {
        AffineMap indexWithOffsetMap =
            AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + pad_begin[i]);
        Value outIV = rewriter.create<AffineApplyOp>(loc, indexWithOffsetMap,
            ArrayRef<Value>{valueLoops.getInductionVar(i)});
        outLoopIVs.emplace_back(outIV);
      }
    }

    auto inVal =
        rewriter.create<AffineLoadOp>(loc, operandAdaptor.data(), inLoopIVs);
    rewriter.create<AffineStoreOp>(loc, inVal, alloc, outLoopIVs);
    rewriter.setInsertionPointToStart(padLoops.getIterateBlock());

    SmallVector<Value, 4> outLoopIVs1;
    for (int i = 0; i < rank; ++i)
      outLoopIVs1.emplace_back(padLoops.getInductionVar(i));

    auto inVal1 = rewriter.create<ConstantOp>(loc, constantValAttr);
    rewriter.create<AffineStoreOp>(loc, inVal1, alloc, outLoopIVs1);

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPadConstantValuePadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadConstantValuePadOpLowering>(ctx);
}
