//===----padconstantvaluepad.cpp - Lowering PadConstantValuePad Op --------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX PadConstantValuePad  Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXPadConstantValuePadOpLowering : public ConversionPattern {
  ONNXPadConstantValuePadOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXPadConstantValuePadOp::getOperationName(),
                          1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Only constant padding is supported now.
    auto padMode = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).mode();
    if (padMode != "constant")
      emitError(loc, "unsupported mode for PadConstantValuePad");
    auto constantValAttr =
        llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).constant_valueAttr();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(tensorType);
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      emitError(loc, "unexpected output has non-Constant shape");

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Iterate over the loop nest using the output shape.
    BuildKrnlLoop padLoops(rewriter, loc, rank);
    padLoops.createDefineAndOptimizeOp();
    for (int i = 0; i < rank; ++i)
      padLoops.pushBounds(0, alloc, i);
    padLoops.createIterateOp();

    // Iterate over the loop nest using the input shape.
    BuildKrnlLoop valueLoops(rewriter, loc, rank);
    valueLoops.createDefineAndOptimizeOp();
    for (int i = 0; i < rank; ++i)
      valueLoops.pushBounds(0, operands[0], i);
    valueLoops.createIterateOp();

    // Copy the input data into the output.
    rewriter.setInsertionPointToStart(valueLoops.getIterateBlock());

    SmallVector<Value, 4> inLoopIVs;
    for (int i = 0; i < rank; ++i)
      inLoopIVs.emplace_back(valueLoops.getInductionVar(i));

    auto pads = llvm::dyn_cast<ONNXPadConstantValuePadOp>(op).pads();
    SmallVector<int64_t, 4> pad_begin;
    for (int i = 0; i < pads.size(); i += 2) {
      pad_begin.emplace_back(pads.getValue()[i].cast<IntegerAttr>().getInt());
    }

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < rank; ++i) {
      // Calculate the index for the load and store.
      if (pad_begin[i] == 0) {
        outLoopIVs.emplace_back(valueLoops.getInductionVar(i));
      } else {
        auto outIV = rewriter.create<AddIOp>(
            loc, rewriter.create<ConstantIndexOp>(loc, pad_begin[i]),
            valueLoops.getInductionVar(i));
        outLoopIVs.emplace_back(outIV);
      }
    }

    auto inVal = rewriter.create<LoadOp>(loc, operands[0], inLoopIVs);
    rewriter.create<StoreOp>(loc, inVal, alloc, outLoopIVs);
    rewriter.setInsertionPointToStart(padLoops.getIterateBlock());

    SmallVector<Value, 4> outLoopIVs1;
    for (int i = 0; i < rank; ++i)
      outLoopIVs1.emplace_back(padLoops.getInductionVar(i));

    auto inVal1 = rewriter.create<ConstantOp>(loc, constantValAttr);
    rewriter.create<StoreOp>(loc, inVal1, alloc, outLoopIVs1);

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXPadConstantValuePadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadConstantValuePadOpLowering>(ctx);
}
