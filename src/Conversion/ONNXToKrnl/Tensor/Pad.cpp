//===-----------------------Pad.cpp - Lowering Pad Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pad  Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXPadOpLowering : public ConversionPattern {
  ONNXPadOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXPadOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXPadOp myOp = llvm::dyn_cast<ONNXPadOp>(op);
    ONNXPadOpAdaptor operandAdaptor(operands);
    auto tensorType = myOp.output().getType();

    auto loc = op->getLoc();

    // Only constant padding is supported now.
    auto padMode = myOp.mode();
    if (padMode != "constant")
      return emitError(loc, "unsupported mode for Pad");
    DenseElementsAttr constantValAttr =
        myOp.getAttr("constant_value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (!constantValAttr)
      return emitError(loc, "unsupported value");

    DenseElementsAttr padsAttributes =
        myOp.getAttr("pads").dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (!padsAttributes)
      return emitError(loc, "Pad: unknown pads");

    auto memRefType = convertToMemRefType(tensorType);
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      return emitError(loc, "unexpected output has non-Constant shape");

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // get the padding vector into a temporary smallvector
    SmallVector<int64_t, 2> pads(rank * 2, -1);
    auto padsIt = padsAttributes.getValues<IntegerAttr>().begin();
    for (int i = 0; i < rank * 2; ++i)
      pads[i] = (*padsIt++).cast<IntegerAttr>().getInt();

    // get the padding value
    auto valueAttr = (*constantValAttr.getValues<FloatAttr>().begin());

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
      valueLoops.pushBounds(0, operandAdaptor.data(), i);
    valueLoops.createIterateOp();

    // Copy the input data into the output.
    rewriter.setInsertionPointToStart(valueLoops.getIterateBlock());

    SmallVector<Value, 4> inLoopIVs;
    for (int i = 0; i < rank; ++i)
      inLoopIVs.emplace_back(valueLoops.getInductionVar(i));

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < rank; ++i) {
      // Calculate the index for the load and store.
      if (pads[i] == 0) {
        outLoopIVs.emplace_back(valueLoops.getInductionVar(i));
      } else {
        auto outIV = rewriter.create<AddIOp>(loc,
            rewriter.create<ConstantIndexOp>(loc, pads[i]),
            valueLoops.getInductionVar(i));
        outLoopIVs.emplace_back(outIV);
      }
    }

    auto originValue =
        rewriter.create<LoadOp>(loc, operandAdaptor.data(), inLoopIVs);
    rewriter.create<StoreOp>(loc, originValue, alloc, outLoopIVs);
    rewriter.setInsertionPointToStart(padLoops.getIterateBlock());

    SmallVector<Value, 4> outLoopIVs1;
    for (int i = 0; i < rank; ++i)
      outLoopIVs1.emplace_back(padLoops.getInductionVar(i));

    auto paddingValue = rewriter.create<ConstantOp>(loc, valueAttr);
    rewriter.create<StoreOp>(loc, paddingValue, alloc, outLoopIVs1);

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPadOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLowering>(ctx);
}
