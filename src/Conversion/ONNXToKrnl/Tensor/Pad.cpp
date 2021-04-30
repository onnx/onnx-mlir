/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

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

    DenseElementsAttr constantValAttr;
    if (getONNXConstantOp(myOp.constant_value())) {
      constantValAttr = getONNXConstantOp(myOp.constant_value())
                            .valueAttr()
                            .dyn_cast_or_null<DenseElementsAttr>();
    }
    if (!constantValAttr)
      return emitError(loc, "unsupported value");

    DenseElementsAttr padsAttributes;
    if (getONNXConstantOp(myOp.pads())) {
      padsAttributes = getONNXConstantOp(myOp.pads())
                           .valueAttr()
                           .dyn_cast_or_null<DenseElementsAttr>();
    }
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

    SmallVector<Value, 4> outLoopIVs;
    for (int i = 0; i < rank; ++i) {
      // Calculate the index for the load and store.
      if (pads[i] == 0) {
        outLoopIVs.emplace_back(valueLoops.getInductionVar(i));
      } else {
        AffineMap indexWithOffsetMap =
            AffineMap::get(1, 0, rewriter.getAffineDimExpr(0) + pads[i]);
        Value outIV = rewriter.create<AffineApplyOp>(loc, indexWithOffsetMap,
            ArrayRef<Value>{valueLoops.getInductionVar(i)});
        outLoopIVs.emplace_back(outIV);
      }
    }

    auto originValue =
        rewriter.create<KrnlLoadOp>(loc, operandAdaptor.data(), inLoopIVs);
    rewriter.create<KrnlStoreOp>(loc, originValue, alloc, outLoopIVs);
    rewriter.setInsertionPointToStart(padLoops.getIterateBlock());

    SmallVector<Value, 4> outLoopIVs1;
    for (int i = 0; i < rank; ++i)
      outLoopIVs1.emplace_back(padLoops.getInductionVar(i));

    auto paddingValue = rewriter.create<ConstantOp>(loc, valueAttr);
    rewriter.create<KrnlStoreOp>(loc, paddingValue, alloc, outLoopIVs1);

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXPadOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLowering>(ctx);
}
