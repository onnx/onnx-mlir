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
    auto loc = op->getLoc();
    ONNXPadOp padOp = llvm::dyn_cast<ONNXPadOp>(op);
    ONNXPadOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.data();
    Value constantValue = operandAdaptor.constant_value();
    StringRef padMode = padOp.mode();

    // Builder helper.
    IndexExprScope outerScope(rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MemRefBuilder createMemRef(createKrnl);

    ONNXPadOpShapeHelper shapeHelper(&padOp, rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    assert(succeeded(shapecomputed));

    auto resMemRefType = convertToMemRefType(*op->result_type_begin());
    auto resElementType = resMemRefType.getElementType();
    // Insert an allocation and deallocation for the output of this operation.
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, shapeHelper.dimsForOutput(0));

    // 'constant' mode: initialize the result to the constant value.
    if (padMode.equals_insensitive("constant")) {
      Value cValue;
      if (constantValue.getType().isa<NoneType>())
        // Default to 0 if constant_value is not specified.
        cValue = emitConstantOp(rewriter, loc, resElementType, 0);
      else
        cValue = createKrnl.load(constantValue, {});
      createKrnl.memset(resMemRef, cValue);
    }

    // Copy values from the input to the result.
    MemRefBoundsIndexCapture dataBounds(data);
    uint64_t dataRank = dataBounds.getRank();
    SmallVector<IndexExpr, 4> lbs, ubs;
    for (uint64_t i = 0; i < dataRank; ++i) {
      lbs.emplace_back(LiteralIndexExpr(0));
      ubs.emplace_back(dataBounds.getDim(i));
    }

    ValueRange mainLoopDef = createKrnl.defineLoops(dataRank);
    createKrnl.iterateIE(mainLoopDef, mainLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange dataLoopInd) {
          SmallVector<IndexExpr, 4> resLoopInd;
          for (uint64_t i = 0; i < dataRank; ++i) {
            IndexExpr resInd =
                DimIndexExpr(dataLoopInd[i]) + shapeHelper.pads[i];
            resLoopInd.emplace_back(resInd);
          }
          Value dataValue = createKrnl.load(data, dataLoopInd);
          createKrnl.storeIE(dataValue, resMemRef, resLoopInd);
        });

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, resMemRef);

    return success();
  }
};

void populateLoweringONNXPadOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLowering>(ctx);
}
