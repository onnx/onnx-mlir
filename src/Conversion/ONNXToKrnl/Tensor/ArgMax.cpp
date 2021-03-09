//===---------------- ArgMax.cpp - Lowering ArgMax Op -------------------===//
//
// This file lowers the ONNX ArgMax Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXArgMaxOpLowering : public ConversionPattern {
  ONNXArgMaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXArgMaxOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    ONNXArgMaxOpAdaptor operandAdaptor(operands);
    ONNXArgMaxOp argMaxOp = llvm::cast<ONNXArgMaxOp>(op);

    // shape helper
    ONNXArgMaxOpShapeHelper shapeHelper(&argMaxOp, &rewriter);

    auto shapecomputed = shapeHelper.Compute(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    // reduced output
    auto reducedMemRefType = convertToMemRefType(*op->result_type_begin());
    auto reducedElementType = reducedMemRefType.getElementType();
    auto reducedMemrefShape = reducedMemRefType.getShape();
    int64_t reducedRank = reducedMemRefType.getRank();

    // data input
    auto data = operandAdaptor.data();
    auto dataType = data.getType().cast<MemRefType>();
    int64_t dataRank = dataType.getRank();

    // axis & keepdims attribute
    int64_t axis = argMaxOp.axis();
    assert(axis >= -dataRank && axis <= dataRank - 1);
    axis = axis >= 0 ? axis : (dataRank + axis);

    int64_t keepdims = argMaxOp.keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    llvm::SmallVector<int64_t, 1> axes;
    axes.push_back(axis);
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(dataType, llvm::makeArrayRef(axes), isKeepdims);

    // Insert alloc and dealloc
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, reducedMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Constant Value
    auto minusOne = emitConstantOp(rewriter, loc, reducedElementType, -1);
    auto zero = emitConstantOp(rewriter, loc, reducedElementType, 0);
    auto zeroIndex = rewriter.create<ConstantIndexOp>(loc, 0);

    // 1. Krnl loops to initialize the result.
    BuildKrnlLoop initLoops(rewriter, loc, reducedRank);
    initLoops.createDefineOp();
    initLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    initLoops.createIterateOp();
    auto initLoopBody = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initLoops.getIterateBlock());

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : initLoops.getAllInductionVar()) {
      loopIVs.push_back(arg);
    }

    rewriter.create<KrnlStoreOp>(loc, minusOne, alloc, loopIVs);

    rewriter.restoreInsertionPoint(initLoopBody);

    // 2. Krnl loop to calculate argmax.
    BuildKrnlLoop calcLoops(rewriter, loc, dataRank);
    calcLoops.createDefineOp();
    for (int i = 0; i < dataRank; ++i)
      calcLoops.pushBounds(0, data, i);
    calcLoops.createIterateOp();
    rewriter.setInsertionPointToStart(calcLoops.getIterateBlock());

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs, maxLoopIVs;

    for (int i = 0; i < dataRank; ++i) {
      inLoopIVs.push_back(calcLoops.getInductionVar(i));
    }

    for (int i = 0; i < reducedRank; ++i) {
      if (outInDimMap.find(i) != outInDimMap.end()) {
        outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
      } else {
        outLoopIVs.push_back(zeroIndex);
      }
    }

    Value next = rewriter.create<KrnlLoadOp>(loc, data, inLoopIVs);
    Value idx = rewriter.create<KrnlLoadOp>(loc, alloc, outLoopIVs);

    // if index is less than 0, we should set 0 as initial position
    Value lessThanZero =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, idx, zero);
    idx = rewriter.create<SelectOp>(loc, lessThanZero, zero, idx);

    // induction variables of current max value
    for (int i = 0; i < dataRank; ++i) {
      if (i != axis)
        maxLoopIVs.push_back(calcLoops.getInductionVar(i));
      else
        maxLoopIVs.push_back(
            rewriter.create<IndexCastOp>(loc, idx, rewriter.getIndexType()));
    }
    Value maxVal = rewriter.create<KrnlLoadOp>(loc, data, maxLoopIVs);

    // if next value is larger than current max value, update index
    Value greaterThanMax =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, next, maxVal);
    Value pos = rewriter.create<IndexCastOp>(
        loc, inLoopIVs[axis], rewriter.getIntegerType(64));
    idx = rewriter.create<SelectOp>(loc, greaterThanMax, pos, idx);
    rewriter.create<KrnlStoreOp>(loc, idx, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXArgMaxOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXArgMaxOpLowering>(ctx);
}
