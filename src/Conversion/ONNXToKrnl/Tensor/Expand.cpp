//===---------------- Expand.cpp - Lowering Expand Op ---------------------===//
//
// This file lowers the ONNX Expand Operator to Krnl dialect.
//
// onnx.Expand(%input, %new_shape) : broadcast %input to match %new_shape. The
// works at a high level as:
//   - determine the final broadcast shape along each dimension and verify that
//   %input's shape and %new_shape agree. Dimensions agree if they have the same
//   value, or one of the values is 1.
//   - allocate the final expanded shape
//   - populate the final expanded shape with the appropriate elements of
//   %input:
//      affine.for %arg0 = 0 to (dim %expanded_shape 0) {
//          ...
//          affine.for %argj = 0 to (dim %expanded_shape j) {
//              affine.if ((dim %input 0) == 1) { %index0 = 0 }
//                                          else { %index0 = %arg0 }
//              ...
//              affine.if ((dim %input j) == 1) { %indexj = 0 }
//                                          else { %indexj = %argj }
//
//              %68 = load %input[%index0, ..., %indexj]
//              store %68, %expanded_shape[%arg0, %argj]
//          }
//      }
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IntegerSet.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

#define DEBUG_TYPE "expand-lowering"

struct ONNXExpandOpLowering : public ConversionPattern {
  ONNXExpandOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXExpandOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXExpandOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();
    Value shape = operandAdaptor.shape();

    auto inputShape = input.getType().cast<MemRefType>().getShape();
    int inputRank = inputShape.size();
    auto outputShape = shape.getType().cast<MemRefType>().getShape();
    assert(outputShape.size() == 1 && "shape argument needs to be a 1-D array");
    int outputRank = outputShape[0];

    // When broadcasting an input to a larger dimension, align on the rightmost
    // dimension. E.g. for a 2x3 input and 4x2x3 output, the dimensions
    // correspond as such:
    //                      2 x 3
    //                  4 x 2 x 3
    int outputDimensionOffset = outputRank - inputRank;
    assert(outputRank >= inputRank &&
           "Output rank must be the same or larger and the input rank.");

    // Create the affine broadcast check: if ()[s0] : (s0 == 1), then the
    // dimension corresponding to s0 is treated as a broadcast
    auto s0 = getAffineSymbolExpr(0, op->getContext());
    auto affine_1 = getAffineConstantExpr(1, op->getContext());
    auto affineEqToOne = IntegerSet::get(0, 1, {s0 - affine_1}, {true});
    Value i1 = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);

    // determine the size of each output dimension
    llvm::SmallVector<Value, 4> expandDimArray;
    llvm::SmallVector<Value, 4> inputDimsArray;
    for (int i = 0; i < outputRank; ++i) {
      Value outDimIndex =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
      auto outputDim = rewriter.create<KrnlLoadOp>(loc, shape, outDimIndex);
      auto outputDimCast =
          rewriter.create<IndexCastOp>(loc, outputDim, rewriter.getIndexType());
      if (i < outputDimensionOffset) {
        // If the input shape has an n-smaller rank than the output shape, the
        // output shape determines the first n dimension sizes
        expandDimArray.emplace_back(outputDimCast);
        continue;
      }

      Value inDimIndex = emitConstantOp(
          rewriter, loc, rewriter.getIndexType(), i - outputDimensionOffset);
      auto inputDim = rewriter.create<memref::DimOp>(loc, input, inDimIndex);
      inputDimsArray.emplace_back(inputDim);

      // %ifOp.results() = affine.if %inputDim == 1 {
      //    affine.yield %outputDim
      auto ifInputIsBroadcastDimOp = rewriter.create<AffineIfOp>(loc,
          rewriter.getIndexType(), affineEqToOne, ValueRange({inputDim}), true);
      rewriter.setInsertionPointToStart(
          &ifInputIsBroadcastDimOp.thenRegion().front());
      rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>(outputDimCast));
      rewriter.setInsertionPointToStart(
          &ifInputIsBroadcastDimOp.elseRegion().front());

      // } else {
      //    assert %outputDim == 1 || %outputDim == %inputDim
      //    affine.yield %inputDim
      // }
      auto checkOutputDimIsOne =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, outputDimCast, i1);
      auto checkInOutDimsAreEqual = rewriter.create<CmpIOp>(
          loc, CmpIPredicate::eq, outputDimCast, inputDim);
      auto checkOutputDim =
          rewriter.create<OrOp>(loc, checkOutputDimIsOne.getType(),
              checkOutputDimIsOne, checkInOutDimsAreEqual);
      rewriter.create<AssertOp>(loc, checkOutputDimIsOne, "Dimension mismatch");
      rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>(inputDim));
      rewriter.setInsertionPointAfter(ifInputIsBroadcastDimOp);
      expandDimArray.emplace_back(ifInputIsBroadcastDimOp.results()[0]);
    }

    // Insert an allocation and deallocation for the result of this operation.
    // Strip the static dimensions from the output shape. Alloc ops only
    // take dynamic dimensions as arguments.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefTypeShape = memRefType.getShape();
    llvm::SmallVector<Value, 4> dynamicAllocDims;
    for (int i = 0; i < memRefTypeShape.size(); ++i) {
      if (memRefTypeShape[i] == ShapedType::kDynamicSize) {
        dynamicAllocDims.emplace_back(expandDimArray[i]);
      }
    }
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = rewriter.create<memref::AllocOp>(loc, memRefType, dynamicAllocDims);
    if (insertDealloc) {
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
      dealloc.getOperation()->moveBefore(&parentBlock->back());
    }

    // Generate loop header
    std::vector<Value> copyLoop;
    defineLoops(rewriter, loc, copyLoop, outputRank);
    KrnlIterateOperandPack packInit(rewriter, copyLoop);
    for (int i = 0; i < outputRank; ++i) {
      addDimensionToPack(rewriter, loc, packInit, alloc, i);
    }
    auto iterateOpInit = rewriter.create<KrnlIterateOp>(loc, packInit);
    Block &iterationBlockInit = iterateOpInit.bodyRegion().front();

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);
    llvm::SmallVector<Value, 4> inputIdxList;
    llvm::SmallVector<Value, 4> loopIdxList;
    Value c0 = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
    for (int i = 0; i < outputRank; ++i) {
      Value loopIndex = iterationBlockInit.getArgument(i);
      loopIdxList.emplace_back(loopIndex);
      if (i < outputDimensionOffset)
        continue;

      // %ifOp.results() = affine.if %inputDim == 1 {
      //    affine.yield 0
      //  } else {
      //    affine.yield %loopIndex
      //  }
      auto ifOp = rewriter.create<AffineIfOp>(loc, rewriter.getIndexType(),
          affineEqToOne,
          ValueRange({inputDimsArray[i - outputDimensionOffset]}), true);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
      rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>(c0));
      rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
      rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>(loopIndex));
      rewriter.setInsertionPointAfter(ifOp);
      inputIdxList.emplace_back(ifOp.results()[0]);
    }

    auto loadOp = rewriter.create<KrnlLoadOp>(loc, input, inputIdxList);
    rewriter.create<KrnlStoreOp>(loc, loadOp, alloc, loopIdxList);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXExpandOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXExpandOpLowering>(ctx);
}
