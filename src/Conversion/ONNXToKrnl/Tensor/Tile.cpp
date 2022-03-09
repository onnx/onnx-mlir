/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Tile.cpp - Lowering Tile Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Tile Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//

Value insertAllocAndDeallocForTile(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, bool insertDealloc, Value inputOperand,
    Value repeatsOperand) {
  MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
      rewriter, loc);
  auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
  size_t inputRank = inputShape.size();
  auto outputShape = memRefType.getShape();

  SmallVector<Value, 4> allocOperands;
  for (size_t i = 0; i < inputRank; ++i) {
    if (outputShape[i] == -1) {
      Value indexVal = create.math.constantIndex(i);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      Value repeatsLoadVal = create.krnl.load(repeatsOperand, repeatsMemRefVal);
      Value repeatsElementVal = create.math.castToIndex(repeatsLoadVal);
      Value dimVal = create.mem.dim(inputOperand, i);
      Value allocDimVal = create.math.mul(dimVal, repeatsElementVal);
      allocOperands.emplace_back(allocDimVal);
    }
  }

  memref::AllocOp alloc = create.mem.alignedAlloc(memRefType, allocOperands);
  if (insertDealloc) {
    Block *parentBlock = alloc.getOperation()->getBlock();
    memref::DeallocOp dealloc = create.mem.dealloc(alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  return alloc;
}

struct ONNXTileOpLowering : public ConversionPattern {
  ONNXTileOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    ONNXTileOp tileOp = cast<ONNXTileOp>(op);
    Location loc = op->getLoc();

    ONNXTileOpShapeHelper shapeHelper(&tileOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);

    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    Value input = operandAdaptor.input();

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, shapeHelper.dimsForOutput(0));

    // Define loops and iteration trip counts (equivalent to size of output)
    BuildKrnlLoop outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    SmallVector<Value, 4> loadIndices;
    // This implementation is to iterate the output tensor.
    // The store has simple affine subscript expression.
    // Alternative implementation is to iterate the input tensor and repeats.
    // The load of elements in input tensor can be reused explicitly.
    // But the subscript of store is not contigious, or even not affine.
    // Alternative implementation can be found at the end of this file.

    for (int64_t i = 0; i < outputRank; i++) {
      // Scope is created for each dimension because they are independent
      IndexExprScope IEScope(&rewriter, loc);
      DimIndexExpr index(outputLoops.getInductionVar(i));
      MemRefBoundsIndexCapture inputBounds(input);
      DimIndexExpr dimSize(inputBounds.getDim(i));
      IndexExpr exprVal = index % dimSize;
      loadIndices.emplace_back(exprVal.getValue());
    }

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);
    Value loadVal = create.krnl.load(input, loadIndices);

    SmallVector<Value, 4> storeIndices;
    for (int64_t i = 0; i < outputRank; ++i)
      storeIndices.emplace_back(outputLoops.getInductionVar(i));
    create.krnl.store(loadVal, alloc, storeIndices);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

// This is the alternative way of lowering.
// It is kept here for record in case this implementation is needed
struct ONNXTileOpLoweringAlternative : public ConversionPattern {
  ONNXTileOpLoweringAlternative(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXTileOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTileOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // get input operands, shapes, and rank
    Value input = operandAdaptor.input();
    auto inputShape = input.getType().cast<MemRefType>().getShape();
    int64_t inputRank = inputShape.size();
    Value repeats = operandAdaptor.repeats();

    // get output info
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = (hasAllConstantDimensions(outputMemRefType))
                      ? insertAllocAndDealloc(
                            outputMemRefType, loc, rewriter, insertDealloc)
                      : insertAllocAndDeallocForTile(outputMemRefType, loc,
                            rewriter, insertDealloc, input, repeats);

    // Define loops and iteration trip counts (equivalent to size of output)
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank * 2);
    KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (int64_t ii = 0; ii < outputRank; ++ii) {
      addDimensionToPack(rewriter, loc, pack, input, ii);
      pack.pushConstantBound(0);
      Value indexVal =
          emitConstantOp(rewriter, loc, rewriter.getIndexType(), ii);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      Value repeatsLoadVal = create.krnl.load(repeats, repeatsMemRefVal);
      auto repeatsElementVal = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), repeatsLoadVal);
      pack.pushOperandBound(repeatsElementVal);
    }

    // Create the loops
    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operations.
    SmallVector<Value, 4> inputMemRefVal;
    for (int64_t j = 0; j < inputRank; ++j)
      inputMemRefVal.emplace_back(iterationBlock.getArguments()[j * 2]);

    SmallVector<Value, 4> outputMemRefVal;
    for (int64_t i = 0; i < inputRank; ++i) {
      Value inputDimSizeVal = create.mem.dim(input, i);
      if (inputShape[i] != -1) {
        AffineExpr inputIndexAE = rewriter.getAffineDimExpr(0);
        AffineExpr repeatsIndexAE = rewriter.getAffineDimExpr(1);
        AffineExpr inputDimAE = rewriter.getAffineSymbolExpr(0);

        auto dimMap =
            AffineMap::get(2, 1, inputDimAE * repeatsIndexAE + inputIndexAE);
        auto dimExprVal = rewriter.create<AffineApplyOp>(loc, dimMap,
            ArrayRef<Value>{iterationBlock.getArguments()[2 * i],
                iterationBlock.getArguments()[2 * i + 1], inputDimSizeVal});
        outputMemRefVal.emplace_back(dimExprVal);
      } else {
        auto inputIndex = iterationBlock.getArguments()[2 * i];
        auto repeatsIndex = iterationBlock.getArguments()[2 * i + 1];
        Value dimExprVal = create.math.add(
            inputIndex, create.math.mul(repeatsIndex, inputDimSizeVal));
        outputMemRefVal.emplace_back(dimExprVal);
      }
    }

    Value inputVal = create.krnl.load(input, inputMemRefVal);
    create.krnl.store(inputVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXTileOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLowering>(typeConverter, ctx);
}
