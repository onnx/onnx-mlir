/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------Tile.cpp - Lowering Tile Op----------------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Tile Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//

Value insertAllocForTile(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, Value inputOperand,
    Value repeatsOperand) {
  MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
      rewriter, loc);
  auto inputShape = mlir::cast<MemRefType>(inputOperand.getType()).getShape();
  size_t inputRank = inputShape.size();
  auto outputShape = memRefType.getShape();

  SmallVector<Value, 4> allocOperands;
  for (size_t i = 0; i < inputRank; ++i) {
    if (ShapedType::isDynamic(outputShape[i])) {
      Value indexVal = create.math.constantIndex(i);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      Value repeatsLoadVal = create.krnl.load(repeatsOperand, repeatsMemRefVal);
      Value repeatsElementVal = create.math.castToIndex(repeatsLoadVal);
      Value dimVal = create.mem.dim(inputOperand, i);
      Value allocDimVal = create.math.mul(dimVal, repeatsElementVal);
      allocOperands.emplace_back(allocDimVal);
    }
  }

  return create.mem.alignedAlloc(memRefType, allocOperands);
}

struct ONNXTileOpLowering : public OpConversionPattern<ONNXTileOp> {
  ONNXTileOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXTileOp tileOp, ONNXTileOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = tileOp.getOperation();
    Location loc = ONNXLoc<ONNXTileOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXTileOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t outputRank = memRefShape.size();

    Value input = adaptor.getInput();
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims());

    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));

    create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          // Compute the indices used by the input tensor load operation.
          // Note: An alternative implementation can be found at the end of this
          // file.
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
              createKrnl);

          SmallVector<Value, 4> loadIndices;
          for (uint64_t i = 0; i < outputRank; ++i) {
            DimIndexExpr index(indices[i]);
            IndexExpr dimSize = create.krnlIE.getShapeAsSymbol(input, i);
            IndexExpr exprVal = index % dimSize;
            loadIndices.emplace_back(exprVal.getValue());
          }

          Value loadVal = create.krnl.load(input, loadIndices);
          create.krnl.store(loadVal, alloc, indices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

// This is the alternative way of lowering.
// It is kept here for record in case this implementation is needed
struct ONNXTileOpLoweringAlternative : public OpConversionPattern<ONNXTileOp> {
  ONNXTileOpLoweringAlternative(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXTileOp tileOp, ONNXTileOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = tileOp.getOperation();
    Location loc = ONNXLoc<ONNXTileOp>(op);

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // get input operands, shapes, and rank
    Value input = adaptor.getInput();
    auto inputShape = mlir::cast<MemRefType>(input.getType()).getShape();
    int64_t inputRank = inputShape.size();
    Value repeats = adaptor.getRepeats();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    auto outputMemRefShape = outputMemRefType.getShape();
    int64_t outputRank = outputMemRefShape.size();

    Value alloc = (hasAllConstantDimensions(outputMemRefType))
                      ? create.mem.alignedAlloc(outputMemRefType)
                      : insertAllocForTile(
                            outputMemRefType, loc, rewriter, input, repeats);

    // Define loops and iteration trip counts (equivalent to size of output)
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, outputRank * 2);
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (int64_t ii = 0; ii < outputRank; ++ii) {
      addDimensionToPack(rewriter, loc, pack, input, ii);
      pack.pushConstantBound(0);
      Value indexVal = create.math.constant(rewriter.getIndexType(), ii);
      SmallVector<Value, 1> repeatsMemRefVal = {indexVal};
      Value repeatsLoadVal = create.krnl.load(repeats, repeatsMemRefVal);
      auto repeatsElementVal = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), repeatsLoadVal);
      pack.pushOperandBound(repeatsElementVal);
    }

    // Create the loops
    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();

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
      if (!ShapedType::isDynamic(inputShape[i])) {
        AffineExpr inputIndexAE = rewriter.getAffineDimExpr(0);
        AffineExpr repeatsIndexAE = rewriter.getAffineDimExpr(1);
        AffineExpr inputDimAE = rewriter.getAffineSymbolExpr(0);

        auto dimMap =
            AffineMap::get(2, 1, inputDimAE * repeatsIndexAE + inputIndexAE);
        auto dimExprVal = rewriter.create<affine::AffineApplyOp>(loc, dimMap,
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
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXTileOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXTileOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
