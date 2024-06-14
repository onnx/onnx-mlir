/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- ScatterND.cpp - Lowering ScatterND Op ----------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file lowers the ONNX ScatterND Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXScatterNDOpLoweringToStablehlo
    : public OpConversionPattern<ONNXScatterNDOp> {
  ONNXScatterNDOpLoweringToStablehlo(MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(ONNXScatterNDOp scatterNDOp,
      ONNXScatterNDOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scatterNDOp.getOperation();
    Location loc = ONNXLoc<ONNXScatterNDOp>(op);

    // Operands and attributes.
    Value data = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    auto dataType = mlir::cast<ShapedType>(data.getType());
    auto indicesType = mlir::cast<ShapedType>(indices.getType());
    int64_t dataRank = dataType.getRank();
    int64_t indicesRank = indicesType.getRank();
    if (indicesType.isDynamicDim(indicesRank - 1))
      return rewriter.notifyMatchFailure(
          op, "only support indices with static last dim");
    int64_t partialIdxDim = indicesType.getDimSize(indicesRank - 1);

    assert(dataRank >= 1 && "The rank of 'data' must be >= 1");
    assert(indicesRank >= 1 && "The rank of 'indices' must be >= 1");

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = mlir::cast<ShapedType>(outputType);
    int64_t outputRank = outputShapedType.getRank();
    assert(outputRank == dataRank && "Output rank not equal to data rank");
    auto scatter_dimension_numbers =
        mlir::stablehlo::ScatterDimensionNumbersAttr::get(
            /*context=*/rewriter.getContext(),
            /*updateWindowDims*/
            llvm::to_vector<4>(llvm::seq<int64_t>(partialIdxDim, dataRank)),
            /*insertedWindowDims*/
            llvm::to_vector<4>(llvm::seq<int64_t>(0, partialIdxDim)),
            /*inputBatchingDims=*/{},
            /*scatterIndicesBatchingDims=*/{},
            /*scatterDimsToOperandDims*/
            llvm::to_vector<4>(llvm::seq<int64_t>(0, partialIdxDim)),
            /*indexVectorDim=*/indicesRank - 1);
    auto scatterOp = rewriter.create<stablehlo::ScatterOp>(
        loc, outputType, data, indices, updates, scatter_dimension_numbers);
    // config update computation function: just return the element from src.
    Block &block = scatterOp.getUpdateComputation().emplaceBlock();
    // add block arguments
    auto blockArgumentType =
        RankedTensorType::get({}, dataType.getElementType());
    block.addArgument(blockArgumentType, loc);
    block.addArgument(blockArgumentType, loc);

    auto *lhsArg = block.args_begin();
    auto *rhsArg = std::next(lhsArg);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      rewriter.create<stablehlo::ReturnOp>(loc, *rhsArg);
    }

    rewriter.replaceOp(op, scatterOp.getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXScatterNDOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXScatterNDOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
