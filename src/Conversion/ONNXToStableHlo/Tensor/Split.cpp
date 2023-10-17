/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSplitOp(A) is implemented using StableHlo sliceOp
struct ONNXSplitOpLoweringToStableHlo : public ConversionPattern {
  ONNXSplitOpLoweringToStableHlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSplitOpAdaptor operandAdaptor(operands);
    ONNXSplitOp splitOp = llvm::cast<ONNXSplitOp>(op);
    Value input = splitOp.getInput();
    Value split = splitOp.getSplit();
    assert(isRankedShapedType(input.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type indiceType = rewriter.getI64Type();
    Location loc = op->getLoc();
    uint64_t rank = inputType.getRank();
    uint64_t outputNum = splitOp.getNumResults();
    int64_t dimIndex = splitOp.getAxis();
    if (dimIndex < 0)
      dimIndex += rank;
    int64_t inputDimSize = inputType.getDimSize(dimIndex);

    // Get a shape helper (not used?)
    IndexExprBuilderForStableHlo createIE(rewriter, loc);
    ONNXSplitOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t, 4> splitSizes;
    if (auto splitAttr = getElementAttributeFromONNXValue(split)) {
      for (IntegerAttr value : splitAttr.getValues<IntegerAttr>()) {
        int64_t splitSize = value.cast<IntegerAttr>().getInt();
        splitSizes.push_back(splitSize);
      }
    } else if (split.getType().template isa<NoneType>()) {
      assert(!ShapedType::isDynamic(inputDimSize) &&
             "input dim size can't be dynamic");
      int64_t sliceSize = inputDimSize / outputNum;
      for (unsigned i = 0; i < outputNum; ++i)
        splitSizes.push_back(sliceSize);
    } else {
      assert(false && "dynamic split not yet supported");
    }

    SmallVector<int64_t, 4> sliceShape =
        llvm::to_vector<4>(inputType.getShape());
    SmallVector<int64_t, 4> beginIndices(rank, 0);
    SmallVector<int64_t, 4> endIndices =
        llvm::to_vector<4>(inputType.getShape());
    SmallVector<int64_t, 4> strides(rank, 1);
    SmallVector<Value, 4> slices;
    slices.reserve(outputNum);
    int64_t beginIndice = 0;
    int64_t endIndice = 0;
    for (uint64_t i = 0; i < outputNum; ++i) {
      sliceShape[dimIndex] = splitSizes[i];
      Type sliceType =
          RankedTensorType::get(sliceShape, inputType.getElementType());
      endIndice += splitSizes[i];
      beginIndices[dimIndex] = beginIndice;
      endIndices[dimIndex] = endIndice;
      slices.push_back(
          rewriter.create<stablehlo::SliceOp>(loc, sliceType, input,
              DenseIntElementsAttr::get(
                  RankedTensorType::get(
                      {static_cast<int64_t>(beginIndices.size())}, indiceType),
                  beginIndices),
              DenseIntElementsAttr::get(
                  RankedTensorType::get(
                      {static_cast<int64_t>(endIndices.size())}, indiceType),
                  endIndices),
              DenseIntElementsAttr::get(
                  RankedTensorType::get(
                      {static_cast<int64_t>(strides.size())}, indiceType),
                  strides)));
      beginIndice = endIndice;
    }
    rewriter.replaceOp(op, slices);
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToStableHloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToStableHlo>(ctx);
}

} // namespace onnx_mlir
