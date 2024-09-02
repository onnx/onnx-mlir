/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Split.cpp - Lowering Split Op -----------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Split Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// ONNXSplitOp(A) is implemented using Stablehlo sliceOp
struct ONNXSplitOpLoweringToStablehlo : public ConversionPattern {
  ONNXSplitOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSplitOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXSplitOpAdaptor operandAdaptor(operands);
    ONNXSplitOp splitOp = llvm::cast<ONNXSplitOp>(op);
    Value input = splitOp.getInput();
    Value split = splitOp.getSplit();
    assert(isRankedShapedType(input.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    uint64_t rank = inputType.getRank();
    uint64_t outputNum = splitOp.getNumResults();
    int64_t dimIndex = splitOp.getAxis();
    if (dimIndex < 0)
      dimIndex += rank;
    int64_t inputDimSize = inputType.getDimSize(dimIndex);

    // Get a shape helper (not used?)
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXSplitOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    SmallVector<int64_t, 4> splitSizes;
    if (auto splitAttr = getElementAttributeFromONNXValue(split)) {
      for (IntegerAttr value : splitAttr.getValues<IntegerAttr>()) {
        int64_t splitSize = mlir::cast<IntegerAttr>(value).getInt();
        splitSizes.push_back(splitSize);
      }
    } else if (mlir::isa<NoneType>(split.getType())) {
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
      slices.push_back(rewriter.create<stablehlo::SliceOp>(loc, sliceType,
          input, DenseI64ArrayAttr::get(context, beginIndices),
          DenseI64ArrayAttr::get(context, endIndices),
          DenseI64ArrayAttr::get(context, strides)));
      beginIndice = endIndice;
    }
    rewriter.replaceOp(op, slices);
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
