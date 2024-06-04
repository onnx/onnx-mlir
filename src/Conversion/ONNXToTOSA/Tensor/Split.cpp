/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Split.cpp - Split Op---------===//
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX SplitOp operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
namespace onnx_mlir {
namespace {
class ONNXSplitOpLoweringToTOSA : public OpConversionPattern<ONNXSplitOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXSplitOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSplitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    ShapedType inputType = cast<ShapedType>(input.getType());

    // tosa.slice does not allow a dynamic entry in the size attribute
    if (!hasStaticShape(inputType))
      return rewriter.notifyMatchFailure(
          op, "only static shapes are supported");

    uint64_t rank = inputType.getRank();
    int64_t splitAxis = adaptor.getAxis();
    if (splitAxis < 0)
      splitAxis += rank;

    IndexExprBuilderForTosa createTosaIE(rewriter, op->getLoc());
    ONNXSplitOpShapeHelper shapeHelper(
        op, adaptor.getOperands(), &createTosaIE);

    // compute shape
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(op, "could not compute shape.");

    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    uint64_t outputNum = op.getNumResults();
    SmallVector<Value, 4> slices;
    slices.reserve(outputNum);

    llvm::SmallVector<int64_t, 4> size;
    llvm::SmallVector<int64_t, 4> starts(rank, 0);
    int64_t start = 0;

    for (uint64_t i = 0; i < outputNum; i++) {
      DimsExpr outputDim = shapeHelper.getOutputDims(i);
      IndexExpr::getShape(outputDim, size);
      starts[splitAxis] = start;
      slices.push_back(tosaBuilder.slice(input, size, starts));
      start += size[splitAxis];
    }
    rewriter.replaceOp(op, slices);
    return success();
  }
};
} // namespace

void populateLoweringONNXSplitOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSplitOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir