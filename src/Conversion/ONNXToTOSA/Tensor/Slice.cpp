/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Slice.cpp - Slice Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX SliceOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXSliceLoweringToTOSA : public OpConversionPattern<ONNXSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXSliceOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    if (!adaptor.getStarts().getDefiningOp<mlir::tosa::ConstOp>()) {
      return rewriter.notifyMatchFailure(op, "starts must be constant");
    }
    if (!adaptor.getEnds().getDefiningOp<mlir::tosa::ConstOp>()) {
      return rewriter.notifyMatchFailure(op, "ends must be constant");
    }

    // Get shape.
    IndexExprBuilderForTosa createTosaIE(rewriter, loc);
    ONNXSliceOpShapeHelper shapeHelper(op, {}, &createTosaIE);
    if (failed(shapeHelper.computeShape())) {
      return rewriter.notifyMatchFailure(op, "could not compute shape.");
    }

    auto inTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto outTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only ranked tensor types are supported");
    }
    auto inShape = inTy.getShape();
    auto outShape = outTy.getShape();

    TosaBuilder tosaBuilder(rewriter, loc);
    Value val = adaptor.getData();

    if (!(IndexExpr::isLiteral(shapeHelper.starts)))
      return rewriter.notifyMatchFailure(op, "starts has no literals.");
    if (!(IndexExpr::isLiteral(shapeHelper.ends)))
      return rewriter.notifyMatchFailure(op, "ends has no literals.");
    if (!(IndexExpr::isLiteral(shapeHelper.steps)))
      return rewriter.notifyMatchFailure(op, "steps has no literals.");

    llvm::SmallVector<int64_t, 4> starts;
    IndexExpr::getLiteral(shapeHelper.starts, starts);
    llvm::SmallVector<int64_t, 4> ends;
    IndexExpr::getLiteral(shapeHelper.ends, ends);
    llvm::SmallVector<int64_t, 4> steps;
    IndexExpr::getLiteral(shapeHelper.steps, steps);

    if (llvm::any_of(steps, [](int64_t step) { return step < 0; })) {
      return rewriter.notifyMatchFailure(op, "negative step not supported.");
    }

    // 1: Pad if not enough data at the end to fit the final step
    // start (S) = 2, end (E) = -1, step (T) = 4
    // |SSXTTTXTTTXTTTXTTTXTE| => |SSXTTTXTTTXTTTXTTTXTEP|
    llvm::SmallVector<int64_t, 4> paddedOutShape(inShape);
    llvm::SmallVector<int64_t, 8> pads(inShape.size(), 0);
    pads.resize(inShape.size() * 2);
    for (size_t i = 0; i < inShape.size(); i++) {
      int64_t padNeeded =
          std::max((outShape[i] * steps[i]) - (inShape[i] - starts[i]), 0l);

      pads[inShape.size() + i] = padNeeded;

      paddedOutShape[i] += padNeeded;
    }
    if (llvm::any_of(pads, [](int64_t pad) { return pad > 0; })) {
      Value constPads =
          tosa::buildOnnxToTosaPaddingConstOp(rewriter, pads, loc, {}, {});
      auto constTosaTensor =
          tosaBuilder.getSplattedConst(0.0, inTy.getElementType(), 0);
      val = rewriter.create<mlir::tosa::PadOp>(loc,
          mlir::RankedTensorType::get(paddedOutShape, inTy.getElementType()),
          val, constPads, constTosaTensor);
    }

    // 2: Slice the edges leaving enough data to fit the final step
    // start (S) = 2, end (E) = -5, step (T) = 4
    // |SSXTTTXTTTXTTTXTTTXTEEEEE| => |XTTTXTTTXTTTXTTTXTEE|
    llvm::SmallVector<int64_t, 4> sizes;
    sizes.resize(inShape.size());
    for (size_t i = 0; i < inShape.size(); i++) {
      sizes[i] = steps[i] * outShape[i];
    }
    if (sizes != paddedOutShape) {
      val = tosaBuilder.slice(val, sizes, starts);
    }

    // If we all the steps are 1, then we are done
    if (llvm::all_of(steps, [](int64_t step) { return step == 1; })) {
      rewriter.replaceOp(op, val);
      return success();
    }

    // 3: Reshape along steps
    // step (T) = 4
    //                           |XTTT|
    //                           |XTTT|
    // |XTTTXTTTXTTTXTTTXTTT| => |XTTT|
    //                           |XTTT|
    //                           |XTTT|
    llvm::SmallVector<int64_t, 4> newShape;
    for (size_t i = 0; i < inShape.size(); i++) {
      newShape.push_back(outShape[i]);
      if (steps[i] != 1)
        newShape.push_back(steps[i]);
    }
    val = tosaBuilder.reshape(val, newShape);

    // 4: Slice the steps
    // |XTTT|    |X|
    // |XTTT|    |X|
    // |XTTT| => |X|
    // |XTTT|    |X|
    // |XTTT|    |X|
    llvm::SmallVector<int64_t, 4> newSizes;
    for (size_t i = 0; i < inShape.size(); i++) {
      newSizes.push_back(outShape[i]);
      if (steps[i] != 1)
        newSizes.push_back(1);
    }
    val = tosaBuilder.slice(
        val, newSizes, llvm::SmallVector<int64_t>(newShape.size(), 0));

    // 5: Reshape to original output size
    // |X|
    // |X|
    // |X| => |XXXXX|
    // |X|
    // |X|
    val = tosaBuilder.reshape(val, outShape);

    rewriter.replaceOp(op, val);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXSliceLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
