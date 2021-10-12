/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {

  assert(origAttrs && "handle EXISTING ArrayAttr only");
  if (origAttrs.getValue()[0].dyn_cast<FloatAttr>()) {
    mlir::Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
  if (origAttrs.getValue()[0].dyn_cast<IntegerAttr>()) {
    mlir::Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
  llvm_unreachable("unexpected attribute type");
}

ConstantOp createUnitConstant(PatternRewriter &rewriter, Location loc) {
  return rewriter.create<ConstantOp>(loc, rewriter.getUnitAttr());
}

// Create an DenseElementsAttr of ArrayAttr.
// When ArrayAttr is Null, an empty Integer DenseElementAttr is returned
DenseElementsAttr createDenseArrayAttrOrEmpty(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {

  if (origAttrs) {
    return createDenseArrayAttr(rewriter, origAttrs);
  } else {
    mlir::Type elementType = rewriter.getIntegerType(64);
    int nElements = 0;
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = i;
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

class ConvertONNXDepthToSpacePattern
    : public OpRewritePattern<ONNXDepthToSpaceOp> {
public:
  using OpRewritePattern<ONNXDepthToSpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXDepthToSpaceOp depthToSpaceOp,
      PatternRewriter &rewriter) const override {
    auto loc = depthToSpaceOp.getLoc();
    Value input = depthToSpaceOp.input();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elementType = inputType.getElementType();
    auto blocksize = depthToSpaceOp.blocksize();
    auto N = inputShape[0];
    auto C = inputShape[1];
    auto H = inputShape[2];
    auto W = inputShape[3];

    SmallVector<int64_t, 6> shape1;
    ArrayAttr perm;
    if (depthToSpaceOp.mode() == "DCR") {
      shape1 = {N, blocksize, blocksize, C / (blocksize * blocksize), H, W};
      perm = rewriter.getI64ArrayAttr({0, 3, 4, 1, 5, 2});
    } else {
      assert(depthToSpaceOp.mode() == "CRD" && "Unexpected DepthToShape mode");
      shape1 = {N, C / (blocksize * blocksize), blocksize, blocksize, H, W};
      perm = rewriter.getI64ArrayAttr({0, 1, 4, 2, 5, 3});
    }

    // DCR: reshape = onnx.Reshape(input, [N, blocksize, blocksize,
    //   C / (blocksize * blocksize), H, W])
    // CRD: reshape = onnx.Reshape(input, [N, C / (blocksize * blocksize),
    //   blocksize, blocksize, H, W])
    auto shapeConstantOp1 = getONNXConstantOpFromDenseAttr(
        rewriter, loc, rewriter.getI64TensorAttr({shape1}));
    auto reshape = rewriter
                       .create<ONNXReshapeOp>(loc,
                           RankedTensorType::get(shape1, elementType), input,
                           shapeConstantOp1)
                       .getResult();

    // DCR: transpose = onnx.Transpose(reshape, [0, 3, 4, 1, 5, 2])
    // CRD: transpose = onnx.Transpose(reshape, [0, 1, 4, 2, 5, 3])
    SmallVector<int64_t, 6> transposeShape = {
        N, C / (blocksize * blocksize), H, blocksize, W, blocksize};
    auto transpose = rewriter
                         .create<ONNXTransposeOp>(loc,
                             RankedTensorType::get(transposeShape, elementType),
                             reshape, perm)
                         .getResult();

    // result = onnx.Reshape(transpose, [N, C / (blocksize * blocksize), H *
    // blocksize, W * blocksize])
    SmallVector<int64_t, 4> shape2 = {
        N, C / (blocksize * blocksize), H * blocksize, W * blocksize};
    auto shapeConstantOp2 = getONNXConstantOpFromDenseAttr(
        rewriter, loc, rewriter.getI64TensorAttr({shape2}));
    rewriter.replaceOpWithNewOp<ONNXReshapeOp>(depthToSpaceOp,
        RankedTensorType::get(shape2, elementType), transpose,
        shapeConstantOp2);

    return success();
  }
};

namespace {

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void DecomposeONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect, StandardOpsDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXDepthToSpaceOp>();
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXUpsampleOp>();
  target.addIllegalOp<ONNXUpsampleV9Op>();
  target.addIllegalOp<ONNXUpsampleV7Op>();
  target.addIllegalOp<ONNXPadV2Op>();
  target.addIllegalOp<ONNXPadV11Op>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<ConvertONNXDepthToSpacePattern>(&getContext());

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}
