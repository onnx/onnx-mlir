//===------------- Conv.cpp - Lowering Conv Op to Linalg ----------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Conv Operator to Linalg dialect.
//
// Current support: 2D convolution (4D tensors) only
// Requirements:
//   - padding=0
//   - dilation=1
//   - bias=none
//   - group=1
// TODO: Add support for:
//   - padding > 0
//   - dilation > 1
//   - bias
//   - group > 1
//   - dynamic shapes
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConvOpLoweringToLinalg : public OpRewritePattern<ONNXConvOp> {
  ONNXConvOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXConvOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const final {
    // Check if this operation should be converted to Linalg based on
    // --linalg-ops option
    Operation *op = convOp.getOperation();
    if (!shouldConvertToLinalg(op, linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          convOp, "operation not selected for Linalg conversion");
    }

    ONNXConvOpAdaptor adaptor(convOp);
    Location loc = convOp.getLoc();
    Value X = adaptor.getX();
    Value W = adaptor.getW();
    Value B = adaptor.getB();

    // Type Check: Get input types
    auto xType = dyn_cast<RankedTensorType>(X.getType());
    auto wType = dyn_cast<RankedTensorType>(W.getType());
    if (!xType || !wType) {
      return rewriter.notifyMatchFailure(
          convOp, "expected ranked tensor types");
    }

    // Rank Check: Only support 4D tensors (2D convolution)
    if (xType.getRank() != 4 || wType.getRank() != 4) {
      return rewriter.notifyMatchFailure(convOp,
          "only 4D tensors (2D convolution) are currently supported in Linalg "
          "lowering");
    }

    // Group Check: Only support group=1
    int64_t group = convOp.getGroup();
    if (group != 1) {
      return rewriter.notifyMatchFailure(
          convOp, "only group=1 is currently supported in Linalg lowering");
    }

    // Auto-pad Check: Only support auto_pad="NOTSET"
    StringRef autoPad = convOp.getAutoPad();
    if (autoPad != "NOTSET") {
      return rewriter.notifyMatchFailure(
          convOp, "only auto_pad=NOTSET is currently supported in Linalg "
                  "lowering");
    }

    // Padding Check: Only support padding=0
    auto padsOpt = convOp.getPads();
    if (padsOpt.has_value()) {
      ArrayAttr pads = padsOpt.value();
      // pads format: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
      for (size_t i = 0; i < pads.size(); ++i) {
        int64_t padVal = ArrayAttrIntVal(pads, i);
        if (padVal != 0) {
          return rewriter.notifyMatchFailure(
              convOp, "only padding=0 is currently supported in Linalg "
                      "lowering");
        }
      }
    }

    // Dilation Check: Only support dilation=1
    auto dilationsOpt = convOp.getDilations();
    if (dilationsOpt.has_value()) {
      ArrayAttr dilations = dilationsOpt.value();
      for (size_t i = 0; i < dilations.size(); ++i) {
        int64_t dilationVal = ArrayAttrIntVal(dilations, i);
        if (dilationVal != 1) {
          return rewriter.notifyMatchFailure(
              convOp, "only dilation=1 is currently supported in Linalg "
                      "lowering");
        }
      }
    }

    // Bias Check: Only support bias=none
    if (!isNoneValue(B)) {
      return rewriter.notifyMatchFailure(
          convOp, "only bias=none is currently supported in Linalg lowering");
    }

    // Extract attributes: Strides
    SmallVector<int64_t> strides = {1, 1};
    auto stridesOpt = convOp.getStrides();
    if (stridesOpt.has_value()) {
      ArrayAttr stridesAttr = stridesOpt.value();
      strides[0] = ArrayAttrIntVal(stridesAttr, 0);
      strides[1] = ArrayAttrIntVal(stridesAttr, 1);
    }
    auto stridesDenseAttr = rewriter.getI64TensorAttr(strides);

    // Dilations: Fixed to [1, 1] since we only support dilation=1
    auto dilationsDenseAttr = rewriter.getI64TensorAttr({1, 1});

    // Get output type and shape
    Type outputType = convOp.getResult().getType();
    auto outputTensorType = dyn_cast<RankedTensorType>(outputType);
    if (!outputTensorType) {
      return rewriter.notifyMatchFailure(
          convOp, "expected ranked output tensor type");
    }

    // For now, use static shapes from the output type
    // TODO: Handle dynamic shapes properly with ShapeHelper
    ArrayRef<int64_t> outputShape = outputTensorType.getShape();

    // Create initialization tensor
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputShape, outputTensorType.getElementType());

    // Create zero constant for initialization
    Value zero = arith::ConstantOp::create(rewriter, loc,
        outputTensorType.getElementType(),
        rewriter.getZeroAttr(outputTensorType.getElementType()));

    // Fill the output tensor with zeros
    Value filledTensor = linalg::FillOp::create(
        rewriter, loc, ValueRange{zero}, ValueRange{emptyTensor})
                             .getResult(0);

    // Create linalg.conv_2d_nchw_fchw operation
    Value convResult = linalg::Conv2DNchwFchwOp::create(rewriter, loc,
        TypeRange{outputTensorType},  // result type
        ValueRange{X, W},             // inputs: [input, filter]
        ValueRange{filledTensor},     // outputs: [init tensor]
        stridesDenseAttr,            // DenseIntElementsAttr [2]
        dilationsDenseAttr)          // DenseIntElementsAttr [2] = [1, 1]
                             .getResult(0);

    rewriter.replaceOp(convOp, convResult);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXConvOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXConvOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir

