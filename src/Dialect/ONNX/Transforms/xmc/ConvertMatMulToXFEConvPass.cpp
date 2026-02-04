// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include <numeric>

using namespace mlir;

namespace {

/// Structure to hold convolution shape information
struct ConvShapes {
  SmallVector<int64_t> inputShape;  // [M, H, W, C]
  SmallVector<int64_t> weightShape; // [C_out, H', W', C]
  SmallVector<int64_t> stride;      // [H', W']
};

/// TODO: Implement proper convolution shapes computation based on HW info.
/// Helper function to compute convolution shapes from MatMul shapes.
/// For now, uses dummy logic: H=W=H'=W'=1, so C=K and C_out=N
ConvShapes computeConvShapes(ArrayRef<int64_t> inputShape,
                             ArrayRef<int64_t> weightShape) {
  ConvShapes shapes;

  // Extract K (last dimension of input) and N (last dimension of weight)
  if (inputShape.empty() || weightShape.size() < 2) {
    return shapes; // Invalid shapes
  }

  int64_t K = inputShape.back();
  int64_t N = weightShape.back();

  // Compute M: product of all but last input dimension
  int64_t M = 1;
  for (size_t i = 0; i < inputShape.size() - 1; ++i) {
    if (inputShape[i] == ShapedType::kDynamic) {
      M = ShapedType::kDynamic;
      break;
    }
    M *= inputShape[i];
  }

  // Dummy logic: H=W=H'=W'=1
  int64_t H = 1;
  int64_t W = 1;
  int64_t H_prime = 1;
  int64_t W_prime = 1;
  int64_t C = K;
  int64_t C_out = N;

  // Input shape for XFEConv: [M, H, W, C]
  shapes.inputShape = {M, H, W, C};

  // Weight shape for XFEConv: [C_out, H', W', C]
  shapes.weightShape = {C_out, H_prime, W_prime, C};

  // Stride: [H', W']
  shapes.stride = {H_prime, W_prime};

  return shapes;
}

/// Helper function to create a shape constant for ONNX Reshape
Value createShapeConstant(PatternRewriter &rewriter, Location loc,
                          ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Pattern to convert MatMul to Reshape -> XFEConv -> Reshape
struct MatMulToXFEConvPattern : public OpRewritePattern<ONNXMatMulOp> {
  using OpRewritePattern<ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp matMulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matMulOp.getLoc();

    // Get input and weight
    Value input = matMulOp.getA();
    Value weight = matMulOp.getB();

    // Get types
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());

    if (!inputType || !weightType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape()) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();

    // Verify MatMul shape: input [D1, D2, ..., Dn, K], weight [K, N]
    if (inputShape.empty() || weightShape.size() < 2 ||
        inputShape.back() != weightShape[0]) {
      return failure();
    }

    // Compute convolution shapes
    ConvShapes convShapes = computeConvShapes(inputShape, weightShape);

    // Get element type (preserve quantization type)
    Type elementType = inputType.getElementType();

    // Create first Reshape: [D1, D2, ..., Dn, K] -> [M, H, W, C]
    auto reshape1OutputType =
        RankedTensorType::get(convShapes.inputShape, elementType);
    auto shapeConst1 =
        createShapeConstant(rewriter, loc, convShapes.inputShape);
    Value reshape1Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape1OutputType, input, shapeConst1);

    // Format weight: [K, N] -> transpose to [N, K] -> reshape to [N, 1, 1, K].
    auto weightElementType = weightType.getElementType();
    auto convWeightType =
        RankedTensorType::get(convShapes.weightShape, weightElementType);
    auto weightShapeConst =
        createShapeConstant(rewriter, loc, convShapes.weightShape);

    // Require weight to be a 2D constant so we can safely transpose it.
    auto weightConstOp = weight.getDefiningOp<ONNXConstantOp>();
    if (!weightConstOp || weightShape.size() != 2) {
      return failure();
    }

    // Transpose [K, N] -> [N, K].
    auto transposedWeightType = RankedTensorType::get(
        {weightShape[1], weightShape[0]}, weightElementType);
    auto permAttr = rewriter.getI64ArrayAttr({1, 0});
    Value transposedWeight = rewriter.create<ONNXTransposeOp>(
        loc, transposedWeightType, weight, permAttr);

    // Reshape to [N, 1, 1, K] for XFEConv.
    Value convWeight = rewriter.create<ONNXReshapeOp>(
        loc, convWeightType, transposedWeight, weightShapeConst);

    // Create XFEConv
    // XFEConv expects: input [M, H, W, C], weight [C_out, H', W', C]
    // Output: [M, H/H', W/W', C_out]
    int64_t outputH = convShapes.inputShape[1] / convShapes.stride[0];
    int64_t outputW = convShapes.inputShape[2] / convShapes.stride[1];
    SmallVector<int64_t> convOutputShape = {convShapes.inputShape[0], outputH,
                                            outputW, convShapes.weightShape[0]};

    auto convOutputType = RankedTensorType::get(convOutputShape, elementType);

    // Create attributes for XFEConv
    auto autoPadAttr = rewriter.getStringAttr("NOTSET");
    auto stridesAttr = rewriter.getI64ArrayAttr(convShapes.stride);
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(
        {convShapes.weightShape[1], convShapes.weightShape[2]});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto dilationsAttr = rewriter.getI64ArrayAttr({1, 1});
    auto groupAttr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true),
        APInt(64, 1, /*isSigned=*/true));

    // Create none value for bias
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value noneBias = onnxBuilder.none();

    // Create XFEConv operation
    auto convOp = rewriter.create<XFEConvOp>(loc, convOutputType, reshape1Output,
                                             convWeight, noneBias, autoPadAttr,
                                             dilationsAttr, groupAttr,
                                             kernelShapeAttr, padsAttr,
                                             stridesAttr);

    // Create second Reshape: [M, H/H', W/W', C_out] -> [D1, D2, ..., Dn, N]
    // Original output shape: [D1, D2, ..., Dn, N]
    SmallVector<int64_t> outputShape;
    for (size_t i = 0; i < inputShape.size() - 1; ++i) {
      outputShape.push_back(inputShape[i]);
    }
    outputShape.push_back(weightShape.back()); // N

    auto reshape2OutputType =
        RankedTensorType::get(outputShape, elementType);
    auto shapeConst2 = createShapeConstant(rewriter, loc, outputShape);
    Value reshape2Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape2OutputType, convOp.getResult(), shapeConst2);

    // Replace MatMul with the final Reshape output
    rewriter.replaceOp(matMulOp, reshape2Output);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ConvertMatMulToXFEConvPass
    : public PassWrapper<ConvertMatMulToXFEConvPass,
                         OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "convert-matmul-to-xfe-conv"; }
  StringRef getDescription() const override {
    return "Convert MatMul operations to XFEConv operations";
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Add pattern
    patterns.add<MatMulToXFEConvPattern>(context);

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertMatMulToXFEConvPass() {
  return std::make_unique<ConvertMatMulToXFEConvPass>();
}

} // namespace onnx_mlir
