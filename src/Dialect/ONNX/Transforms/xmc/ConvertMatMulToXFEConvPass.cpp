// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"

#include <algorithm>
#include <numeric>
#include <optional>

using namespace mlir;

/// Max channel for Conv.
static constexpr int64_t kMaxChannel = 4096;

/// Max kernel/stride for H,W in general fallback composition.
/// Derived from conv_limit().stride() for IPU_STRIX/AIE4 target ("1-4").
static constexpr int64_t kMaxKernelLimitDefault = 4;

namespace {

/// If `elemType` is a UniformQuantizedPerAxisType, return a copy with
/// quantizedDimension set to `newAxis`.  Otherwise return `elemType` unchanged.
static Type remapPerAxisQuantDim(Type elemType, int32_t newAxis) {
  if (auto perAxis = dyn_cast<quant::UniformQuantizedPerAxisType>(elemType)) {
    return quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(),
        perAxis.getStorageType(), perAxis.getExpressedType(),
        perAxis.getScales(), perAxis.getZeroPoints(), newAxis,
        perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
  }
  return elemType;
}

/// Structure to hold convolution shape information
struct ConvShapes {
  SmallVector<int64_t> inputShape;  // [M, H, W, C]
  SmallVector<int64_t> weightShape; // [C_out, H', W', C]
  SmallVector<int64_t> stride;      // [H', W']
};

/// Factorize into primes (2s first, then odd primes ascending).
/// Returns (factors.size() >= num, factors).
static std::pair<bool, SmallVector<int64_t>> integerDecomposition(
    int64_t input, int64_t num = 1) {
  SmallVector<int64_t> ret;
  int64_t n = input;
  while (n % 2 == 0) {
    ret.push_back(2);
    n /= 2;
  }
  for (int64_t f = 3; f * f <= n; f += 2) {
    while (n % f == 0) {
      ret.push_back(f);
      n /= f;
    }
  }
  if (n != 1)
    ret.push_back(n);
  bool if_enough = (int64_t)ret.size() >= num;
  return {if_enough, ret};
}

/// Round-robin distribute factors into 3 buckets (H, W, C).
/// Bounds (low, high): each result must satisfy low <= ret[idx] < high.
static std::pair<bool, SmallVector<int64_t>> integerComposition(
    SmallVector<int64_t> integers,
    std::array<std::pair<int64_t, int64_t>, 3> bounds) {
  int64_t originalProd = 1;
  for (int64_t x : integers)
    originalProd *= x;
  std::sort(integers.begin(), integers.end());
  SmallVector<int64_t> ret = {1, 1, 1};
  constexpr size_t retSize = 3;
  if (integers.size() < retSize)
    integers.resize(retSize, 1);

  size_t idx = 0;
  int credits = retSize;
  auto it = integers.begin();
  while (it != integers.end() && credits > 0) {
    --credits;
    int64_t top = bounds[idx].second;
    int64_t cur = *it;
    if (ret[idx] < top && ret[idx] * cur < top) {
      ret[idx] *= cur;
      it = integers.erase(it);
      credits = retSize;
    }
    idx = (idx + 1) % retSize;
  }
  int64_t retProd = ret[0] * ret[1] * ret[2];
  bool if_success = (retProd == originalProd);
  for (size_t i = 0; i < retSize; ++i) {
    if_success &= (bounds[i].first <= ret[i] && ret[i] < bounds[i].second);
  }
  return {if_success, ret};
}

/// Decompose K into H*W*C. Tries IPU-preferred tight bounds first,
/// then falls back to general bounds.
static std::optional<std::tuple<int64_t, int64_t, int64_t>> decomposeK(
    int64_t K, int64_t maxKernelLimit = kMaxKernelLimitDefault) {
  auto [dec_ok, dec_rlts] = integerDecomposition(K);
  if (!dec_ok)
    return std::nullopt;

  if (dec_rlts.size() < 3) {
    dec_rlts.clear();
    dec_rlts.push_back(K);
    dec_rlts.resize(3, 1);
  }

  // Try IPU-preferred tight bounds: H,W < 2 (i.e. H=W=1)
  std::array<std::pair<int64_t, int64_t>, 3> ipuBounds = {
      std::make_pair(1, 2),
      std::make_pair(1, 2),
      std::make_pair(1, kMaxChannel),
  };
  auto [ipu_ok, ipu_hwc] = integerComposition(dec_rlts, ipuBounds);
  if (ipu_ok && ipu_hwc.size() >= 3)
    return {{ipu_hwc[0], ipu_hwc[1], ipu_hwc[2]}};

  // General fallback: H,W < maxKernelLimit, C < 4096
  std::array<std::pair<int64_t, int64_t>, 3> bounds = {
      std::make_pair(1, maxKernelLimit),
      std::make_pair(1, maxKernelLimit),
      std::make_pair(1, kMaxChannel),
  };
  auto [comp_ok, hwc] = integerComposition(dec_rlts, bounds);
  if (!comp_ok || hwc.size() < 3)
    return std::nullopt;
  return {{hwc[0], hwc[1], hwc[2]}};
}

/// Compute convolution shapes. Returns nullopt when decomposition fails.
std::optional<ConvShapes> computeConvShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape) {
  if (inputShape.empty() || weightShape.size() < 2)
    return std::nullopt;

  int64_t K = inputShape.back();
  int64_t N = weightShape.back();

  int64_t M = 1;
  for (size_t i = 0; i < inputShape.size() - 1; ++i) {
    if (inputShape[i] == ShapedType::kDynamic) {
      M = ShapedType::kDynamic;
      break;
    }
    M *= inputShape[i];
  }

  auto hwc = decomposeK(K);
  if (!hwc)
    return std::nullopt;

  auto [H, W, C] = *hwc;
  ConvShapes shapes;
  shapes.inputShape = {M, H, W, C};
  shapes.weightShape = {N, H, W, C};
  shapes.stride = {H, W};
  return shapes;
}

/// Helper function to transfer onnx_node_name attribute from source to target
/// op
void transferOnnxNodeName(Operation *sourceOp, Operation *targetOp) {
  if (!sourceOp || !targetOp)
    return;

  // Get onnx_node_name from source operation
  auto onnxNodeName =
      sourceOp->getAttrOfType<mlir::StringAttr>("onnx_node_name");

  // If source has onnx_node_name, set it on target
  if (onnxNodeName && !onnxNodeName.getValue().empty()) {
    targetOp->setAttr("onnx_node_name", onnxNodeName);
  }
}

/// Helper function to create a shape constant for ONNX Reshape
Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(shape);
}

/// Pattern to convert MatMul to Reshape -> XFEConv -> Reshape
struct MatMulToXFEConvPattern : public OpRewritePattern<ONNXMatMulOp> {
  using OpRewritePattern<ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
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

    // Weight must be a 2D constant [K, N] -- reject batched/higher-dim MatMul.
    if (weightShape.size() != 2)
      return failure();

    auto weightConstOp = weight.getDefiningOp<ONNXConstantOp>();
    if (!weightConstOp)
      return failure();

    // Verify MatMul shape: input [D1, D2, ..., Dn, K], weight [K, N]
    if (inputShape.empty() || inputShape.back() != weightShape[0])
      return failure();

    auto convShapesOpt = computeConvShapes(inputShape, weightShape);
    if (!convShapesOpt)
      return failure();
    ConvShapes convShapes = *convShapesOpt;

    auto resultType = cast<RankedTensorType>(matMulOp.getResult().getType());
    Type outputElementType = resultType.getElementType();
    Type inputElementType = inputType.getElementType();

    // Create first Reshape: [D1, D2, ..., Dn, K] -> [M, H, W, C]
    auto reshape1OutputType =
        RankedTensorType::get(convShapes.inputShape, inputElementType);
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

    // Transpose [K, N] -> [N, K].
    // If weight has per-axis quant on axis 1 (N dim), after transpose it
    // moves to axis 0.
    auto transposedElemType = remapPerAxisQuantDim(weightElementType, 0);
    auto transposedWeightType = RankedTensorType::get(
        {weightShape[1], weightShape[0]}, transposedElemType);
    auto permAttr = rewriter.getI64ArrayAttr({1, 0});
    Value transposedWeight = rewriter.create<ONNXTransposeOp>(
        loc, transposedWeightType, weight, permAttr);

    // Reshape to [N, H, W, C] for XFEConv.  Axis 0 (N=C_out) stays axis 0.
    auto convWeightElemType = remapPerAxisQuantDim(weightElementType, 0);
    convWeightType =
        RankedTensorType::get(convShapes.weightShape, convWeightElemType);
    Value convWeight = rewriter.create<ONNXReshapeOp>(
        loc, convWeightType, transposedWeight, weightShapeConst);

    // Create XFEConv
    // XFEConv expects: input [M, H, W, C], weight [C_out, H', W', C]
    // Output: [M, H/H', W/W', C_out]
    int64_t outputH = convShapes.inputShape[1] / convShapes.stride[0];
    int64_t outputW = convShapes.inputShape[2] / convShapes.stride[1];
    SmallVector<int64_t> convOutputShape = {
        convShapes.inputShape[0], outputH, outputW, convShapes.weightShape[0]};

    auto convOutputType =
        RankedTensorType::get(convOutputShape, outputElementType);

    // Create attributes for XFEConv
    auto autoPadAttr = rewriter.getStringAttr("NOTSET");
    auto stridesAttr = rewriter.getI64ArrayAttr(convShapes.stride);
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(
        {convShapes.weightShape[1], convShapes.weightShape[2]});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto dilationsAttr = rewriter.getI64ArrayAttr({1, 1});
    auto groupAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 1, /*isSigned=*/true));

    // Create none value for bias
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value noneBias = onnxBuilder.none();

    // Create XFEConv operation
    auto convOp =
        rewriter.create<XFEConvOp>(loc, convOutputType, reshape1Output,
            convWeight, noneBias, rewriter.getStringAttr("NONE"), autoPadAttr,
            dilationsAttr, groupAttr, kernelShapeAttr, padsAttr, stridesAttr);

    // Transfer onnx_node_name attribute from MatMul to XFEConv
    transferOnnxNodeName(matMulOp, convOp);

    // Create second Reshape: [M, H/H', W/W', C_out] -> [D1, D2, ..., Dn, N]
    // Original output shape: [D1, D2, ..., Dn, N]
    SmallVector<int64_t> outputShape;
    for (size_t i = 0; i < inputShape.size() - 1; ++i) {
      outputShape.push_back(inputShape[i]);
    }
    outputShape.push_back(weightShape.back()); // N

    auto reshape2OutputType =
        RankedTensorType::get(outputShape, outputElementType);
    auto shapeConst2 = createShapeConstant(rewriter, loc, outputShape);
    Value reshape2Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape2OutputType, convOp.getResult(), shapeConst2);

    // Replace MatMul with the final Reshape output
    rewriter.replaceOp(matMulOp, reshape2Output);
    return success();
  }
};

/// Pattern to convert GEMM to Reshape -> XFEConv -> Reshape
/// GEMM: Y = alpha * (A^T if transA) * (B^T if transB) + beta * C
/// Handles transA and transB when 0 or 1.
struct GemmToXFEConvPattern : public OpRewritePattern<ONNXGemmOp> {
  using OpRewritePattern<ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    auto loc = gemmOp.getLoc();

    // Get inputs
    Value A = gemmOp.getA();
    Value B = gemmOp.getB();
    Value C = gemmOp.getC(); // Optional bias

    // Get types
    auto aType = dyn_cast<RankedTensorType>(A.getType());
    auto bType = dyn_cast<RankedTensorType>(B.getType());

    if (!aType || !bType || !aType.hasStaticShape() ||
        !bType.hasStaticShape()) {
      return failure();
    }

    int64_t transA = gemmOp.getTransA();
    int64_t transB = gemmOp.getTransB();
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();

    // GEMM requires 2D inputs; B must be a constant.
    if (aShape.size() != 2 || bShape.size() != 2)
      return failure();

    auto bConstOp = B.getDefiningOp<ONNXConstantOp>();
    if (!bConstOp)
      return failure();

    int64_t K_A = (transA == 0) ? aShape[1] : aShape[0];
    int64_t K_B = (transB == 0) ? bShape[0] : bShape[1];
    if (K_A != K_B)
      return failure();

    // Handle alpha: if alpha != 1.0, we need to scale the weight
    // For now, only handle alpha = 1.0
    // TODO: Handle alpha != 1.0 by scaling the weight
    float alpha = gemmOp.getAlpha().convertToFloat();
    if (alpha != 1.0f) {
      return failure(); // TODO: Handle non-unit alpha
    }

    // Compute convolution shapes
    // Treat A as input [M, K] -> reshape to [M, 1, 1, K]
    // Treat B as weight [K, N] -> reshape to [N, 1, 1, K]
    int64_t M = (transA == 0) ? aShape[0] : aShape[1];
    int64_t N = (transB == 0) ? bShape[1] : bShape[0];
    SmallVector<int64_t> effectiveAShape = {M, K_A};
    SmallVector<int64_t> effectiveBShape = {K_B, N};
    auto convShapesOpt = computeConvShapes(effectiveAShape, effectiveBShape);
    if (!convShapesOpt)
      return failure();
    ConvShapes convShapes = *convShapesOpt;

    // Use original op's result element type for conv and final output.
    auto resultType = cast<RankedTensorType>(gemmOp.getResult().getType());
    Type outputElementType = resultType.getElementType();
    Type inputElementType = aType.getElementType();

    // Effective input A: [M, K]. Transpose if transA=1 (A is [K, M])
    Value effectiveA = A;
    if (transA != 0) {
      auto permAttr = rewriter.getI64ArrayAttr({1, 0});
      auto transposedAType =
          RankedTensorType::get({aShape[1], aShape[0]}, inputElementType);
      effectiveA =
          rewriter.create<ONNXTransposeOp>(loc, transposedAType, A, permAttr);
    }

    // Create first Reshape: [M, K] -> [M, H, W, C]
    auto reshape1OutputType =
        RankedTensorType::get(convShapes.inputShape, inputElementType);
    auto shapeConst1 =
        createShapeConstant(rewriter, loc, convShapes.inputShape);
    Value reshape1Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape1OutputType, effectiveA, shapeConst1);

    // Format weight: [K, N] -> transpose to [N, K] -> reshape to [N, H, W, C]
    // (or [N, K] -> reshape directly when transB=1)
    auto bElementType = bType.getElementType();
    auto convWeightType =
        RankedTensorType::get(convShapes.weightShape, bElementType);
    auto weightShapeConst =
        createShapeConstant(rewriter, loc, convShapes.weightShape);

    Value convWeight;
    if (transB != 0) {
      // B is [N, K], per-axis quant on axis 0 (N=C_out) stays axis 0
      // after reshape to [N, H, W, C].
      auto convWeightElemType = remapPerAxisQuantDim(bElementType, 0);
      convWeightType =
          RankedTensorType::get(convShapes.weightShape, convWeightElemType);
      convWeight = rewriter.create<ONNXReshapeOp>(
          loc, convWeightType, B, weightShapeConst);
    } else {
      // Transpose [K, N] -> [N, K].
      // Per-axis quant on axis 1 (N) moves to axis 0.
      auto transposedElemType = remapPerAxisQuantDim(bElementType, 0);
      auto transposedBType =
          RankedTensorType::get({bShape[1], bShape[0]}, transposedElemType);
      auto permAttr = rewriter.getI64ArrayAttr({1, 0});
      Value transposedB =
          rewriter.create<ONNXTransposeOp>(loc, transposedBType, B, permAttr);
      // Reshape to [N, H, W, C] for XFEConv. Axis 0 stays axis 0.
      auto convWeightElemType = remapPerAxisQuantDim(bElementType, 0);
      convWeightType =
          RankedTensorType::get(convShapes.weightShape, convWeightElemType);
      convWeight = rewriter.create<ONNXReshapeOp>(
          loc, convWeightType, transposedB, weightShapeConst);
    }

    // Create XFEConv
    int64_t outputH = convShapes.inputShape[1] / convShapes.stride[0];
    int64_t outputW = convShapes.inputShape[2] / convShapes.stride[1];
    SmallVector<int64_t> convOutputShape = {
        convShapes.inputShape[0], outputH, outputW, convShapes.weightShape[0]};

    auto convOutputType =
        RankedTensorType::get(convOutputShape, outputElementType);

    // Create attributes for XFEConv
    auto autoPadAttr = rewriter.getStringAttr("NOTSET");
    auto stridesAttr = rewriter.getI64ArrayAttr(convShapes.stride);
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(
        {convShapes.weightShape[1], convShapes.weightShape[2]});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    auto dilationsAttr = rewriter.getI64ArrayAttr({1, 1});
    auto groupAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 1, /*isSigned=*/true));

    // Handle bias: if C is provided and beta != 0, use it as bias
    // Otherwise use none
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value bias;
    float beta = gemmOp.getBeta().convertToFloat();
    if (isa<NoneType>(C.getType()) || beta == 0.0f) {
      bias = onnxBuilder.none();
    } else {
      // For beta != 1.0, we would need to scale C
      // For now, only handle beta = 1.0
      if (beta != 1.0f) {
        return failure(); // TODO: Handle non-unit beta
      }
      bias = C;
    }

    // Create XFEConv operation
    auto convOp =
        rewriter.create<XFEConvOp>(loc, convOutputType, reshape1Output,
            convWeight, bias, rewriter.getStringAttr("NONE"), autoPadAttr,
            dilationsAttr, groupAttr, kernelShapeAttr, padsAttr, stridesAttr);

    // Transfer onnx_node_name attribute from GEMM to XFEConv
    transferOnnxNodeName(gemmOp, convOp);

    // Create second Reshape: [M, H/H', W/W', C_out] -> [M, N]
    SmallVector<int64_t> outputShape = {M, N};
    auto reshape2OutputType =
        RankedTensorType::get(outputShape, outputElementType);
    auto shapeConst2 = createShapeConstant(rewriter, loc, outputShape);
    Value reshape2Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape2OutputType, convOp.getResult(), shapeConst2);

    // Replace GEMM with the final Reshape output
    rewriter.replaceOp(gemmOp, reshape2Output);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ConvertMatMulToXFEConvPass
    : public PassWrapper<ConvertMatMulToXFEConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "convert-matmul-to-xfe-conv";
  }
  StringRef getDescription() const override {
    return "Convert MatMul and GEMM operations to XFEConv operations";
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Add patterns
    patterns.add<MatMulToXFEConvPattern>(context);
    patterns.add<GemmToXFEConvPattern>(context);

    GreedyRewriteConfig config;
    onnx_mlir::ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertMatMulToXFEConvPass() {
  return std::make_unique<ConvertMatMulToXFEConvPass>();
}

} // namespace onnx_mlir
