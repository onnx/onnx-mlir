// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

/// Try to fold a transpose ([K,N]->[N,K]) and reshape ([N,K]->[N,1,1,K])
/// directly into the constant, returning the folded constant value.
/// Returns nullptr if the weight is not a constant.
static Value tryFoldTransposeReshapeConst(PatternRewriter &rewriter,
    Location loc, Value weight, ArrayRef<uint64_t> perm,
    ArrayRef<int64_t> reshapedShape, Type outputElemType) {
  auto constOp = weight.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return nullptr;
  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr || !isa<DenseElementsAttr, DisposableElementsAttr>(valueAttr))
    return nullptr;

  auto wElements = mlir::cast<ElementsAttr>(valueAttr);

  onnx_mlir::OnnxElementsAttrBuilder elemBuilder(rewriter.getContext());
  ElementsAttr transposedElements = elemBuilder.transpose(wElements, perm);
  ElementsAttr reshapedElements =
      elemBuilder.reshape(transposedElements, reshapedShape);
  DenseElementsAttr denseFolded =
      elemBuilder.toDenseElementsAttr(reshapedElements);

  auto newConstOp =
      rewriter.create<ONNXConstantOp>(loc, Attribute(), denseFolded);
  auto resultType = RankedTensorType::get(reshapedShape, outputElemType);
  newConstOp.getResult().setType(resultType);
  return newConstOp.getResult();
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

/// Try to find a following Add that can be absorbed as conv bias.
/// Checks the direct user of the MatMul result for an Add with a constant.
/// Returns the Add op and the bias constant value, or {nullptr, nullptr}.
static std::pair<ONNXAddOp, Value> findFusibleBiasAdd(
    ONNXMatMulOp matMulOp, int64_t outputChannels) {
  if (!matMulOp.getResult().hasOneUse())
    return {nullptr, nullptr};

  Operation *user = *matMulOp.getResult().getUsers().begin();
  auto addOp = dyn_cast<ONNXAddOp>(user);
  if (!addOp)
    return {nullptr, nullptr};

  // Identify which Add operand is the bias constant
  Value biasVal = nullptr;
  if (addOp.getA() == matMulOp.getResult()) {
    biasVal = addOp.getB();
  } else {
    biasVal = addOp.getA();
  }

  auto biasConstOp = biasVal.getDefiningOp<ONNXConstantOp>();
  if (!biasConstOp)
    return {nullptr, nullptr};

  // Bias element count must match output channels.
  auto biasType = dyn_cast<RankedTensorType>(biasVal.getType());
  if (!biasType)
    return {nullptr, nullptr};
  int64_t biasElements = 1;
  for (auto d : biasType.getShape())
    biasElements *= d;
  if (biasElements != outputChannels)
    return {nullptr, nullptr};

  return {addOp, biasVal};
}

/// Re-quantize bias into the conv accumulation domain (int32).
///
/// Dequantizes each element to float, then re-quantizes into the conv
/// accumulation scale (x_scale * w_scale):
///   new_bias[i] = round( (orig_bias[i] - bias_zp) * bias_scale
///                        / (x_scale * w_scale) )
///
/// The result is a 1D int32 constant with quant type
///   !quant.uniform<i32:f32, x_scale * w_scale : 0>
/// so that XFEToXIRDialectPass extracts the correct b_scale / b_zero_point.
///
/// Falls back to a simple 1D reshape when the input is not quantized.
static Value requantizeBiasForConv(PatternRewriter &rewriter, Location loc,
    Value biasVal, Value input, Value weight, int64_t N) {
  auto biasType = dyn_cast<RankedTensorType>(biasVal.getType());
  if (!biasType)
    return biasVal;

  auto biasConstOp = biasVal.getDefiningOp<ONNXConstantOp>();
  if (!biasConstOp)
    return biasVal;
  auto denseAttr = dyn_cast<DenseElementsAttr>(biasConstOp.getValueAttr());
  if (!denseAttr)
    return biasVal;

  auto biasElemType = biasType.getElementType();
  auto biasQType = dyn_cast<quant::QuantizedType>(biasElemType);

  // Non-quantized path: just reshape to 1D [N].
  if (!biasQType) {
    if (biasType.getRank() == 1 && biasType.getShape()[0] == N)
      return biasVal;
    auto storageElemType = denseAttr.getType().getElementType();
    auto newStorageType = RankedTensorType::get({N}, storageElemType);
    auto newAttr = denseAttr.reshape(newStorageType);
    auto newResultType = RankedTensorType::get({N}, biasElemType);
    auto valueNamedAttr = rewriter.getNamedAttr("value", newAttr);
    return rewriter
        .create<ONNXConstantOp>(loc, newResultType, mlir::ValueRange{},
            mlir::ArrayRef<mlir::NamedAttribute>{valueNamedAttr})
        .getResult();
  }

  // Quantized path: re-quantize bias into accumulation domain.
  // Matches golden xcompiler TransferQDQMatMulToConv2dPass: uses scale[0]
  // for all channels (per-tensor treatment even for per-axis types).
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  if (!inputType || !weightType)
    return biasVal;

  auto inputQType = dyn_cast<quant::QuantizedType>(inputType.getElementType());
  if (!inputQType)
    return biasVal;

  // Extract scale[0] and zp[0] from any quantized type (per-tensor or
  // per-axis).
  auto getScale0 = [](quant::QuantizedType qt) -> double {
    if (auto pt = dyn_cast<quant::UniformQuantizedType>(qt))
      return pt.getScale();
    if (auto pa = dyn_cast<quant::UniformQuantizedPerAxisType>(qt))
      return pa.getScales().empty() ? 1.0 : pa.getScales()[0];
    return 1.0;
  };
  auto getZP0 = [](quant::QuantizedType qt) -> int64_t {
    if (auto pt = dyn_cast<quant::UniformQuantizedType>(qt))
      return pt.getZeroPoint();
    if (auto pa = dyn_cast<quant::UniformQuantizedPerAxisType>(qt))
      return pa.getZeroPoints().empty() ? 0 : pa.getZeroPoints()[0];
    return 0;
  };

  double inputScale = getScale0(inputQType);

  auto weightQType =
      dyn_cast<quant::QuantizedType>(weightType.getElementType());
  if (!weightQType)
    return biasVal;
  double weightScale = getScale0(weightQType);

  double biasScale = getScale0(biasQType);
  int64_t biasZP = getZP0(biasQType);
  double accumScale = inputScale * weightScale;

  // Flatten original bias data and re-quantize to int32.
  //   new_bias[i] = round( (raw[i] - biasZP) * biasScale / accumScale )
  auto flatStorageType =
      RankedTensorType::get({N}, denseAttr.getType().getElementType());
  auto flatAttr = denseAttr.reshape(flatStorageType);
  SmallVector<int32_t> newBiasData;
  newBiasData.reserve(N);

  unsigned bitWidth = biasQType.getStorageType().getIntOrFloatBitWidth();
  bool isSigned = biasQType.isSigned();
  for (int64_t i = 0; i < N; ++i) {
    int64_t raw = 0;
    if (bitWidth <= 8) {
      if (isSigned)
        raw = flatAttr.getValues<int8_t>()[i];
      else
        raw = flatAttr.getValues<uint8_t>()[i];
    } else if (bitWidth <= 16) {
      if (isSigned)
        raw = flatAttr.getValues<int16_t>()[i];
      else
        raw = flatAttr.getValues<uint16_t>()[i];
    } else {
      if (isSigned)
        raw = flatAttr.getValues<int32_t>()[i];
      else
        raw = static_cast<int64_t>(flatAttr.getValues<uint32_t>()[i]);
    }
    double floatBias = static_cast<double>(raw - biasZP) * biasScale;
    newBiasData.push_back(
        static_cast<int32_t>(std::round(floatBias / accumScale)));
  }

  // Create int32 dense attribute.
  auto i32Type = rewriter.getIntegerType(32);
  auto newStorageType = RankedTensorType::get({N}, i32Type);
  auto newDenseAttr = DenseElementsAttr::get(
      newStorageType, llvm::ArrayRef<int32_t>(newBiasData));

  // Result type: !quant.uniform<i32:f32, accumScale:0>
  auto newBiasQType2 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, i32Type, rewriter.getF32Type(),
      accumScale, /*zeroPoint=*/0, std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max());
  auto newResultType = RankedTensorType::get({N}, newBiasQType2);

  auto valueNamedAttr = rewriter.getNamedAttr("value", newDenseAttr);
  return rewriter
      .create<ONNXConstantOp>(loc, newResultType, mlir::ValueRange{},
          mlir::ArrayRef<mlir::NamedAttribute>{valueNamedAttr})
      .getResult();
}

/// Pattern to convert MatMul to Reshape -> XFEConv -> Reshape
/// Also fuses a following Add(MatMul, constant) into the conv bias.
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

    // Check for a fusible bias Add following the MatMul
    int64_t N = weightShape[1];
    auto [addOp, biasVal] = findFusibleBiasAdd(matMulOp, N);
    Operation *opToReplace =
        addOp ? addOp.getOperation() : matMulOp.getOperation();
    Type finalOutputElemType =
        addOp ? cast<RankedTensorType>(addOp.getResult().getType())
                    .getElementType()
              : outputElementType;

    // Create first Reshape: [D1, D2, ..., Dn, K] -> [M, H, W, C]
    auto reshape1OutputType =
        RankedTensorType::get(convShapes.inputShape, inputElementType);
    auto shapeConst1 =
        createShapeConstant(rewriter, loc, convShapes.inputShape);
    Value reshape1Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape1OutputType, input, shapeConst1);

    // Format weight: [K, N] -> transpose to [N, K] -> reshape to [N, 1, 1, K].
    auto weightElementType = weightType.getElementType();
    auto convWeightElemType = remapPerAxisQuantDim(weightElementType, 0);
    auto convWeightType =
        RankedTensorType::get(convShapes.weightShape, convWeightElemType);

    // Try to fold transpose+reshape directly into the constant.
    Value convWeight = tryFoldTransposeReshapeConst(rewriter, loc, weight,
        {1, 0}, convShapes.weightShape, convWeightElemType);

    if (!convWeight) {
      // Non-constant weight: emit Transpose + Reshape ops.
      auto weightShapeConst =
          createShapeConstant(rewriter, loc, convShapes.weightShape);
      auto transposedWeightType = RankedTensorType::get(
          {weightShape[1], weightShape[0]}, convWeightElemType);
      auto permAttr = rewriter.getI64ArrayAttr({1, 0});
      Value transposedWeight = rewriter.create<ONNXTransposeOp>(
          loc, transposedWeightType, weight, permAttr);
      convWeight = rewriter.create<ONNXReshapeOp>(
          loc, convWeightType, transposedWeight, weightShapeConst);
    }

    // Create XFEConv
    // XFEConv expects: input [M, H, W, C], weight [C_out, H', W', C]
    // Output: [M, H/H', W/W', C_out]
    int64_t outputH = convShapes.inputShape[1] / convShapes.stride[0];
    int64_t outputW = convShapes.inputShape[2] / convShapes.stride[1];
    SmallVector<int64_t> convOutputShape = {
        convShapes.inputShape[0], outputH, outputW, convShapes.weightShape[0]};

    auto convOutputType =
        RankedTensorType::get(convOutputShape, finalOutputElemType);

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

    // Prepare bias: fuse from Add or use none
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value bias;
    if (addOp && biasVal) {
      bias = requantizeBiasForConv(rewriter, loc, biasVal, input, weight, N);
    } else {
      bias = onnxBuilder.none();
    }

    // Create XFEConv operation
    auto convOp = rewriter.create<XFEConvOp>(loc, convOutputType,
        reshape1Output, convWeight, bias, rewriter.getStringAttr("NONE"),
        autoPadAttr, dilationsAttr, groupAttr, kernelShapeAttr,
        /*leakyrelu_alpha=*/FloatAttr(), padsAttr,
        /*prelu_in=*/IntegerAttr(), /*prelu_shift=*/IntegerAttr(), stridesAttr);

    // Transfer onnx_node_name attribute from MatMul to XFEConv
    transferOnnxNodeName(matMulOp, convOp);

    // Create second Reshape: [M, H/H', W/W', C_out] -> [D1, D2, ..., Dn, N]
    // Original output shape: [D1, D2, ..., Dn, N]
    SmallVector<int64_t> outputShape;
    for (size_t i = 0; i < inputShape.size() - 1; ++i) {
      outputShape.push_back(inputShape[i]);
    }
    outputShape.push_back(N);

    auto reshape2OutputType =
        RankedTensorType::get(outputShape, finalOutputElemType);
    auto shapeConst2 = createShapeConstant(rewriter, loc, outputShape);
    Value reshape2Output = rewriter.create<ONNXReshapeOp>(
        loc, reshape2OutputType, convOp.getResult(), shapeConst2);

    // Replace MatMul (or MatMul+Add) with the final Reshape output.
    // When fusing bias, replace the Add op; the MatMul becomes dead and
    // will be cleaned up by the greedy rewrite driver.
    rewriter.replaceOp(opToReplace, reshape2Output);
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
      // Transpose [K, N] -> [N, K] then reshape to [N, H, W, C].
      // Per-axis quant on axis 1 (N) moves to axis 0.
      auto convWeightElemType = remapPerAxisQuantDim(bElementType, 0);
      convWeightType =
          RankedTensorType::get(convShapes.weightShape, convWeightElemType);

      convWeight = tryFoldTransposeReshapeConst(
          rewriter, loc, B, {1, 0}, convShapes.weightShape, convWeightElemType);

      if (!convWeight) {
        auto transposedBType =
            RankedTensorType::get({bShape[1], bShape[0]}, convWeightElemType);
        auto permAttr = rewriter.getI64ArrayAttr({1, 0});
        Value transposedB =
            rewriter.create<ONNXTransposeOp>(loc, transposedBType, B, permAttr);
        convWeight = rewriter.create<ONNXReshapeOp>(
            loc, convWeightType, transposedB, weightShapeConst);
      }
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
    auto convOp = rewriter.create<XFEConvOp>(loc, convOutputType,
        reshape1Output, convWeight, bias, rewriter.getStringAttr("NONE"),
        autoPadAttr, dilationsAttr, groupAttr, kernelShapeAttr,
        /*leakyrelu_alpha=*/FloatAttr(), padsAttr,
        /*prelu_in=*/IntegerAttr(), /*prelu_shift=*/IntegerAttr(), stridesAttr);

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
