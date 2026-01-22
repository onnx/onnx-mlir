// Copyright (C) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// This pass converts pool operations with kernel size 1 and stride > kernel
// to equivalent resize (downsample) operations with mode "nearest".
// This is a translation of the XIR TransferPoolFixToDownsampleFixPass to MLIR.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "transfer-pool-to-downsample"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Extract int64_t values from an ArrayAttr
SmallVector<int64_t> extractI64ArrayAttr(ArrayAttr attr) {
  SmallVector<int64_t> result;
  if (!attr)
    return result;
  for (auto elem : attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(elem)) {
      result.push_back(intAttr.getInt());
    }
  }
  return result;
}

/// Overload for optional ArrayAttr
SmallVector<int64_t> extractI64ArrayAttr(std::optional<ArrayAttr> attr) {
  if (!attr)
    return {};
  return extractI64ArrayAttr(*attr);
}

/// Check if all padding values are zero
bool hasNonZeroPadding(ArrayRef<int64_t> pads) {
  for (int64_t pad : pads) {
    if (pad != 0)
      return true;
  }
  return false;
}

/// Check if pool operation qualifies for conversion to downsample
/// Conditions from XIR pass:
/// - kernel_w == 1 AND kernel_h == 1
/// - kernel_w < stride_w AND kernel_h < stride_h
/// - All padding values must be zero
bool isEligibleForDownsampleConversion(ArrayRef<int64_t> kernel,
    ArrayRef<int64_t> stride, ArrayRef<int64_t> pads) {
  if (kernel.size() < 2 || stride.size() < 2)
    return false;

  // Only match 2D pooling (kernel size 2)
  if (kernel.size() != 2)
    return false;

  // Padding must be zero - resize doesn't support padding semantics
  if (hasNonZeroPadding(pads))
    return false;

  int64_t kernelH = kernel[0];
  int64_t kernelW = kernel[1];
  int64_t strideH = stride[0];
  int64_t strideW = stride[1];

  // Original XIR condition: kernel == 1 AND kernel < stride for both dimensions
  return (kernelH == 1) && (kernelW == 1) && (kernelH < strideH) &&
         (kernelW < strideW);
}

/// Calculate output shape for resize operation (NCHW format)
SmallVector<int64_t> calculateResizeOutputShape(ArrayRef<int64_t> inputShape,
    ArrayRef<int64_t> kernel, ArrayRef<int64_t> stride) {
  SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());

  // For NCHW format: spatial dimensions are at indices 2 and 3
  if (inputShape.size() >= 4 && kernel.size() >= 2 && stride.size() >= 2) {
    // H dimension
    outputShape[2] = inputShape[2] / stride[0];
    // W dimension
    outputShape[3] = inputShape[3] / stride[1];
  }

  return outputShape;
}

/// Calculate output shape for resize operation (NHWC/channel-last format)
SmallVector<int64_t> calculateResizeOutputShapeChannelLast(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> kernel,
    ArrayRef<int64_t> stride) {
  SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());

  // For NHWC format: spatial dimensions are at indices 1 and 2
  if (inputShape.size() >= 4 && kernel.size() >= 2 && stride.size() >= 2) {
    // H dimension
    outputShape[1] = inputShape[1] / stride[0];
    // W dimension
    outputShape[2] = inputShape[2] / stride[1];
  }

  return outputShape;
}

/// Create an empty ROI constant for resize op
Value createEmptyRoiConstant(PatternRewriter &rewriter, Location loc) {
  auto roiType = RankedTensorType::get({0}, rewriter.getF32Type());
  auto roiAttr = DenseElementsAttr::get(roiType, ArrayRef<float>{});

  return rewriter.create<ONNXConstantOp>(loc, roiType, Attribute(), roiAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

/// Create output sizes constant for resize op
Value createSizesConstant(PatternRewriter &rewriter, Location loc,
    ArrayRef<int64_t> outputShape) {
  auto sizesType = RankedTensorType::get(
      {static_cast<int64_t>(outputShape.size())}, rewriter.getI64Type());
  auto sizesAttr = DenseElementsAttr::get(sizesType, outputShape);

  return rewriter.create<ONNXConstantOp>(loc, sizesType, Attribute(), sizesAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

//===----------------------------------------------------------------------===//
// Pattern: Pool → Resize (nearest) - Templated for MaxPool and AvgPool
//===----------------------------------------------------------------------===//

template <typename PoolOpT>
struct TransferPoolToDownsamplePattern : public OpRewritePattern<PoolOpT> {
  using OpRewritePattern<PoolOpT>::OpRewritePattern;

  static constexpr const char *getPoolName() {
    if constexpr (std::is_same_v<PoolOpT, ONNXMaxPoolSingleOutOp>)
      return "MaxPool";
    else
      return "AvgPool";
  }

  LogicalResult matchAndRewrite(PoolOpT poolOp,
      PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(poolOp.getX().getType());
    auto outputType = dyn_cast<RankedTensorType>(poolOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    // Extract kernel, stride, and pads attributes
    auto kernel = extractI64ArrayAttr(poolOp.getKernelShape());
    auto stride = extractI64ArrayAttr(poolOp.getStrides());
    auto pads = extractI64ArrayAttr(poolOp.getPads());

    if (kernel.empty() || stride.empty()) {
      return failure();
    }

    // Check eligibility: kernel == 1 AND kernel < stride AND no padding
    if (!isEligibleForDownsampleConversion(kernel, stride, pads)) {
      LLVM_DEBUG(llvm::dbgs()
                 << getPoolName() << " not eligible for downsample conversion: "
                 << "kernel=[" << kernel[0] << ", " << kernel[1] << "] "
                 << "stride=[" << stride[0] << ", " << stride[1] << "]"
                 << (hasNonZeroPadding(pads) ? " (has non-zero padding)" : "")
                 << "\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Converting " << getPoolName() << " to Resize (nearest): "
               << "kernel=[" << kernel[0] << ", " << kernel[1] << "] "
               << "stride=[" << stride[0] << ", " << stride[1] << "]\n");

    // Create constants for resize op
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t> expectedOutputShape =
        calculateResizeOutputShape(inputShape, kernel, stride);

    Value roiConst = createEmptyRoiConstant(rewriter, loc);
    // Use none for scales - we specify sizes instead (cannot have both)
    Value scalesNone = rewriter.create<ONNXNoneOp>(loc).getResult();
    Value sizesConst = createSizesConstant(rewriter, loc, expectedOutputShape);

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    // Create resize op with mode "nearest"
    auto resizeOp = rewriter.create<ONNXResizeOp>(loc, outputType,
        poolOp.getX(),
        /*roi=*/roiConst,
        /*scales=*/scalesNone,
        /*sizes=*/sizesConst,
        /*antialias=*/IntegerAttr::get(si64Type, 0),
        /*axes=*/nullptr,
        /*coordinate_transformation_mode=*/
        rewriter.getStringAttr("asymmetric"),
        /*cubic_coeff_a=*/rewriter.getF32FloatAttr(-0.75f),
        /*exclude_outside=*/IntegerAttr::get(si64Type, 0),
        /*extrapolation_value=*/rewriter.getF32FloatAttr(0.0f),
        /*keep_aspect_ratio_policy=*/rewriter.getStringAttr("stretch"),
        /*mode=*/rewriter.getStringAttr("nearest"),
        /*nearest_mode=*/rewriter.getStringAttr("floor"));

    rewriter.replaceOp(poolOp, resizeOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Successfully converted " << getPoolName()
                            << " to Resize (nearest)\n");

    return success();
  }
};

// Type aliases for clarity
using TransferMaxPoolToDownsamplePattern =
    TransferPoolToDownsamplePattern<ONNXMaxPoolSingleOutOp>;
using TransferAvgPoolToDownsamplePattern =
    TransferPoolToDownsamplePattern<ONNXAveragePoolOp>;

//===----------------------------------------------------------------------===//
// Pattern: ONNX XFE Pool → ONNX XFE Resize (channel-last layout)
//===----------------------------------------------------------------------===//

template <typename ONNXXFEPoolOpT>
struct TransferONNXXFEPoolToDownsamplePattern
    : public OpRewritePattern<ONNXXFEPoolOpT> {
  using OpRewritePattern<ONNXXFEPoolOpT>::OpRewritePattern;

  static constexpr const char *getPoolName() {
    if constexpr (std::is_same_v<ONNXXFEPoolOpT, XFEMaxPoolOp>)
      return "ONNX_XFE_MaxPool";
    else
      return "ONNX_XFE_AvgPool";
  }

  LogicalResult matchAndRewrite(ONNXXFEPoolOpT poolOp,
      PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(poolOp.getX().getType());
    auto outputType = dyn_cast<RankedTensorType>(poolOp.getY().getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    // Extract kernel, stride, and pads attributes
    auto kernel = extractI64ArrayAttr(poolOp.getKernelShape());
    auto stride = extractI64ArrayAttr(poolOp.getStrides());
    auto pads = extractI64ArrayAttr(poolOp.getPads());

    if (kernel.empty() || stride.empty()) {
      return failure();
    }

    // Check eligibility: kernel == 1 AND kernel < stride AND no padding
    if (!isEligibleForDownsampleConversion(kernel, stride, pads)) {
      LLVM_DEBUG(llvm::dbgs()
                 << getPoolName() << " not eligible for downsample conversion: "
                 << "kernel=[" << kernel[0] << ", " << kernel[1] << "] "
                 << "stride=[" << stride[0] << ", " << stride[1] << "]"
                 << (hasNonZeroPadding(pads) ? " (has non-zero padding)" : "")
                 << "\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Converting " << getPoolName()
               << " to ONNX_XFE_Resize (nearest): "
               << "kernel=[" << kernel[0] << ", " << kernel[1] << "] "
               << "stride=[" << stride[0] << ", " << stride[1] << "]\n");

    // Create constants for resize op (channel-last format)
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t> expectedOutputShape =
        calculateResizeOutputShapeChannelLast(inputShape, kernel, stride);

    // Create ROI constant (empty)
    Value roiConst = createEmptyRoiConstant(rewriter, loc);
    // Use none for scales - we specify sizes instead
    Value scalesNone = rewriter.create<ONNXNoneOp>(loc).getResult();
    // Create sizes constant
    Value sizesConst = createSizesConstant(rewriter, loc, expectedOutputShape);

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    // Create XFE resize op with mode "nearest"
    auto resizeOp = rewriter.create<XFEResizeOp>(loc, outputType,
        poolOp.getX(),
        /*roi=*/roiConst,
        /*scales=*/scalesNone,
        /*sizes=*/sizesConst,
        /*antialias=*/IntegerAttr::get(si64Type, 0),
        /*axes=*/nullptr,
        /*coordinate_transformation_mode=*/
        rewriter.getStringAttr("asymmetric"),
        /*cubic_coeff_a=*/rewriter.getF32FloatAttr(-0.75f),
        /*exclude_outside=*/IntegerAttr::get(si64Type, 0),
        /*extrapolation_value=*/rewriter.getF32FloatAttr(0.0f),
        /*keep_aspect_ratio_policy=*/rewriter.getStringAttr("stretch"),
        /*mode=*/rewriter.getStringAttr("nearest"),
        /*nearest_mode=*/rewriter.getStringAttr("floor"));

    rewriter.replaceOp(poolOp, resizeOp.getY());

    LLVM_DEBUG(llvm::dbgs() << "Successfully converted " << getPoolName()
                            << " to ONNX_XFE_Resize (nearest)\n");

    return success();
  }
};

// Type aliases for ONNX XFE patterns
using TransferONNXXFEMaxPoolToDownsamplePattern =
    TransferONNXXFEPoolToDownsamplePattern<XFEMaxPoolOp>;
using TransferONNXXFEAvgPoolToDownsamplePattern =
    TransferONNXXFEPoolToDownsamplePattern<XFEAveragePoolOp>;

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass to transfer pool-fix operations to downsample-fix operations.
/// This matches pool operations with kernel=1 and stride > kernel,
/// converting them to resize (nearest neighbor downsample) operations.
struct TransferPoolFixToDownsampleFixPass
    : public PassWrapper<TransferPoolFixToDownsampleFixPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-pool-fix-to-downsample-fix";
  }
  StringRef getDescription() const override {
    return "Convert pool operations with kernel=1 and stride>kernel to resize "
           "(downsample) operations";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Add patterns for ONNX MaxPool and AvgPool
    patterns.add<TransferMaxPoolToDownsamplePattern>(ctx);
    patterns.add<TransferAvgPoolToDownsamplePattern>(ctx);

    // Add patterns for ONNX XFE MaxPool and AvgPool (channel-last layout)
    patterns.add<TransferONNXXFEMaxPoolToDownsamplePattern>(ctx);
    patterns.add<TransferONNXXFEAvgPoolToDownsamplePattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
            config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferPoolFixToDownsampleFixPass() {
  return std::make_unique<TransferPoolFixToDownsampleFixPass>();
}

} // namespace onnx_mlir
