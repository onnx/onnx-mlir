// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// ConvertXFEConvToDepthwiseConvPass
//
// This pass converts XFEConv to XCOMPILERDepthwiseConv when the convolution
// is a depthwise convolution (group == input_channels).
//
// Depthwise convolution is a special case where each input channel is
// convolved with its own set of filters independently. This is more
// computationally efficient than standard convolution.
//
// Conditions for conversion:
//   1. XFEConv has group attribute set
//   2. group == number of input channels (C)
//   3. Weight shape has channel multiplier == 1 (depthwise)
//
// XFEConv uses NHWC layout with OHWI weights:
//   - Input X: [N, H, W, C] for 2D or [N, D, H, W, C] for 3D
//   - Weight W: [C_out, kH, kW, C_in/group] (OHWI) for 2D
//               [C_out, kD, kH, kW, C_in/group] (ODHWI) for 3D
//   - Bias B: [C] (optional)
//   - Output Y: [N, outH, outW, C] for 2D or [N, outD, outH, outW, C] for 3D
//
// XCOMPILERDepthwiseConv uses IHWO weight format for downstream XIR:
//   - Weight W: [C_in/group, kH, kW, C_out] (IHWO) for 2D
//
// This pass physically transposes the constant weight data from OHWI → IHWO
// during the conversion so that the downstream XFEToXIR pass does not need
// to introduce an additional transpose op.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include <iostream>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

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

/// Check if the XFEConv operation is a depthwise convolution.
/// Depthwise conv: group == input_channels and channel_multiplier == 1
///
/// XFEConv uses:
///   - Input X: NHWC layout [N, H, W, C_in] for 2D, [N, D, H, W, C_in] for 3D
///   - Weight W: OHWI layout [C_out, kH, kW, C_in/group] for 2D
///                           [C_out, kD, kH, kW, C_in/group] for 3D
/// For depthwise: C_out = C_in, group = C_in, so C_in/group = 1
bool isDepthwiseConv(XFEConvOp convOp) {
  // Get group attribute (defaults to 1)
  int64_t group = convOp.getGroup();

  // Get input X shape to determine number of channels
  Value X = convOp.getX();
  auto xType = mlir::dyn_cast<RankedTensorType>(X.getType());
  if (!xType || !xType.hasRank())
    return false;

  int64_t xRank = xType.getRank();
  // Must be 4D (2D conv) or 5D (3D conv) with NHWC layout
  if (xRank != 4 && xRank != 5)
    return false;

  // In NHWC layout, channels are in the last dimension
  int64_t inputChannels = xType.getDimSize(xRank - 1);
  if (inputChannels == ShapedType::kDynamic)
    return false;

  // Check if group == input_channels (depthwise condition)
  if (group != inputChannels)
    return false;

  // Get weight W shape to verify it's a depthwise conv
  // Weight format is OHWI: [C_out, kH, kW, C_in/group]
  Value W = convOp.getW();
  auto wType = mlir::dyn_cast<RankedTensorType>(W.getType());
  if (!wType || !wType.hasRank())
    return false;

  int64_t wRank = wType.getRank();
  // Weight rank should match input rank
  if (wRank != xRank)
    return false;

  // For OHWI format: [C_out, spatial_dims..., C_in/group]
  // C_out is the first dimension
  int64_t outputChannels = wType.getDimSize(0);

  // C_in/group is the last dimension (should be 1 for depthwise)
  int64_t cInPerGroup = wType.getDimSize(wRank - 1);
  if (cInPerGroup != ShapedType::kDynamic && cInPerGroup != 1)
    return false;

  // For depthwise, output channels should equal input channels
  return outputChannels == ShapedType::kDynamic ||
         outputChannels == inputChannels;
}

/// Extract kernel shape from weights or attribute
/// Weight format is OHWI: [C_out, kH, kW, C_in/group] for 2D
///                        [C_out, kD, kH, kW, C_in/group] for 3D
SmallVector<int64_t> getKernelShape(XFEConvOp convOp) {
  // First try to get from attribute
  if (auto kernelShapeAttr = convOp.getKernelShapeAttr()) {
    SmallVector<int64_t> kernelShape;
    for (auto attr : kernelShapeAttr)
      kernelShape.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    return kernelShape;
  }

  // Otherwise infer from weight shape (OHWI format)
  Value W = convOp.getW();
  auto wType = mlir::dyn_cast<RankedTensorType>(W.getType());
  if (!wType || !wType.hasRank())
    return {};

  // Weight shape OHWI: [C_out, kH, kW, C_in/group] for 2D
  // Kernel dimensions are at indices 1 to wRank-2 (skip C_out and C_in/group)
  SmallVector<int64_t> kernelShape;
  int64_t wRank = wType.getRank();
  for (int64_t i = 1; i < wRank - 1; ++i)
    kernelShape.push_back(wType.getDimSize(i));

  return kernelShape;
}

//===----------------------------------------------------------------------===//
// Pattern: ConvertXFEConvToDepthwiseConv
//===----------------------------------------------------------------------===//

/// Pattern to convert XFEConv to XCOMPILERDepthwiseConv when conditions match
struct ConvertXFEConvToDepthwiseConvPattern
    : public OpRewritePattern<XFEConvOp> {
  using OpRewritePattern<XFEConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      XFEConvOp convOp, PatternRewriter &rewriter) const override {
    // Check if this is a depthwise convolution
    if (!isDepthwiseConv(convOp))
      return failure();

    Location loc = convOp.getLoc();

    // Get operands
    Value X = convOp.getX();
    Value W = convOp.getW();
    Value B = convOp.getB();

    // Get or compute kernel shape
    SmallVector<int64_t> kernelShape = getKernelShape(convOp);
    if (kernelShape.empty())
      return failure();
    // Create kernel_shape attribute (required for DepthwiseConv)
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);

    // Get optional attributes
    auto autoPadAttr = convOp.getAutoPadAttr();
    auto stridesAttr = convOp.getStridesAttr();
    auto padsAttr = convOp.getPadsAttr();
    auto dilationsAttr = convOp.getDilationsAttr();

    // Physically transpose weight constant data from OHWI to IHWO.
    // This avoids inserting a runtime transpose op.
    auto wType = mlir::dyn_cast<RankedTensorType>(W.getType());
    if (!wType || !wType.hasRank())
      return failure();
    auto wShape = wType.getShape();
    int64_t wRank = wType.getRank();

    SmallVector<int64_t> newWShape;
    SmallVector<int64_t> perm;
    if (wRank == 4) {
      // 2D: OHWI {O,H,W,I} → IHWO {I,H,W,O} = perm {3,1,2,0}
      newWShape = {wShape[3], wShape[1], wShape[2], wShape[0]};
      perm = {3, 1, 2, 0};
    } else if (wRank == 5) {
      // 3D: ODHWI {O,D,H,W,I} → IDHWO {I,D,H,W,O} = perm {4,1,2,3,0}
      newWShape = {wShape[4], wShape[1], wShape[2], wShape[3], wShape[0]};
      perm = {4, 1, 2, 3, 0};
    } else {
      return failure();
    }

    // Get the weight's constant ElementsAttr
    ElementsAttr wElements = onnx_mlir::getElementAttributeFromONNXValue(W);
    if (!wElements)
      return failure(); // Weight must be a constant

    // Use OnnxElementsAttrBuilder to transpose the constant data in-place
    SmallVector<uint64_t> permU64(perm.begin(), perm.end());
    onnx_mlir::OnnxElementsAttrBuilder elemBuilder(rewriter.getContext());
    ElementsAttr transposedElements = elemBuilder.transpose(wElements, permU64);
    // Convert to DenseElementsAttr to avoid DisposableElementsAttr
    DenseElementsAttr transposedDense =
        elemBuilder.toDenseElementsAttr(transposedElements);

    // Create new constant with transposed data
    auto newWConst =
        rewriter.create<ONNXConstantOp>(loc, Attribute(), transposedDense);

    // Preserve the original element type (including quant type) with
    // the new IHWO shape. The DenseElementsAttr uses storage types
    // (e.g. i8) but the value flowing through the graph may carry a
    // quant type (e.g. !quant.uniform<i8:f32, ...>).
    auto newWType = RankedTensorType::get(newWShape, wType.getElementType());
    newWConst.getResult().setType(newWType);

    // Get the activation attribute from the source XFEConv op
    auto activationAttr = convOp.getActivationAttr();

    // Create the DepthwiseConv operation with IHWO weights
    auto depthwiseConv = rewriter.create<XCOMPILERDepthwiseConvOp>(loc,
        convOp.getResult().getType(), // Output type
        X,                            // Input
        newWConst,                    // Weights (IHWO, data transposed)
        B,                            // Bias (optional)
        activationAttr,               // activation
        autoPadAttr,                  // auto_pad
        dilationsAttr,                // dilations
        kernelShapeAttr,              // kernel_shape (required)
        padsAttr,                     // pads
        stridesAttr                   // strides
    );

    // Transfer onnx_node_name attribute from XFEConv to XCOMPILERDepthwiseConv
    transferOnnxNodeName(convOp, depthwiseConv);

    // Replace XFEConv with DepthwiseConv
    rewriter.replaceOp(convOp, depthwiseConv.getResult());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct ConvertXFEConvToDepthwiseConvPass
    : public PassWrapper<ConvertXFEConvToDepthwiseConvPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "convert-xfe-conv-to-depthwise-conv";
  }
  StringRef getDescription() const override {
    return "Convert XFEConv to XCOMPILERDepthwiseConv when group == "
           "input_channels (depthwise convolution)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertXFEConvToDepthwiseConvPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertXFEConvToDepthwiseConvPass() {
  return std::make_unique<ConvertXFEConvToDepthwiseConvPass>();
}

} // namespace onnx_mlir
