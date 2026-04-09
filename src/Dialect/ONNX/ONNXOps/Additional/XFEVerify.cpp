// IMPLEMENT YOUR VERIFY LOGIC HERE
// Move to: src/Dialect/ONNX/ONNXOps/Additional/XFEVerify.cpp

#include "XFEVerify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {

namespace {

// Fused activation attributes (xfe_ops_schema Conv): activation, optional
// leakyrelu_alpha, prelu_in, prelu_shift. Aligns with FuseConvActivationPass /
// NormalizeConvActivationPass.

static bool isValidXFEFusedActivation(StringRef activation) {
  return activation == "NONE" || activation == "RELU" ||
         activation == "LEAKYRELU" || activation == "PRELU" ||
         activation == "HSIGMOID" || activation == "RELU6" ||
         activation == "SIGMOID";
}

template <typename ConvLikeOp>
static LogicalResult verifyXFEFusedConvActivationAttrs(ConvLikeOp convOp) {
  StringRef activation = convOp.getActivation();
  if (!isValidXFEFusedActivation(activation))
    return convOp->emitOpError(
               "'activation' must be one of NONE, RELU, LEAKYRELU, PRELU, "
               "HSIGMOID, RELU6, SIGMOID, got '")
           << activation << "'";

  bool isPrelu = activation == "PRELU";

  if (convOp.getLeakyreluAlphaAttr() && activation != "LEAKYRELU" && !isPrelu)
    return convOp->emitOpError("'leakyrelu_alpha' is only valid when "
                               "activation is LEAKYRELU or PRELU");

  if (convOp.getPreluInAttr() && !isPrelu)
    return convOp->emitOpError(
        "'prelu_in' is only valid when activation is PRELU");

  if (convOp.getPreluShiftAttr() && !isPrelu)
    return convOp->emitOpError(
        "'prelu_shift' is only valid when activation is PRELU");

  if (isPrelu && (!convOp.getPreluInAttr() || !convOp.getPreluShiftAttr()))
    return convOp->emitOpError(
        "activation PRELU requires both 'prelu_in' and 'prelu_shift'");

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Channel-wise (per-axis) quantization verifier for all XFE ops.
//
// XFE ops use channel-last layout:
//   Data tensors (X / input): [N, spatial..., C]  => channel axis = rank - 1
//   Weight tensors (W):       [C_out, spatial..., C_in/group] => axis 0
//
// If a tensor carries UniformQuantizedPerAxisType, its quantizedDimension
// must match the channel axis expected by the layout.
//===----------------------------------------------------------------------===//

static LogicalResult verifyPerAxisQuantAxis(
    Operation *op, Value tensor, int64_t expectedAxis, StringRef tensorName) {
  auto shapedType = mlir::dyn_cast<ShapedType>(tensor.getType());
  if (!shapedType)
    return success();
  auto perAxisType =
      dyn_cast<quant::UniformQuantizedPerAxisType>(shapedType.getElementType());
  if (!perAxisType)
    return success();
  int32_t quantAxis = perAxisType.getQuantizedDimension();
  if (static_cast<int64_t>(quantAxis) != expectedAxis)
    return op->emitError() << tensorName << " per-axis quantization axis is "
                           << quantAxis
                           << ", but channel-last layout requires axis "
                           << expectedAxis;
  return success();
}

LogicalResult XFEChannelWiseQuantizationVerify(Operation *op) {
  // MatMulBias: A and B quantized along their last axis
  if (auto matmulOp = dyn_cast<XFEMatMulBiasOp>(op)) {
    Value A = matmulOp.getA();
    Value B = matmulOp.getB();
    if (hasShapeAndRank(A)) {
      int64_t aRank = mlir::cast<ShapedType>(A.getType()).getRank();
      if (failed(verifyPerAxisQuantAxis(op, A, aRank - 1, "input A")))
        return failure();
    }
    if (hasShapeAndRank(B)) {
      int64_t bRank = mlir::cast<ShapedType>(B.getType()).getRank();
      if (failed(verifyPerAxisQuantAxis(op, B, bRank - 1, "input B")))
        return failure();
    }
    return success();
  }

  // Conv / ConvTranspose: X channel axis = rank-1, W output-channel axis = 0
  if (auto convOp = dyn_cast<XFEConvOp>(op)) {
    Value X = convOp.getX();
    Value W = convOp.getW();
    if (hasShapeAndRank(X)) {
      int64_t xRank = mlir::cast<ShapedType>(X.getType()).getRank();
      if (failed(verifyPerAxisQuantAxis(op, X, xRank - 1, "input X")))
        return failure();
    }
    if (hasShapeAndRank(W)) {
      if (failed(verifyPerAxisQuantAxis(op, W, 0, "weight W")))
        return failure();
    }
    return success();
  }
  if (auto convTransposeOp = dyn_cast<XFEConvTransposeOp>(op)) {
    Value X = convTransposeOp.getX();
    Value W = convTransposeOp.getW();
    if (hasShapeAndRank(X)) {
      int64_t xRank = mlir::cast<ShapedType>(X.getType()).getRank();
      if (failed(verifyPerAxisQuantAxis(op, X, xRank - 1, "input X")))
        return failure();
    }
    if (hasShapeAndRank(W)) {
      if (failed(verifyPerAxisQuantAxis(op, W, 0, "weight W")))
        return failure();
    }
    return success();
  }

  // Single-input ops: channel axis = rank - 1
  Value inputTensor;
  StringRef inputName;
  if (auto poolOp = dyn_cast<XFEAveragePoolOp>(op)) {
    inputTensor = poolOp.getX();
    inputName = "input X";
  } else if (auto poolOp = dyn_cast<XFEMaxPoolOp>(op)) {
    inputTensor = poolOp.getX();
    inputName = "input X";
  } else if (auto poolOp = dyn_cast<XFEGlobalAveragePoolOp>(op)) {
    inputTensor = poolOp.getX();
    inputName = "input X";
  } else if (auto poolOp = dyn_cast<XFEGlobalMaxPoolOp>(op)) {
    inputTensor = poolOp.getX();
    inputName = "input X";
  } else if (auto normOp = dyn_cast<XFEInstanceNormalizationOp>(op)) {
    inputTensor = normOp.getInput();
    inputName = "input";
  } else if (auto d2sOp = dyn_cast<XFEDepthToSpaceOp>(op)) {
    inputTensor = d2sOp.getInput();
    inputName = "input";
  } else if (auto s2dOp = dyn_cast<XFESpaceToDepthOp>(op)) {
    inputTensor = s2dOp.getInput();
    inputName = "input";
  } else if (auto resizeOp = dyn_cast<XFEResizeOp>(op)) {
    inputTensor = resizeOp.getX();
    inputName = "input X";
  } else {
    return success();
  }

  if (!hasShapeAndRank(inputTensor))
    return success();
  int64_t rank = mlir::cast<ShapedType>(inputTensor.getType()).getRank();
  return verifyPerAxisQuantAxis(op, inputTensor, rank - 1, inputName);
}

LogicalResult XFEMatMulBiasOpVerify(Operation *op) {
  auto matmulOp = dyn_cast<XFEMatMulBiasOp>(op);
  if (!matmulOp)
    return failure();
  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEConvOpVerify(Operation *op) {
  auto convOp = dyn_cast<XFEConvOp>(op);
  if (!convOp)
    return failure();

  if (failed(verifyXFEFusedConvActivationAttrs(convOp)))
    return failure();

  Value X = convOp.getX();
  Value W = convOp.getW();
  if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto wType = mlir::cast<ShapedType>(W.getType());
  auto xShape = xType.getShape();
  auto wShape = wType.getShape();

  if (xShape.size() < 3 || wShape.size() < 3 || xShape.size() != wShape.size())
    return op->emitError("ConvChannelLast requires matching rank tensors with "
                         "at least 3 dimensions");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEConvTransposeOpVerify(Operation *op) {
  auto convTransposeOp = dyn_cast<XFEConvTransposeOp>(op);
  if (!convTransposeOp)
    return failure();

  if (failed(verifyXFEFusedConvActivationAttrs(convTransposeOp)))
    return failure();

  Value X = convTransposeOp.getX();
  Value W = convTransposeOp.getW();
  if (!hasShapeAndRank(X) || !hasShapeAndRank(W))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto wType = mlir::cast<ShapedType>(W.getType());
  auto xShape = xType.getShape();
  auto wShape = wType.getShape();

  if (xShape.size() < 3 || wShape.size() < 3 || xShape.size() != wShape.size())
    return op->emitError(
        "ConvTransposeChannelLast requires matching rank tensors with "
        "at least 3 dimensions");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEAveragePoolOpVerify(Operation *op) {
  auto poolOp = dyn_cast<XFEAveragePoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape();
  if (xShape.size() < 3)
    return op->emitError(
        "AveragePoolChannelLast requires at least 3D input tensor");

  int64_t numSpatialDims = static_cast<int64_t>(xShape.size()) - 2;
  auto kernelShapeAttr = poolOp.getKernelShape();
  if (!kernelShapeAttr.has_value() ||
      static_cast<int64_t>(kernelShapeAttr->size()) < numSpatialDims)
    return op->emitError(
        "kernel_shape attribute required with matching spatial dimensions");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEMaxPoolOpVerify(Operation *op) {
  auto poolOp = dyn_cast<XFEMaxPoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape();
  if (xShape.size() < 3)
    return op->emitError(
        "MaxPoolChannelLast requires at least 3D input tensor");

  int64_t numSpatialDims = static_cast<int64_t>(xShape.size()) - 2;
  auto kernelShapeAttr = poolOp.getKernelShape();
  if (!kernelShapeAttr.has_value() ||
      static_cast<int64_t>(kernelShapeAttr->size()) < numSpatialDims)
    return op->emitError(
        "kernel_shape attribute required with matching spatial dimensions");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEGlobalAveragePoolOpVerify(Operation *op) {
  auto poolOp = dyn_cast<XFEGlobalAveragePoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape();
  if (xShape.size() < 3)
    return op->emitError(
        "GlobalAveragePoolChannelLast requires at least 3D input tensor");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEGlobalMaxPoolOpVerify(Operation *op) {
  auto poolOp = dyn_cast<XFEGlobalMaxPoolOp>(op);
  if (!poolOp)
    return failure();

  Value X = poolOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  auto xShape = xType.getShape();
  if (xShape.size() < 3)
    return op->emitError(
        "GlobalMaxPoolChannelLast requires at least 3D input tensor");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEInstanceNormalizationOpVerify(Operation *op) {
  auto normOp = dyn_cast<XFEInstanceNormalizationOp>(op);
  if (!normOp)
    return failure();

  Value input = normOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  if (inputShape.size() < 3)
    return op->emitError(
        "InstanceNormalizationChannelLast requires at least 3D input tensor");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEDepthToSpaceOpVerify(Operation *op) {
  auto d2sOp = dyn_cast<XFEDepthToSpaceOp>(op);
  if (!d2sOp)
    return failure();

  Value input = d2sOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return op->emitError("DepthToSpaceChannelLast requires 4D input tensor");

  auto blocksizeAttr = d2sOp.getBlocksize();
  if (!blocksizeAttr.has_value())
    return op->emitError("blocksize attribute is required");

  int64_t blocksize = blocksizeAttr.value();
  if (blocksize <= 0)
    return op->emitError("blocksize must be positive");

  int64_t C = inputShape[3];
  int64_t blocksizeSq = blocksize * blocksize;
  if (C != ShapedType::kDynamic && C % blocksizeSq != 0)
    return op->emitError("input channels must be divisible by blocksize^2");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFESpaceToDepthOpVerify(Operation *op) {
  auto s2dOp = dyn_cast<XFESpaceToDepthOp>(op);
  if (!s2dOp)
    return failure();

  Value input = s2dOp.getInput();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 4)
    return op->emitError("SpaceToDepthChannelLast requires 4D input tensor");

  auto blocksizeAttr = s2dOp.getBlocksize();
  if (!blocksizeAttr.has_value())
    return op->emitError("blocksize attribute is required");

  int64_t blocksize = blocksizeAttr.value();
  if (blocksize <= 0)
    return op->emitError("blocksize must be positive");

  int64_t H = inputShape[1];
  int64_t W = inputShape[2];
  if (H != ShapedType::kDynamic && H % blocksize != 0)
    return op->emitError("input height must be divisible by blocksize");
  if (W != ShapedType::kDynamic && W % blocksize != 0)
    return op->emitError("input width must be divisible by blocksize");

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEResizeOpVerify(Operation *op) {
  auto resizeOp = dyn_cast<XFEResizeOp>(op);
  if (!resizeOp)
    return failure();

  Value X = resizeOp.getX();
  if (!hasShapeAndRank(X))
    return success();

  auto xType = mlir::cast<ShapedType>(X.getType());
  int64_t rank = xType.getRank();

  // Keep aligned with shape inference: if axes is present, skip additional
  // verification because shape inference currently does not support it.
  if (resizeOp.getAxes().has_value())
    return success();

  auto isAbsent = [](Value input) -> bool {
    if (isa<NoneType>(input.getType()))
      return true;
    if (auto shapedType = mlir::dyn_cast<ShapedType>(input.getType())) {
      return shapedType.hasStaticShape() && shapedType.getNumElements() == 0;
    }
    return false;
  };

  Value scales = resizeOp.getScales();
  Value sizes = resizeOp.getSizes();
  bool scalesIsAbsent = isAbsent(scales);
  bool sizesIsAbsent = isAbsent(sizes);

  if (!scalesIsAbsent) {
    DenseElementsAttr scalesAttr;
    if (auto defOp = scales.getDefiningOp<ONNXConstantOp>()) {
      if (auto valueAttr = defOp.getValue()) {
        scalesAttr = mlir::dyn_cast<DenseElementsAttr>(*valueAttr);
      }
    }
    if (scalesAttr && static_cast<int64_t>(scalesAttr.size()) != rank)
      return op->emitError("scales size must match input rank");
  } else if (!sizesIsAbsent) {
    DenseElementsAttr sizesAttr;
    if (auto defOp = sizes.getDefiningOp<ONNXConstantOp>()) {
      if (auto valueAttr = defOp.getValue()) {
        sizesAttr = mlir::dyn_cast<DenseElementsAttr>(*valueAttr);
      }
    }
    if (sizesAttr && static_cast<int64_t>(sizesAttr.size()) != rank)
      return op->emitError("sizes size must match input rank");
  }

  return XFEChannelWiseQuantizationVerify(op);
}

LogicalResult XFEBatchNormalizationOpVerify(Operation *op) {
  auto bnOp = dyn_cast<XFEBatchNormalizationOp>(op);
  if (!bnOp)
    return failure();

  Value input = bnOp.getX();
  if (!hasShapeAndRank(input))
    return success();

  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  if (inputShape.size() < 3)
    return op->emitError(
        "BatchNormalizationChannelLast requires at least 3D input tensor");

  return XFEChannelWiseQuantizationVerify(op);
}

} // namespace mlir