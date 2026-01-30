
#include "XCOMPILERVerify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {

LogicalResult XCOMPILERFusedEltwiseOpVerify(Operation *op) {

  auto fusedEltwiseOp = dyn_cast<XCOMPILERFusedEltwiseOp>(op);
  if (!fusedEltwiseOp)
    return failure();

  StringRef nonlinear = fusedEltwiseOp.getNonlinear();
  StringRef type = fusedEltwiseOp.getType();

  // leakyrelu_alpha, prelu_in, prelu_shift exist only if nonlinear == LEAKYRELU
  bool isLeakyRelu = (nonlinear == "LEAKYRELU");
  if (fusedEltwiseOp.getLeakyreluAlphaAttr() && !isLeakyRelu)
    return op->emitOpError(
        "'leakyrelu_alpha' is only valid when nonlinear is LEAKYRELU");
  if (fusedEltwiseOp.getPreluInAttr() && !isLeakyRelu)
    return op->emitOpError(
        "'prelu_in' is only valid when nonlinear is LEAKYRELU");
  if (fusedEltwiseOp.getPreluShiftAttr() && !isLeakyRelu)
    return op->emitOpError(
        "'prelu_shift' is only valid when nonlinear is LEAKYRELU");

  // clip_min, clip_max exist only if type == CLIP
  bool isClip = (type == "CLIP");
  if (fusedEltwiseOp.getClipMinAttr() && !isClip)
    return op->emitOpError("'clip_min' is only valid when type is CLIP");
  if (fusedEltwiseOp.getClipMaxAttr() && !isClip)
    return op->emitOpError("'clip_max' is only valid when type is CLIP");

  // nonlinear_in_scales and nonlinear_in_zeropoints exist only if nonlinear is
  // not NONE
  bool nonlinearIsNone = (nonlinear == "NONE");
  if (fusedEltwiseOp.getNonlinearInScalesAttr() && nonlinearIsNone)
    return op->emitOpError(
        "'nonlinear_in_scales' is only valid when nonlinear is not NONE");
  if (fusedEltwiseOp.getNonlinearInZeropointsAttr() && nonlinearIsNone)
    return op->emitOpError(
        "'nonlinear_in_zeropoints' is only valid when nonlinear is not NONE");

  return success();
}

LogicalResult XCOMPILERDepthwiseConvOpVerify(Operation *op) {
  auto convOp = dyn_cast<XCOMPILERDepthwiseConvOp>(op);
  if (!convOp)
    return failure();

  // Verify input X is 4D [N, H, W, C] or 5D [N, D, H, W, C] (NHWC layout)
  Value X = convOp.getX();
  int64_t xRank = 0;
  int64_t inputChannels = ShapedType::kDynamic;
  if (auto xType = mlir::dyn_cast<ShapedType>(X.getType())) {
    if (xType.hasRank()) {
      xRank = xType.getRank();
      if (xRank != 4 && xRank != 5)
        return op->emitOpError(
                   "input X must be 4D [N, H, W, C] or 5D [N, D, H, W, C], got rank ")
               << xRank;
      // Channel is last dimension in NHWC layout
      if (!xType.isDynamicDim(xRank - 1))
        inputChannels = xType.getDimSize(xRank - 1);
    }
  }

  bool is3D = (xRank == 5);
  size_t numSpatialDims = is3D ? 3 : 2;

  // Verify weight W has correct rank for OHWI layout (consistent with XFEConv)
  // 2D: [C, kH, kW, 1], 3D: [C, kD, kH, kW, 1]
  Value W = convOp.getW();
  if (auto wType = mlir::dyn_cast<ShapedType>(W.getType())) {
    if (wType.hasRank()) {
      int64_t expectedWRank = numSpatialDims + 2; // C + spatial dims + multiplier
      if (wType.getRank() != expectedWRank)
        return op->emitOpError("weight W must be ")
               << expectedWRank << "D tensor for " << (is3D ? "3D" : "2D")
               << " convolution, got rank " << wType.getRank();

      // For depthwise conv with OHWI, last dimension (C_in/group) should be 1
      int64_t multiplierIdx = wType.getRank() - 1;
      if (!wType.isDynamicDim(multiplierIdx) &&
          wType.getDimSize(multiplierIdx) != 1)
        return op->emitOpError(
                   "depthwise conv weight channel multiplier (last dim) should "
                   "be 1, got ")
               << wType.getDimSize(multiplierIdx);

      // Verify input channels match weight output channels (first dim in OHWI weight)
      int64_t wChannelIdx = 0;  // C_out is first dimension in OHWI format
      if (inputChannels != ShapedType::kDynamic &&
          !wType.isDynamicDim(wChannelIdx)) {
        if (inputChannels != wType.getDimSize(wChannelIdx))
          return op->emitOpError("input channels (")
                 << inputChannels << ") must match weight channels ("
                 << wType.getDimSize(wChannelIdx) << ")";
      }
    }
  }

  // Verify bias B (if provided) is 1D [C]
  Value B = convOp.getB();
  if (B && !mlir::isa<NoneType>(B.getType())) {
    if (auto bType = mlir::dyn_cast<ShapedType>(B.getType())) {
      if (bType.hasRank() && bType.getRank() != 1)
        return op->emitOpError("bias B must be 1D tensor [C], got rank ")
               << bType.getRank();
    }
  }

  // Verify kernel_shape has correct number of elements
  if (auto kernelShapeAttr = op->getAttrOfType<ArrayAttr>("kernel_shape")) {
    if (xRank > 0 && kernelShapeAttr.size() != numSpatialDims)
      return op->emitOpError("kernel_shape must have ")
             << numSpatialDims << " elements for " << (is3D ? "3D" : "2D")
             << " convolution, got " << kernelShapeAttr.size();
  }

  // Verify strides has correct number of elements (if provided)
  if (auto stridesAttr = op->getAttrOfType<ArrayAttr>("strides")) {
    if (xRank > 0 && stridesAttr.size() != numSpatialDims)
      return op->emitOpError("strides must have ")
             << numSpatialDims << " elements, got " << stridesAttr.size();
  }

  // Verify dilations has correct number of elements (if provided)
  if (auto dilationsAttr = op->getAttrOfType<ArrayAttr>("dilations")) {
    if (xRank > 0 && dilationsAttr.size() != numSpatialDims)
      return op->emitOpError("dilations must have ")
             << numSpatialDims << " elements, got " << dilationsAttr.size();
  }

  // Verify pads has correct number of elements (if provided)
  // pads format: [begin0, begin1, ..., end0, end1, ...]
  if (auto padsAttr = op->getAttrOfType<ArrayAttr>("pads")) {
    size_t expectedPads = numSpatialDims * 2;
    if (xRank > 0 && padsAttr.size() != expectedPads)
      return op->emitOpError("pads must have ")
             << expectedPads << " elements, got " << padsAttr.size();
  }

  // Verify auto_pad is valid
  auto autoPad = convOp.getAutoPad();
  if (autoPad != "NOTSET" && autoPad != "SAME_UPPER" &&
      autoPad != "SAME_LOWER" && autoPad != "VALID")
    return op->emitOpError("auto_pad must be one of NOTSET, SAME_UPPER, "
                           "SAME_LOWER, VALID, got '")
           << autoPad << "'";

  return success();
}

} // namespace mlir
