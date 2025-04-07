/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- STFT.cpp - Lowering STFT Op ------------------------===//
//
// Copyright 2025 AMD.
//
// =============================================================================
//
// This file provides definition of ONNX dialect STFT shape inference.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

template <>
LogicalResult ONNXGenericSTFTOpShapeHelper<ONNXSTFTOp>::computeShape() {
  ONNXSTFTOp::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get inputs
  auto signal = operandAdaptor.getSignal();
  auto frameStep = operandAdaptor.getFrameStep();
  auto window = operandAdaptor.getWindow();
  auto frameLength = operandAdaptor.getFrameLength();

  // Get input types
  auto windowType = window.getType();
  auto frameLengthType = frameLength.getType();

  if (!hasShapeAndRank(signal))
    return failure();
  auto oneSided = operandAdaptor.getOnesided() == 1;

  // ---------------------- Get FFT window/frame size --------------------------
  if (mlir::isa<NoneType>(windowType) && mlir::isa<NoneType>(frameLengthType)) {
    // Neither window nor frame length is provided.
    // In this case, return error.
    return op->emitError("STFT requires either a window or frame length");
  }
  // Get the FFT window size
  IndexExpr fftWindowSizeIE;
  if (!mlir::isa<NoneType>(windowType)) {
    // Window is provided.
    // Check if window is a tensor of rank 1
    if (!hasShapeAndRank(window) || getRank(window.getType()) != 1)
      return op->emitError("Window must be a 1D tensor");
    auto windowSizeIE = createIE->getShapeAsDim(window, 0);

    if (!mlir::isa<NoneType>(frameLengthType)) {
      // Both window and frame length are provided.
      auto frameLengthIE =
          createIE->getIntAsSymbol(operandAdaptor.getFrameLength());
      if (windowSizeIE.isLiteral() && frameLengthIE.isLiteral()) {
        // window size must be equal to frame length
        if (windowSizeIE.getLiteral() != frameLengthIE.getLiteral())
          return op->emitError("Window size must be equal to frame length");
        fftWindowSizeIE = windowSizeIE;
      } else if (frameLengthIE.isLiteral()) {
        // Frame length is known but window size is unknown.
        // In this case, use frame length as window size.
        fftWindowSizeIE = frameLengthIE;
      } else {
        // Frame Length is unknown
        // In this case, use window size as FFt window size.
        fftWindowSizeIE = windowSizeIE;
      }
    } else {
      // Only window is provided so use window size as FFT window size.
      fftWindowSizeIE = windowSizeIE;
    }
  } else if (!mlir::isa<NoneType>(frameLengthType)) {
    // Frame length is provided, but window is not.
    // In this case, use frame length as window size.
    fftWindowSizeIE = createIE->getIntAsSymbol(operandAdaptor.getFrameLength());
  } else {
    // Neither window nor frame length is provided.
    // In this case, return error.
    return op->emitError("STFT requires either a window or frame length");
  }
  // ---------------------------------------------------------------------------

  // ----------------------- Compute Output Dims -------------------------------
  auto frameStepIE = createIE->getIntAsSymbol(frameStep);
  auto batchSizeIE = createIE->getShapeAsDim(signal, 0);
  auto signalLengthIE = createIE->getShapeAsDim(signal, 1);

  // compute number of frames
  LiteralIndexExpr one(1);
  auto numFramesIE =
      (signalLengthIE - fftWindowSizeIE).floorDiv(frameStepIE) + one;

  DimsExpr outputDims;
  outputDims.emplace_back(batchSizeIE);
  outputDims.emplace_back(numFramesIE);
  if (oneSided) {
    outputDims.emplace_back(fftWindowSizeIE.floorDiv(2) + one);
  } else {
    outputDims.emplace_back(fftWindowSizeIE);
  }
  outputDims.emplace_back(LitIE(2));
  // ---------------------------------------------------------------------------

  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// STFT Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSTFTOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  Type elementType =
      mlir::cast<ShapedType>(getSignal().getType()).getElementType();
  ONNXSTFTOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXGenericSTFTOpShapeHelper<ONNXSTFTOp>;
} // namespace onnx_mlir
