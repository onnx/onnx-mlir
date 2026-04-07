// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "src/Interface/TensorNameInference.hpp"

namespace mlir {

#include "src/Interface/TensorNameInference.cpp.inc"

}

#include "src/Dialect/ONNX/TensorName.hpp"

using namespace mlir;

namespace onnx_mlir {

std::unique_ptr<Transform>
ReshapeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto reshapeOp = cast<ONNXReshapeOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(reshapeOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  return std::make_unique<ReshapeTransform>(
      inType.getShape(), outType.getShape());
}

std::unique_ptr<Transform>
TransposeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto transposeOp = cast<ONNXTransposeOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(transposeOp.getOperand().getType());
  auto outType = cast<RankedTensorType>(transposeOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  auto perm = Transform::arrayToVector(transposeOp.getPermAttr());

  return std::make_unique<TransposeTransform>(
      inType.getShape(), perm, outType.getShape());
}

std::unique_ptr<Transform> PadOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto padOp = cast<ONNXPadOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(padOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(padOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  // Only mode = "constant" is supported
  if (padOp.getMode() != "constant")
    return nullptr;
  auto constOp = padOp.getConstantValue().getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return nullptr;
  Attribute constant = constOp.getValueAttr();
  if (!constant)
    return nullptr;

  // Validate if pads is constant
  auto pads = Transform::valToVector(padOp.getPads());
  if (pads.size() == 0)
    return nullptr;

  auto splitAt = pads.size() / 2;
  auto starts = SmallVector<int64_t>(pads.begin(), pads.begin() + splitAt);
  auto ends = SmallVector<int64_t>(pads.begin() + splitAt, pads.end());
  auto axes = Transform::axesToVector(padOp.getAxes(), inType.getRank());

  return std::make_unique<PadTransform>(
      inType.getShape(), starts, ends, axes, constant, outType.getShape());
}

std::unique_ptr<Transform> SliceOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto sliceOp = cast<ONNXSliceOp>(op);

  // Validate if shapes are static
  auto inType = cast<RankedTensorType>(sliceOp.getOperand(0).getType());
  auto outType = cast<RankedTensorType>(sliceOp.getResult().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape())
    return nullptr;

  // Validate if starts & ends are constant and steps is always 1
  auto starts = Transform::valToVector(sliceOp.getStarts());
  auto ends = Transform::valToVector(sliceOp.getEnds());
  auto steps = Transform::valToVector(sliceOp.getSteps());
  if (starts.size() == 0 || ends.size() == 0 ||
      llvm::any_of(steps, [](int64_t s) { return s != 1; }))
    return nullptr;

  auto inShape = inType.getShape();
  auto axes = Transform::axesToVector(sliceOp.getAxes(), inShape.size());

  // Clip end values
  for (const auto &[ax, en] : llvm::zip_equal(axes, ends))
    en = std::min(en, inShape[ax]);

  return std::make_unique<SliceTransform>(
      inShape, starts, ends, axes, outType.getShape());
}

void registerTensorNameInferenceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension<ONNXDialect>(
      +[](MLIRContext *ctx, ONNXDialect * /*dialect*/) {
        ONNXTransposeOp::attachInterface<TransposeOpTensorNameInference>(*ctx);
        ONNXReshapeOp::attachInterface<ReshapeOpTensorNameInference>(*ctx);
        ONNXPadOp::attachInterface<PadOpTensorNameInference>(*ctx);
        ONNXSliceOp::attachInterface<SliceOpTensorNameInference>(*ctx);
      });
}

} // namespace onnx_mlir
