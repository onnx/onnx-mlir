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

  auto inShape =
      cast<RankedTensorType>(reshapeOp.getData().getType()).getShape();
  auto outShape =
      cast<RankedTensorType>(reshapeOp.getResult().getType()).getShape();

  return std::make_unique<ReshapeTransform>(inShape, outShape);
}

std::unique_ptr<Transform>
TransposeOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto transposeOp = cast<ONNXTransposeOp>(op);

  auto inShape =
      cast<RankedTensorType>(transposeOp.getData().getType()).getShape();
  auto perm = Transform::arrayToVector(transposeOp.getPermAttr());
  auto outShape =
      cast<RankedTensorType>(transposeOp.getResult().getType()).getShape();

  return std::make_unique<TransposeTransform>(inShape, perm, outShape);
}

std::unique_ptr<Transform> PadOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto padOp = cast<ONNXPadOp>(op);

  // Only mode = "constant" is supported
  if (padOp.getMode() != "constant")
    return {};
  auto constOp = padOp.getConstantValue().getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return {};
  Attribute constant = constOp.getValueAttr();
  if (!constant)
    return {};

  auto inShape = cast<RankedTensorType>(padOp.getData().getType()).getShape();
  auto pads = Transform::valToVector(padOp.getPads());
  auto splitAt = pads.size() / 2;
  auto starts = SmallVector<int64_t>(pads.begin(), pads.begin() + splitAt);
  auto ends = SmallVector<int64_t>(pads.begin() + splitAt, pads.end());
  auto axes = Transform::axesToVector(padOp.getAxes(), inShape.size());
  auto outShape =
      cast<RankedTensorType>(padOp.getOutput().getType()).getShape();

  return std::make_unique<PadTransform>(
      inShape, starts, ends, axes, constant, outShape);
}

std::unique_ptr<Transform> SliceOpTensorNameInference::inferTensorNameTransform(
    mlir::Operation *op) const {
  auto sliceOp = cast<ONNXSliceOp>(op);

  auto inShape = cast<RankedTensorType>(sliceOp.getData().getType()).getShape();
  auto starts = Transform::valToVector(sliceOp.getStarts());
  auto ends = Transform::valToVector(sliceOp.getEnds());
  auto axes = Transform::axesToVector(sliceOp.getAxes(), inShape.size());
  auto outShape =
      cast<RankedTensorType>(sliceOp.getResult().getType()).getShape();

  // Clip end values
  for (const auto &[ax, en] : llvm::zip_equal(axes, ends))
    en = std::min(en, inShape[ax]);

  return std::make_unique<SliceTransform>(
      inShape, starts, ends, axes, outShape);
}

} // namespace onnx_mlir
