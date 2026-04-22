/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ AMDQuark.cpp - AMD Quark custom ops ---------------===//
//
// Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

template <typename AMDQuarkOpTy>
std::optional<int64_t> getNormalizedAxis(AMDQuarkOpTy *op) {
  const int64_t axis = op->getAxis();
  if (axis >= 0)
    return axis;
  const auto rankedType = dyn_cast<RankedTensorType>(op->getX().getType());
  if (!rankedType)
    return std::nullopt;
  return axis + rankedType.getRank();
}

LogicalResult AMDQuarkBFPQuantizeDequantizeOp::verify() {
  // Verify that the quantization mode is valid.
  const auto method = getBfpMethod();
  if (method != "to_bfp" && method != "to_bfp_prime") {
    return emitOpError("invalid bfp_method attribute value: " + method +
                       ". Supported values are 'to_bfp' and 'to_bfp_prime'.");
  }
  const int64_t roundingMode = getRoundingMode();
  if (roundingMode < 0 || roundingMode > 3) {
    return emitOpError(
        "invalid rounding_mode attribute value: " +
        std::to_string(roundingMode) +
        ". Supported values are 0 for rounding half away from zero, 1 for "
        "rounding half upward and 2 for rounding half to even.");
  }

  if (auto rankedType = dyn_cast<RankedTensorType>(getX().getType())) {
    const int64_t rank = rankedType.getRank();
    const int64_t axis = getAxis();
    if (axis < -rank || axis >= rank)
      return emitOpError("axis attribute value ")
             << axis << " is out of range [-" << rank << ", " << rank << ")";
  }

  return success();
}

namespace {
struct KnownConfig {
  StringRef method;
  int64_t axis;
  int64_t bit_width;
  int64_t block_size;
  int64_t rounding_mode;
  int64_t sub_block_size;
  int64_t sub_block_shift_bits;
};

[[nodiscard]] bool isKnownConfig(AMDQuarkBFPQuantizeDequantizeOp *op,
    const KnownConfig &config, bool ignoreAxis) {
  if (op->getBfpMethod() != config.method)
    return false;
  if (!ignoreAxis) {
    const auto normalizedAxis = op->getNormalizedAxis();
    if (!normalizedAxis || *normalizedAxis != config.axis)
      return false;
  }
  if (op->getBitWidth() != config.bit_width)
    return false;
  if (op->getBlockSize() != config.block_size)
    return false;
  if (op->getRoundingMode() != config.rounding_mode)
    return false;
  if (op->getBfpMethod() == "to_bfp_prime") {
    if (op->getSubBlockSize() != config.sub_block_size)
      return false;
    if (op->getSubBlockShiftBits() != config.sub_block_shift_bits)
      return false;
  }
  return true;
}
} // namespace

bool AMDQuarkBFPQuantizeDequantizeOp::isBFP16(bool ignoreAxis) {
  return isKnownConfig(this, {"to_bfp", 1, 16, 8, 2, 0, 0}, ignoreAxis);
}
bool AMDQuarkBFPQuantizeDequantizeOp::isMX4(bool ignoreAxis) {
  return isKnownConfig(this, {"to_bfp_prime", 1, 11, 16, 2, 2, 1}, ignoreAxis);
}
bool AMDQuarkBFPQuantizeDequantizeOp::isMX6(bool ignoreAxis) {
  return isKnownConfig(this, {"to_bfp_prime", 1, 13, 16, 2, 2, 1}, ignoreAxis);
}
bool AMDQuarkBFPQuantizeDequantizeOp::isMX9(bool ignoreAxis) {
  return isKnownConfig(this, {"to_bfp_prime", 1, 16, 16, 2, 2, 1}, ignoreAxis);
}

std::optional<int64_t> AMDQuarkBFPQuantizeDequantizeOp::getNormalizedAxis() {
  return ::getNormalizedAxis<AMDQuarkBFPQuantizeDequantizeOp>(this);
}

LogicalResult AMDQuarkBFPQuantizeDequantizeOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  return inferShapeForUnaryOps(this->getOperation());
}

// Common methods for
// - AMDQuarkExtendedQuantizeLinearOp
// - AMDQuarkExtendedDequantizeLinearOp

template <typename AMDExtendedQDQOpTy>
LogicalResult inferShapes(AMDExtendedQDQOpTy *op,
    std::function<void(Region &)> /*doShapeInference*/) {
  if (!mlir::dyn_cast<RankedTensorType>(op->getX().getType()))
    return success();
  Type elementType =
      mlir::cast<ShapedType>(op->getY().getType()).getElementType();
  ONNXUnaryOpShapeHelper shapeHelper(op->getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

template <typename AMDExtendedQDQOpTy>
LogicalResult verify(AMDExtendedQDQOpTy *op) {
  if (auto rankedType = dyn_cast<RankedTensorType>(op->getX().getType())) {
    const int64_t rank = rankedType.getRank();
    const int64_t axis = op->getAxis();
    if ((rank != 1) && (axis < -rank || axis >= rank))
      return op->emitOpError("axis attribute value ")
             << axis << " is out of range [-" << rank << ", " << rank << ")";
  }
  return success();
}

// ===----------- AMDQuarkExtendedQuantizeLinearOp ----------===//

LogicalResult AMDQuarkExtendedQuantizeLinearOp::verify() {
  if (failed(::verify<AMDQuarkExtendedQuantizeLinearOp>(this)))
    return failure();
  // Verify that zero_point element type matches the output element type.
  auto zpType = dyn_cast<ShapedType>(getYZeroPoint().getType());
  if (zpType) {
    Type zpElemType = zpType.getElementType();
    Type resultElemType = cast<ShapedType>(getY().getType()).getElementType();
    if (zpElemType != resultElemType)
      return emitOpError("y_zero_point element type ")
             << zpElemType << " must match result element type "
             << resultElemType;
  }
  return success();
}

std::optional<int64_t> AMDQuarkExtendedQuantizeLinearOp::getNormalizedAxis() {
  return ::getNormalizedAxis<AMDQuarkExtendedQuantizeLinearOp>(this);
}

LogicalResult AMDQuarkExtendedQuantizeLinearOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  return ::inferShapes<AMDQuarkExtendedQuantizeLinearOp>(this, [](Region &) {});
}

// ===----------- AMDQuarkExtendedDequantizeLinearOp ----------===//

LogicalResult AMDQuarkExtendedDequantizeLinearOp::verify() {
  if (failed(::verify<AMDQuarkExtendedDequantizeLinearOp>(this)))
    return failure();
  // Verify that zero_point element type matches the input element type.
  auto zpType = dyn_cast<ShapedType>(getXZeroPoint().getType());
  if (zpType) {
    Type zpElemType = zpType.getElementType();
    Type inputElemType = cast<ShapedType>(getX().getType()).getElementType();
    if (zpElemType != inputElemType)
      return emitOpError("x_zero_point element type ")
             << zpElemType << " must match input element type "
             << inputElemType;
  }
  return success();
}

std::optional<int64_t> AMDQuarkExtendedDequantizeLinearOp::getNormalizedAxis() {
  return ::getNormalizedAxis<AMDQuarkExtendedDequantizeLinearOp>(this);
}

LogicalResult AMDQuarkExtendedDequantizeLinearOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  return ::inferShapes<AMDQuarkExtendedDequantizeLinearOp>(
      this, [](Region &) {});
}
