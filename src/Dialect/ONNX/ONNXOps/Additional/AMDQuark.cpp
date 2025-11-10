/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ AMDQuark.cpp - AMD Quark custom ops ---------------===//
//
// Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

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
  if (!ignoreAxis && op->getAxis() != config.axis)
    return false;
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

LogicalResult AMDQuarkBFPQuantizeDequantizeOp::inferShapes(
    std::function<void(Region &)> /*doShapeInference*/) {
  return inferShapeForUnaryOps(this->getOperation());
}