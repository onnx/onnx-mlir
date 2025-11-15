/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.cpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
// Modifications Copyright 2023-2024
//
// =============================================================================
//
// This file defines common base utilities for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXConversionCommon/RNN/RNNBase.hpp"

using namespace mlir;

namespace onnx_mlir {

// Get a dimension of the tensor's shape.
int64_t dimAt(Value val, int index) {
  return mlir::cast<ShapedType>(val.getType()).getShape()[index];
}

// Apply an activation function on a given scalar operand.
Value applyActivation(OpBuilder &rewriter, Location loc,
    RNNActivation activation, Value operand) {
  Value res;

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.value()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.value()));
  }
  Type resType = operand.getType();

  // Change equality to be case insensitive.
  if (activation.name.equals_insensitive("relu"))
    res = ONNXReluOp::create(rewriter, loc, resType, operand);
  else if (activation.name.equals_insensitive("tanh"))
    res = ONNXTanhOp::create(rewriter, loc, resType, operand);
  else if (activation.name.equals_insensitive("sigmoid"))
    res = ONNXSigmoidOp::create(rewriter, loc, resType, operand);
  else if (activation.name.equals_insensitive("affine"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("leakyrelu"))
    res = ONNXLeakyReluOp::create(rewriter, loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("thresholdedrelu"))
    res = ONNXThresholdedReluOp::create(
        rewriter, loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("scaledtanh"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("hardsigmoid"))
    res =
        ONNXHardSigmoidOp::create(rewriter, loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("elu"))
    res = ONNXEluOp::create(rewriter, loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("softsign"))
    res = ONNXSoftsignOp::create(rewriter, loc, resType, operand);
  else if (activation.name.equals_insensitive("softplus"))
    res = ONNXSoftplusOp::create(rewriter, loc, resType, operand);
  else
    llvm_unreachable("Unsupported activation");

  return res;
}

} // namespace onnx_mlir
