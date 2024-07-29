/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
// Modifications Copyright 2023-2024
//
// =============================================================================
//
// This file defines common base utilities for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RNN_BASE_CONV_H
#define ONNX_MLIR_RNN_BASE_CONV_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

static constexpr llvm::StringRef FORWARD = "forward";
static constexpr llvm::StringRef REVERSE = "reverse";
static constexpr llvm::StringRef BIDIRECTIONAL = "bidirectional";

namespace onnx_mlir {

struct RNNActivation {
  llvm::StringRef name;
  std::optional<mlir::FloatAttr> alpha;
  std::optional<mlir::FloatAttr> beta;
};

/// Get a dimension of the tensor's shape.
int64_t dimAt(mlir::Value val, int index);

/// Apply an activation function on a given operand.
mlir::Value applyActivation(mlir::OpBuilder &rewriter, mlir::Location loc,
    RNNActivation activation, mlir::Value operand);

// Override the following methods when lowering an RNN operation:
// - hasAllNoneOutput
// - getActivationPack

// Check whether all outputs have NoneType or not.
template <typename RNNOp>
bool hasAllNoneOutput(RNNOp *op);

// Obtain activations functions for a specific operation.
template <typename RNNOp, typename A>
std::tuple<A, A> getActivationPack(RNNOp *op);

} // namespace onnx_mlir
#endif
