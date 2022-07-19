/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.hpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

class DecomposeEinsumPattern
    : public mlir::OpRewritePattern<mlir::ONNXEinsumOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXEinsumOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXEinsumOp einsumOp,
      mlir::PatternRewriter &rewriter) const override;
};

} // namespace onnx_mlir
