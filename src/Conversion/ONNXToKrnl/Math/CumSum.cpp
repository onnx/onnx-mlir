/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- CumSum.cpp - Lowering CumSum Ops ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CumSum Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXCumSumOpLowering : public ConversionPattern {
  ONNXCumSumOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXCumSumOp::getOperationName(), 1, ctx) {}

  /// We use a naive alogrithm for cumsum as follows:
  /// ```
  /// y = x
  /// for d in range(log2(n)):
  ///   for i range(n):
  ///     if i >= 2^d:
  ///       y[i] = y[i - 2^(d-1)] + y[i]
  ///     else:
  ///       y[i] = y[i]
  ///
  /// ```
  ///
  /// Blelloch algorithm [1] is more work-efficent. However, it is not
  /// affine-friendly, because the inner bounds depend on the outer bounds.
  ///
  /// [1] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications."
  /// Technical Report CMU-CS-90-190, School of Computer Science, Carnegie
  /// Mellon University.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXCumSumOpAdaptor operandAdaptor(operands);
    ONNXCumSumOp gemmOp = llvm::cast<ONNXCumSumOp>(op);
    Location loc = op->getLoc();

    // Builder helper.
    IndexExprScope outerScope(rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);
    MemRefBuilder createMemRef(createKrnl);

    // Common information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value X = operandAdaptor.x();
    Value axis = createKrnl.load(operandAdaptor.axis(), {});

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      resMemRef =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);

    // Set upper and lower bounds.

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXCumSumOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCumSumOpLowering>(ctx);
}
