/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- LowerKrnlShape.cpp ------------------------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass enables the lowering of the krnl.shape operation to use Shape
// dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

/*!
 *  RewritePattern that replaces:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %1 = krnl.shape(%0) : memref<?x10x<type>> -> !shape.shape
 *  with:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %c0 = constant 0 : index
 *    %1 = krnl.dim(%0, %c0) : memref<?x10x<type>, #map>, index
 *    %c1 = constant 1 : index
 *    %2 = krnl.dim(%0, %c1) : memref<?x10x<type>, #map>, index
 *    %shape = shape.from_extents %1, %2
 */

class LowerKrnlShape : public OpRewritePattern<KrnlShapeOp> {
public:
  using OpRewritePattern<KrnlShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlShapeOp krnlShapeOp, PatternRewriter &rewriter) const override {
    Location loc = krnlShapeOp.getLoc();
    size_t rank =
        krnlShapeOp.alloc().getType().dyn_cast<MemRefType>().getShape().size();

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Create MemRef to hold shape information.
    auto memRefType =
        MemRefType::get({static_cast<int64_t>(rank)}, rewriter.getIndexType());
    memref::AllocOp newMemRefAlloc = create.mem.alloc(memRefType);

    for (size_t idx = 0; idx < rank; idx++) {
      Value index = create.math.constantIndex(idx);
      Value operand =
          create.krnl.dim(rewriter.getIndexType(), krnlShapeOp.alloc(), index);

      // Store value in the new MemRef.
      Value idxValue = create.math.constant(rewriter.getIndexType(), idx);
      SmallVector<Value, 1> indexArg = {idxValue};
      rewriter.create<AffineStoreOp>(loc, operand, newMemRefAlloc, indexArg);
    }

    rewriter.replaceOp(krnlShapeOp, newMemRefAlloc.getResult());

    return success();
  }
};

/*!
 *  Function pass that emits the shape of a MemRef.
 */
class LowerKrnlShapePass
    : public PassWrapper<LowerKrnlShapePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKrnlShapePass)

  StringRef getArgument() const override { return "lower-krnl-shape"; }

  StringRef getDescription() const override {
    return "Lower krnl.shape operation to use Shape dialect operations.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<LowerKrnlShape>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

// TODO: integrate with other passes if needed.
std::unique_ptr<Pass> onnx_mlir::createLowerKrnlShapePass() {
  return std::make_unique<LowerKrnlShapePass>();
}
