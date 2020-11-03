//===-------- LowerKrnlShape.cpp ------------------------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This pass enables the lowering of the krnl.shape operation to use Shape
// dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

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
    auto loc = krnlShapeOp.getLoc();
    auto rank =
        convertToMemRefType(krnlShapeOp.alloc().getType()).getShape().size();

    SmallVector<mlir::Value, 4> fromExtentsOpOperands;
    for (int idx = 0; idx < rank; idx++) {
      auto index = rewriter.create<ConstantOp>(
          loc, rewriter.getIntegerAttr(rewriter.getIndexType(), idx));
      auto operand = rewriter.create<KrnlDimOp>(
          loc, rewriter.getIndexType(), krnlShapeOp.alloc(), index);
      fromExtentsOpOperands.emplace_back(operand);
    }

    auto fromExtentsOp = rewriter.create<mlir::shape::FromExtentsOp>(
        loc, rewriter.getType<mlir::shape::ShapeType>(), fromExtentsOpOperands);
    rewriter.replaceOp(krnlShapeOp, fromExtentsOp.getResult());

    return success();
  }
};

/*!
 *  Function pass that emits the shape of a MemRef.
 */
class LowerKrnlShapePass
    : public PassWrapper<LowerKrnlShapePass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<LowerKrnlShape>(&getContext());

    applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};
} // namespace

// TODO: integrate with other passes if needed.
std::unique_ptr<Pass> mlir::createLowerKrnlShapePass() {
  return std::make_unique<LowerKrnlShapePass>();
}
