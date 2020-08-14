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
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 *  RewritePattern that replaces:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %1 = krnl.dim(%0, 0) : memref<?x10x<type>>
 *    %2 = krnl.dim(%0, 1) : memref<?x10x<type>>
 *    %3 = add %1, %2
 *  with:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %2 = constant 10 : index
 *    %3 = add %d, %2
 */

class LowerKrnlShape : public OpRewritePattern<KrnlShapeOp> {
public:
  using OpRewritePattern<KrnlShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlShapeOp krnlShapeOp, PatternRewriter &rewriter) const override {
    auto loc = krnlShapeOp.getLoc();

    printf("Bla \n");

    // // If index is not constant, return failure.
    // // TODO: support dynamic index case.
    // ConstantOp indexOp =
    //     dyn_cast<ConstantOp>(krnlDimOp.index().getDefiningOp());
    // if (!indexOp)
    //   return failure();

    // // Get the integer value of the index.
    // int64_t index = indexOp.getAttrOfType<IntegerAttr>("value").getInt();

    // printf("Index is %d\n", index);

    // // Get defining operation for the MemRef argument.
    // AllocOp allocOp =
    //     dyn_cast<AllocOp>(krnlDimOp.alloc().getDefiningOp());
    // auto memRefShape =
    //     convertToMemRefType(allocOp.getResult().getType()).getShape();
    // auto rank =  memRefShape.size();
    // assert(index >= 0 && index < rank && "Index must be in bounds");

    // Value result;
    // if (memRefShape[index] > -1) {
    //   // If dimension is static, then we can just emit the constant value.
    //   result = rewriter.create<ConstantOp>(loc,
    //       rewriter.getIntegerAttr(rewriter.getIndexType(),
    //           memRefShape[index]));
    // } else {
    //   // If dimension is dynamic we need to return the input alloc Value which
    //   // corresponds to it.
    //   int64_t dynDimIdx = getAllocArgIndex(allocOp, index);
    //   assert(dynDimIdx >= 0 && dynDimIdx < allocOp.getOperands().size() &&
    //       "Dynamic index outside range of alloc argument list.");

    //   result = allocOp.getOperands()[dynDimIdx];
    // }

    // rewriter.replaceOp(krnlDimOp, result);

    return success();
  }
};

/*!
 *  Function pass that disconnects krnl.dim emission from its MemRef alloc.
 */
class LowerKrnlShapePass
    : public PassWrapper<LowerKrnlShapePass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<LowerKrnlShape>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlShapePass() {
  return std::make_unique<LowerKrnlShapePass>();
}