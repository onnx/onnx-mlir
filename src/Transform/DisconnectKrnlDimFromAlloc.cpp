//===-------- DisconnectKrnlDimFromAlloc.cpp ------------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This pass enables the lowering of the krnl.dim operation to a series of
// instruction which do not depend on the alloc of the MemRef whose dim is
// being taken. The krnl.dim operation works in the presence of MemRefs
// which contain affine maps by ignoring the map if present.
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

class DisconnectKrnlDimFromAlloc : public OpRewritePattern<KrnlDimOp> {
public:
  using OpRewritePattern<KrnlDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlDimOp krnlDimOp, PatternRewriter &rewriter) const override {
    auto loc = krnlDimOp.getLoc();

    // If index is not constant, return failure.
    ConstantOp indexOp =
        dyn_cast<ConstantOp>(krnlDimOp.index().getDefiningOp());
    if (!indexOp)
      return failure();

    // Get the integer value of the index.
    int64_t index = indexOp.getAttrOfType<IntegerAttr>("value").getInt();

    // Get the shape of the MemRef argument.
    auto memRefType = convertToMemRefType(krnlDimOp.alloc().getType());
    auto memRefShape = memRefType.getShape();
    auto rank = memRefShape.size();
    assert(index >= 0 && index < rank && "Index must be in bounds");

    // Get the defining operation of the first argument of krnl.dim.
    // If this operation is not an alloc, and the value comes from the
    // list of input arguments, the support is limited to MemRefs without
    // maps.
    auto firstArgDefOp = krnlDimOp.alloc().getDefiningOp();

    Value result;
    if (memRefShape[index] > -1) {
      // If dimension is static, then we can just emit the constant value.
      result = rewriter.create<ConstantOp>(loc,
          rewriter.getIntegerAttr(rewriter.getIndexType(), memRefShape[index]));
    } else if (firstArgDefOp && isa<AllocOp>(firstArgDefOp)) {
      // Get defining operation for the MemRef argument.
      AllocOp allocOp = dyn_cast<AllocOp>(krnlDimOp.alloc().getDefiningOp());

      // If dimension is dynamic we need to return the input alloc Value which
      // corresponds to it.
      int64_t dynDimIdx = getAllocArgIndex(allocOp, index);
      assert(dynDimIdx >= 0 && dynDimIdx < allocOp.getOperands().size() &&
             "Dynamic index outside range of alloc argument list.");
      result = allocOp.getOperands()[dynDimIdx];
    } else if (memRefType.getAffineMaps().empty()) {
      // Use a standard DimOp since no map is present.
      result = rewriter.create<DimOp>(loc, krnlDimOp.alloc(), krnlDimOp.index());
    } else {
      llvm_unreachable("dynamic sized MemRef with map must be defined by an AllocOp");
    }

    rewriter.replaceOp(krnlDimOp, result);

    return success();
  }
};

/*!
 *  Function pass that disconnects krnl.dim emission from its MemRef alloc.
 */
class DisconnectKrnlDimFromAllocPass
    : public PassWrapper<DisconnectKrnlDimFromAllocPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    patterns.insert<DisconnectKrnlDimFromAlloc>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createDisconnectKrnlDimFromAllocPass() {
  return std::make_unique<DisconnectKrnlDimFromAllocPass>();
}
