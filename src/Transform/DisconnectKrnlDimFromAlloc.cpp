/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;

namespace {

/*!
 *  RewritePattern that replaces:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %1 = krnl.dim(%0, 0) : (memref<?x10x<type>, #map>, index) -> index
 *    %2 = krnl.dim(%0, 1) : (memref<?x10x<type>, #map>, index) -> index
 *    %3 = add %1, %2
 *  with:
 *    %0 = alloc(%d) : memref<?x10x<type>, #map>
 *    %2 = constant 10 : index
 *    %3 = add %d, %2
 *
 *  When the first argument of the krnl.dim is an input argument
 * i.e. it is not the output of an alloc operation, we emit either
 * the constant or the strandard dim operation depending on whether
 * the dimension is static or dynamic.
 *
 *  function(%arg0 : memref<?x10x<type>>) {
 *    %0 = krnl.dim(%arg0, 0) : (memref<?x10x<type>>, index) -> index
 *    %1 = krnl.dim(%arg0, 1) : memref<?x10x<type>>
 *  }
 *
 *
 *  becomes:
 *
 *  function(%arg0 : memref<?x10x<type>>) {
 *    %0 = dim %arg0, 0 : (memref<?x10x<type>>, index) -> index
 *    %1 = constant 10 : index
 *  }
 *
 *  The following case is not supported:
 *
 *  function(%arg0 : memref<?x10x<type>, #map>) {
 *    %0 = krnl.dim(%arg0, 0) : (memref<?x10x<type>, #map>, index) -> index
 *  }
 */

class DisconnectKrnlDimFromAlloc : public OpRewritePattern<KrnlDimOp> {
public:
  using OpRewritePattern<KrnlDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlDimOp krnlDimOp, PatternRewriter &rewriter) const override {
    auto loc = krnlDimOp.getLoc();

    // If index is not constant, return failure.
    arith::ConstantOp indexOp =
        dyn_cast<arith::ConstantOp>(krnlDimOp.index().getDefiningOp());
    if (!indexOp)
      return failure();

    // Get the integer value of the index.
    int64_t index = indexOp->getAttrOfType<IntegerAttr>("value").getInt();

    // Get the shape of the MemRef argument.
    auto memRefType = krnlDimOp.alloc().getType().dyn_cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();
    assert(index >= 0 && index < rank && "Index must be in bounds");

    // Get the defining operation of the first argument of krnl.dim.
    // If this operation is not an alloc, and the value comes from the
    // list of input arguments, the support is limited to MemRefs without
    // maps.
    auto firstArgDefOp = krnlDimOp.alloc().getDefiningOp();

    Value result;
    if (memRefShape[index] > -1) {
      // If dimension is static, then we can just emit the constant value.
      result = rewriter.create<arith::ConstantOp>(loc,
          rewriter.getIntegerAttr(rewriter.getIndexType(), memRefShape[index]));
    } else if (firstArgDefOp && isa<memref::AllocOp>(firstArgDefOp)) {
      // Get defining operation for the MemRef argument.
      memref::AllocOp allocOp =
          dyn_cast<memref::AllocOp>(krnlDimOp.alloc().getDefiningOp());

      // If dimension is dynamic we need to return the input alloc Value which
      // corresponds to it.
      int64_t dynDimIdx = getAllocArgIndex(allocOp, index);
      assert(dynDimIdx >= 0 &&
             dynDimIdx < (int64_t)allocOp.getOperands().size() &&
             "Dynamic index outside range of alloc argument list.");
      result = allocOp.getOperands()[dynDimIdx];
    } else if (memRefType.getLayout().isIdentity()) {
      // Use a standard DimOp since no map is present.
      result = rewriter.create<memref::DimOp>(
          loc, krnlDimOp.alloc(), krnlDimOp.index());
    } else {
      llvm_unreachable(
          "dynamic sized MemRef with map must be defined by an AllocOp");
    }

    rewriter.replaceOp(krnlDimOp, result);

    return success();
  }
};

/*!
 *  Function pass that disconnects krnl.dim emission from its MemRef alloc.
 */
class DisconnectKrnlDimFromAllocPass
    : public PassWrapper<DisconnectKrnlDimFromAllocPass,
          OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "lower-krnl-shape-to-std"; }

  StringRef getDescription() const override {
    return "Lowers krnl shape-related operations.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<DisconnectKrnlDimFromAlloc>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createDisconnectKrnlDimFromAllocPass() {
  return std::make_unique<DisconnectKrnlDimFromAllocPass>();
}
