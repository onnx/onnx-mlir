//===------------------ LowerKrnlGlobal.cpp ------------------------------===//
//
// This pass enables lowering krnl.global of floating point scalar constants
// to std.constant.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace llvm;

namespace {

/// RewritePattern that replaces
///
///   %1 = "krnl.global"() {name = "constant_0", shape = [], value =
///     dense<0.00392156886> : tensor<f32>} : () -> memref<f32>
///   %3 = affine.load %1[] : memref<f32>
///   %4 = [OP] %2, %3 : f32
///
/// with
///
///   %cst = constant dense<0.00392156886> : f32
///   %4 = [OP] %2, %cst : f32
class LowerKrnlGlobal : public OpRewritePattern<KrnlGlobalOp> {
public:
  using OpRewritePattern<KrnlGlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlGlobalOp krnlGlobalOp, PatternRewriter &rewriter) const override {

    auto valueAttr = krnlGlobalOp.valueAttr();
    if (!valueAttr)
      return failure();

    auto denseElementsAttr =
        valueAttr.dyn_cast<DenseElementsAttr>();
    if (!denseElementsAttr || !denseElementsAttr.isSplat())
      return failure();

    auto constantFloatAttr = denseElementsAttr.getSplatValue<FloatAttr>();
    if (!constantFloatAttr)
      return failure();

    auto constantOp = rewriter.create<mlir::ConstantOp>(
        krnlGlobalOp.getLoc(), constantFloatAttr);

    llvm::DenseSet<Operation *> eraseSet;
    for (auto &use : krnlGlobalOp.output().getUses()) {
      auto affineLoadOp = dyn_cast<AffineLoadOp>(*use.getOwner());
      assert(affineLoadOp && "Use of krnl.global must be an affine load.");
      affineLoadOp.result().replaceAllUsesWith(constantOp);
      eraseSet.insert(affineLoadOp);
    }

    for (auto *e : eraseSet)
      e->erase();
    krnlGlobalOp.erase();

    return success();
  }
};

///  Function pass that lowers krnl.global of floating point scalar constants to
///  std.constant.
class LowerKrnlGlobalPass
    : public PassWrapper<LowerKrnlGlobalPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto function = getFunction();

    OwningRewritePatternList patterns;
    patterns.insert<LowerKrnlGlobal>(&getContext());

    applyPatternsAndFoldGreedily(function, patterns);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlGlobalPass() {
  return std::make_unique<LowerKrnlGlobalPass>();
}
