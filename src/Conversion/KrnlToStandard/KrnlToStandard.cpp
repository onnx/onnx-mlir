//===------------------------- KrnlToStandard.cpp -------------------------===//
//
// Module pass to convert from Krnl dialect to Standard dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "krnl-to-standard"

using namespace mlir;

namespace {

struct ConvertKrnlToStandardPass
    : public PassWrapper<ConvertKrnlToStandardPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  static const char *const kTVPGlobalMemrefName;
  static int64_t kTVPGlobalMemrefNameCounter;
};

const char *const ConvertKrnlToStandardPass::kTVPGlobalMemrefName =
    "global_memref";
int64_t ConvertKrnlToStandardPass::kTVPGlobalMemrefNameCounter = 0;

bool singleUseFunc(FuncOp funcOp, Operation *parentOp) {
  auto uses = funcOp.getSymbolUses(parentOp);
  return uses.hasValue() && (uses->end() - uses->begin() == 1);
}

// substitute newType in for the index'th argument and update the function
void replaceTypeInFuncSignature(
    FuncOp funcOp, int index, Type newType, PatternRewriter &rewriter) {

  auto funcType = funcOp.getType();
  SmallVector<Type> types{};
  for (auto i = 0; i < funcType.getNumInputs(); i++) {
    types.emplace_back(i == index ? newType : funcType.getInput(i));
  }

  auto nType = rewriter.getFunctionType(types, funcOp.getType().getResults());
  funcOp.setType(nType);
}
/// (1) For scalar constants, replace
///
///   %1 = "krnl.global"() {name = "constant_0", shape = [], value =
///     dense<0.00392156886> : tensor<f32>} : () -> memref<f32>
///   %2 = affine.load %1[] : memref<f32>
///   %3 = [OP] %2, %3 : f32
///
/// with
///
///   %cst = constant dense<0.00392156886> : f32
///   %3 = [OP] %2, %cst : f32
///
/// (2) For vector constants, replace
///
///   %1 = "krnl.global"() {name = "constant_0", shape = [10],
///     value = dense<[-0.631071984, ...]> : tensor<10xf32>}
///     : () -> memref<10xf32>
///
/// with
///
///   memref.global "private" constant @tvp_global_memref1 : memref<10xf32> =
///     dense<[-0.631071984, ...]>
///   %1 = memref.get_global @tvp_global_memref1 : memref<10xf32>
class LowerKrnlGlobal : public OpRewritePattern<KrnlGlobalOp> {
public:
  using OpRewritePattern<KrnlGlobalOp>::OpRewritePattern;

  static const int64_t kDefaultGlobalMemrefThreshold = 1048576;

  LogicalResult matchAndRewrite(
      KrnlGlobalOp krnlGlobalOp, PatternRewriter &rewriter) const override {

    auto valueAttr = krnlGlobalOp.valueAttr();
    if (!valueAttr)
      return failure();

    auto denseElementsAttr = valueAttr.dyn_cast<DenseElementsAttr>();
    if (!denseElementsAttr)
      return failure();

    if (denseElementsAttr.isSplat()) {
      auto constantAttr = denseElementsAttr.getSplatValue();
      if (!constantAttr)
        return failure();

      for (auto &use : krnlGlobalOp.output().getUses()) {
        auto ownerOp = use.getOwner();

        if (auto loadOp = dyn_cast<MemoryEffectOpInterface>(ownerOp)) {
          // if the use we're replacing is a load, remove the load itself
          ownerOp->replaceAllUsesWith(krnlGlobalOp);
        } else if (auto callOp = dyn_cast<CallOpInterface>(ownerOp)) {
          // if we're passing this to a function call, modify the function type
          auto moduleOp = krnlGlobalOp->template getParentOfType<ModuleOp>();
          auto funcOp = dyn_cast_or_null<FuncOp>(callOp.resolveCallable());
          assert(funcOp && funcOp.isDeclaration() &&
                 singleUseFunc(funcOp, moduleOp));

          auto opIndex = use.getOperandNumber();
          replaceTypeInFuncSignature(
              funcOp, opIndex, constantAttr.getType(), rewriter);
        }
      }

      rewriter.replaceOpWithNewOp<ConstantOp>(krnlGlobalOp, constantAttr);

      return success();
    } else {
      if (denseElementsAttr.getNumElements() > kDefaultGlobalMemrefThreshold) {
        krnlGlobalOp.emitWarning(
            "This constant is too big to be turned into a global memref.");
        return failure();
      }
      auto moduleOp = krnlGlobalOp->template getParentOfType<ModuleOp>();
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto sym_name = rewriter.getStringAttr(
          ConvertKrnlToStandardPass::kTVPGlobalMemrefName +
          std::to_string(
              ConvertKrnlToStandardPass::kTVPGlobalMemrefNameCounter++));
      auto sym_visibility = rewriter.getStringAttr("private");
      auto returnType = krnlGlobalOp.getType();
      auto typeAttr = TypeAttr::get(returnType);
      auto constant = rewriter.getUnitAttr();
      auto globalMemrefOp =
          rewriter.create<memref::GlobalOp>(moduleOp.getLoc(), sym_name,
              sym_visibility, typeAttr, krnlGlobalOp.valueAttr(), constant);

      auto symRefAttr = FlatSymbolRefAttr::get(
          krnlGlobalOp.getContext(), sym_name.getValue());

      rewriter.setInsertionPoint(krnlGlobalOp);
      rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
          krnlGlobalOp, returnType, symRefAttr);

      return success();
    }
  }
};

void ConvertKrnlToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<LowerKrnlGlobal>(&getContext());
  applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
} // namespace

std::unique_ptr<Pass> mlir::createConvertKrnlToStandardPass() {
  return std::make_unique<ConvertKrnlToStandardPass>();
}
