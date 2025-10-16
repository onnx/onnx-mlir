#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct FusePadIntoAveragePoolPattern
    : public OpRewritePattern<ONNXAveragePoolOp> {
  using OpRewritePattern<ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAveragePoolOp avgOp, PatternRewriter &rewriter) const override {

    Value input = avgOp.getX();
    auto padOp = input.getDefiningOp<ONNXPadOp>();
    if (!padOp)
      return failure();

    StringAttr modeAttr = padOp.getModeAttr();
    StringRef mode = "constant";
    if (modeAttr)
      mode = modeAttr.getValue();
    if (mode != "constant")
      return failure();
    float padValue = 0.0f;

    Value padsInput = padOp.getPads();
    Value constantValInput = padOp.getConstantValue();

    auto padsConstOp =
        dyn_cast_or_null<ONNXConstantOp>(padsInput.getDefiningOp());
    if (!padsConstOp)
      return failure();
    auto padsAttr = dyn_cast_or_null<ElementsAttr>(padsConstOp.getValueAttr());
    if (!padsAttr)
      return failure();

    auto constOp =
        dyn_cast_or_null<ONNXConstantOp>(constantValInput.getDefiningOp());
    if (!constOp)
      return failure();
    auto constAttr = dyn_cast_or_null<ElementsAttr>(constOp.getValueAttr());

    if (!constAttr)
      return failure();

    auto firstAttr = *constAttr.getValues<Attribute>().begin();
    if (auto fAttr = mlir::dyn_cast<FloatAttr>(firstAttr))
      padValue = fAttr.getValueAsDouble();

    if (padValue != 0.0f)
      return failure();

    SmallVector<int64_t> padsVals;
    for (auto val : padsAttr.getValues<Attribute>()) {
      if (auto iAttr = mlir::dyn_cast<IntegerAttr>(val)) {
        auto pad = iAttr.getInt();
        padsVals.push_back(pad);
      } else {
        padsVals.push_back(0);
      }
    }

    SmallVector<int64_t> mergedPads;
    if (auto existingPadsAttr = avgOp.getPadsAttr()) {
      for (Attribute v : existingPadsAttr) {
        mergedPads.push_back(cast<IntegerAttr>(v).getInt());
      }
    } else {
      mergedPads.resize(padsVals.size() / 2, 0);
    }

    if (mergedPads.size() != padsVals.size() / 2)
      return failure();

    mergedPads[0] += padsVals[2];
    mergedPads[1] += padsVals[3];
    mergedPads[2] += padsVals[6];
    mergedPads[3] += padsVals[7];

    auto mergedPadsAttr =
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>(mergedPads));

    SmallVector<Value, 1> operands;
    operands.push_back(padOp.getData());

    NamedAttrList attrs;
    attrs.set(avgOp.getKernelShapeAttrName(), avgOp.getKernelShapeAttr());
    attrs.set(avgOp.getPadsAttrName(), mergedPadsAttr);
    attrs.set(avgOp.getStridesAttrName(), avgOp.getStridesAttr());
    attrs.set(avgOp.getCeilModeAttrName(), avgOp.getCeilModeAttr());
    attrs.set(
        avgOp.getCountIncludePadAttrName(), avgOp.getCountIncludePadAttr());

    auto newAvgOp = rewriter.create<ONNXAveragePoolOp>(
        avgOp.getLoc(), avgOp->getResultTypes(), operands, attrs);

    rewriter.replaceOp(avgOp, newAvgOp->getResults());
    rewriter.eraseOp(padOp);

    return success();
  }
};

struct FusePadIntoAveragePoolPass
    : public PassWrapper<FusePadIntoAveragePoolPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePadIntoAveragePoolPass)

  StringRef getArgument() const override { return "fuse-pad-into-avgpool"; }
  StringRef getDescription() const override {
    return "Fuse ONNXPadOp into ONNXAveragePoolOp when mode=constant and "
           "pad=0.";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FusePadIntoAveragePoolPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createFusePadIntoAvgpoolPass() {
  return std::make_unique<FusePadIntoAveragePoolPass>();
}
} // namespace onnx_mlir