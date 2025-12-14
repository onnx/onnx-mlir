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

    // Check that pad mode is "constant" (default value, so should never be
    // null)
    StringRef mode = padOp.getMode();
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

    // Only handle 4D tensors (NCHW format)
    auto inputType = padOp.getData().getType().dyn_cast<RankedTensorType>();
    if (!inputType || inputType.getRank() != 4)
      return failure();

    // Extract pad values (guaranteed to be integers by ONNX spec)
    SmallVector<int64_t> padsVals;
    for (auto val : padsAttr.getValues<Attribute>()) {
      auto pad = cast<IntegerAttr>(val).getInt();
      padsVals.push_back(pad);
    }

    // Validate pads array size (2 * rank for begin/end)
    if (padsVals.size() != 8)
      return failure();

    // Only merge when padding is applied only to spatial dimensions (H, W)
    // padsVals layout: [N_begin, C_begin, H_begin, W_begin, N_end, C_end,
    // H_end, W_end]
    if (padsVals[0] != 0 || padsVals[1] != 0 || // N_begin, C_begin
        padsVals[4] != 0 || padsVals[5] != 0) { // N_end, C_end
      return failure(); // Cannot merge if batch or channel dims are padded
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

    // Merge spatial dimension padding (H, W)
    mergedPads[0] += padsVals[2]; // H_begin
    mergedPads[1] += padsVals[3]; // W_begin
    mergedPads[2] += padsVals[6]; // H_end
    mergedPads[3] += padsVals[7]; // W_end

    auto mergedPadsAttr =
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>(mergedPads));

    // Modify the AveragePool op in place instead of creating a new one
    rewriter.modifyOpInPlace(avgOp, [&]() {
      avgOp->setAttr(avgOp.getPadsAttrName(), mergedPadsAttr);
      avgOp.getXMutable().assign(padOp.getData());
      avgOp->setLoc(rewriter.getFusedLoc({padOp.getLoc(), avgOp.getLoc()}));
    });

    rewriter.replaceOp(padOp, avgOp.getResult());

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
