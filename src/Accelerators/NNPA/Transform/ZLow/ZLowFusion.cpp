/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- ZLowFusionPass.cpp - ZLow Rewrite Patterns -------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

// clang-format off
// %116 = "zlow.convert_dlf16"(%alloc_536) {direction = "to_f32"} : (memref<4x1x256x256xf16, #map>) -> memref<4x1x256x256xf32, #map>
// %117 = "zlow.detach_layout"(%116) : (memref<4x1x256x256xf32, #map>) -> memref<4x1x256x256xf32>
// clang-format on

class FuseDetachLayoutPattern : public OpRewritePattern<ZLowDetachLayoutOp> {
public:
  using OpRewritePattern<ZLowDetachLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowDetachLayoutOp detachOp, PatternRewriter &rewriter) const override {
    Location loc = detachOp.getLoc();
    Value input = detachOp.getInput();
    Value output = detachOp.getOutput();
    StringAttr layout = detachOp.getLayoutNameAttr();

    // 1. Match
    if (input.isa<BlockArgument>())
      return failure();

    Operation *producer = input.getDefiningOp();
    ZLowConvertDLF16Op convertOp = llvm::dyn_cast<ZLowConvertDLF16Op>(producer);
    if (!convertOp)
      return failure();

    // 2. Rewrite
    MultiDialectBuilder<MemRefBuilder, IndexExprBuilderForZLow> create(
        rewriter, loc);
    IndexExprScope indexScope(create.zlowIE);

    Value zMemRef = convertOp.getInput();
    SmallVector<IndexExpr, 4> ubs;
    create.zlowIE.getShapeAsDims(zMemRef, ubs);

    // Allocate the output buffer.
    Value alloc =
        create.mem.alignedAlloc(output.getType().cast<MemRefType>(), ubs);

    // Emit zlow.unstick.
    rewriter.create<ZLowUnstickOp>(loc, convertOp.getInput(), alloc, layout);
    output.replaceAllUsesWith(alloc);

    return success();
  }
};

// clang-format off
// %105 = "zlow.attach_layout"(%reinterpret_cast_513) {layout = #map} : (memref<4x1x256x256xf32>) -> memref<4x1x256x256xf32, #map>
// %106 = "zlow.convert_dlf16"(%105) {direction = "from_f32"} : (memref<4x1x256x256xf32, #map>) -> memref<4x1x256x256xf16, #map>
// clang-format on
class FuseAttachLayoutPattern : public OpRewritePattern<ZLowAttachLayoutOp> {
public:
  using OpRewritePattern<ZLowAttachLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowAttachLayoutOp attachOp, PatternRewriter &rewriter) const override {
    Location loc = attachOp.getLoc();
    Value input = attachOp.getInput();
    Value output = attachOp.getOutput();
    StringAttr layout = attachOp.getLayoutNameAttr();

    // 1. Match
    ZLowConvertDLF16Op convertOp;
    for (Operation *user : output.getUsers()) {
      if (auto op = llvm::dyn_cast<ZLowConvertDLF16Op>(user)) {
        // Should not have two ZLowConvertDLF16Op users.
        if (convertOp)
          return failure();
        convertOp = op;
      }
    }
    if (!convertOp)
      return failure();
    Value convertedOutput = convertOp.getOutput();

    // 2. Rewrite
    MultiDialectBuilder<MemRefBuilder, IndexExprBuilderForZLow> create(
        rewriter, loc);
    IndexExprScope indexScope(create.zlowIE);

    SmallVector<IndexExpr, 4> ubs;
    create.zlowIE.getShapeAsDims(input, ubs);

    // Allocate the output buffer.
    Value alloc = create.mem.alignedAlloc(
        convertedOutput.getType().cast<MemRefType>(), ubs, 4096);

    // Emit zlow.unstick.
    rewriter.create<ZLowStickOp>(loc, input, alloc, layout);
    convertedOutput.replaceAllUsesWith(alloc);

    return success();
  }
};

/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowFusionPass
    : public PassWrapper<ZLowFusionPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "zlow-fusion"; }

  StringRef getDescription() const override {
    return "This pass sinks zlow.attach_layout and zlow.detach operations into "
           "affine-for loops";
  }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<FuseDetachLayoutPattern>(&getContext());
    patterns.insert<FuseAttachLayoutPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowFusionPass() {
  return std::make_unique<ZLowFusionPass>();
}

} // namespace zlow
} // namespace onnx_mlir
