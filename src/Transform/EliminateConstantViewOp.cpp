/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- EliminateConstantViewOp.cpp ---------------------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This pass eliminates ops that create a view of a constant tensor.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/// RewritePattern that replaces:
///  %0 = krnl.global() {value = dense<[]> : tensor<5xf32>}: () -> memref<5xf32>
///  %1 = memref.reinterpret_cast %0 : memref<5xf32> to memref<5x1xf32>
/// with:
///  %0 = krnl.global() {value = dense<[]> : tensor<5x1xf32>}
///                    : () -> memref<5x1xf32>
///
/// This pattern applies to any operation that has ViewLikeOpInterface.
/// For example: 'memref.reinterpret_cast', 'memref.cast'.

struct EliminateConstantViewOp : public RewritePattern {
  static int constantID;

  EliminateConstantViewOp(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {
    constantID = 0;
  }

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    // This is a ViewOp.
    if (!dyn_cast<ViewLikeOpInterface>(op))
      return failure();

    Value source = cast<ViewLikeOpInterface>(op).getViewSource();
    Value target = op->getResult(0);
    MemRefType targetType = target.getType().cast<MemRefType>();

    // Source MemRef must be a constant.
    if (!llvm::dyn_cast<KrnlGlobalOp>(source.getDefiningOp()))
      return failure();
    // Target MemRef must has static shape.
    if (!targetType.hasStaticShape())
      return failure();

    KrnlGlobalOp krnlGlobal = llvm::cast<KrnlGlobalOp>(source.getDefiningOp());

    DenseElementsAttr valueAttr =
        krnlGlobal.valueAttr().cast<DenseElementsAttr>();
    // Create a new value attribute with the target type.
    DenseElementsAttr newValueAttr;
    RankedTensorType valueAttrType = RankedTensorType::get(
        targetType.getShape(), targetType.getElementType());
    if (krnlGlobal.value().getValue().isa<OpaqueElementsAttr>()) {
      StringRef data =
          krnlGlobal.value().getValue().cast<OpaqueElementsAttr>().getValue();
      newValueAttr = DenseElementsAttr::getFromRawBuffer(
          valueAttrType, ArrayRef<char>(data.data(), data.size()), false);
    } else if (krnlGlobal.value().getValue().isa<DenseElementsAttr>()) {
      newValueAttr = DenseElementsAttr::getFromRawBuffer(
          valueAttrType, valueAttr.getRawData(), valueAttr.isSplat());
    } else
      llvm_unreachable("Unsupported attribute type");

    // Create a KrnlGlobalOp.
    KrnlGlobalOp constantGlobal = rewriter.create<KrnlGlobalOp>(loc, targetType,
        /*shape=*/
        rewriter.getI64ArrayAttr(targetType.getShape()),
        /*name=*/
        rewriter.getStringAttr(
            "constant_from_view_op_" + std::to_string(constantID)),
        /*value=*/newValueAttr, /*offset=*/krnlGlobal.offsetAttr(),
        /*alignment=*/krnlGlobal.alignmentAttr());

    // Increment constant ID.
    constantID++;

    rewriter.replaceOp(op, constantGlobal.getResult());
    return success();
  }
};

int EliminateConstantViewOp::constantID;

/*!
 *  Function pass that emits the shape of a MemRef.
 */
class EliminateConstantViewOpPass
    : public PassWrapper<EliminateConstantViewOpPass, FunctionPass> {
public:
  StringRef getArgument() const override {
    return "eliminate-constant-view-op";
  }

  StringRef getDescription() const override {
    return "This pass eliminates operations that create a view of a constant "
           "tensor";
  }

  void runOnFunction() override {
    auto function = getFunction();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<EliminateConstantViewOp>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createEliminateConstantViewOpPass() {
  return std::make_unique<EliminateConstantViewOpPass>();
}
