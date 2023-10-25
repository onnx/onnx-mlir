/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowInsertDLF16ConversionPass.cpp - ZLow Rewrite Patterns ---===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This passs removes zlow.unstick and zlow.stick by inserting dlf16 conversion
// directly into affine-for loops;
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

// clang-format off

/// This pattern rewrites
/// ```mlir
/// "zlow.unstick"(%arg0, %alloc) {layout = "2D"} : (memref<5x7xf16, #map>, memref<5x7xf32>) -> ()
/// %out = memref.alloc(%dim) {alignment = 16 : i64} : memref<5x7xf32>
/// affine.for %arg1 = 0 to 5 {
///   affine.for %arg2 = 0 to 7 {
///     %0 = affine.load %alloc[%arg1, %arg2] : memref<5x7xf32>
///     %1 = math.sqrt %0 : f32 
///     affine.store %1, %out[%arg1, %arg2] : memref<5x7xf32>
///   }
/// }
/// 
/// into
/// 
/// %out = memref.alloc(%dim) {alignment = 16 : i64} : memref<5x7xf32>
/// affine.for %arg1 = 0 to 5 {
///   affine.for %arg2 = 0 to 7 {
///     %0 = affine.load %arg0[0, %arg2 floordiv 64, 0, %arg1 floordiv 32, %arg1 mod 32, %arg2 mod 64] : memref<1x1x1x1x32x64xf16>
///     %1 = "zlow.dlf16_to_f32"(%0) : (f16) -> f32
///     %2 = math.sqrt %1 : f32 
///     affine.store %2, %out[%arg1, %arg2] : memref<5x7xf32>
///   }
/// }
/// ```
/// where `affine.load` is now loading data directly from a zTensor.
///

// clang-format on

class DLF16ConversionForLoadPattern : public OpRewritePattern<ZLowUnstickOp> {
public:
  using OpRewritePattern<ZLowUnstickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowUnstickOp unstickOp, PatternRewriter &rewriter) const override {
    Location loc = unstickOp.getLoc();

    Operation *op = unstickOp.getOperation();
    Value zMemRef = unstickOp.getX();
    Value cpuMemRef = unstickOp.getOut();
    std::string layout = unstickOp.getLayout().value().str();

    // 1. Match

    // Only support fp32 and identity affine layout in the CPU MemRef.
    if (auto type = dyn_cast<MemRefType>(cpuMemRef.getType())) {
      if (!type.getElementType().isa<Float32Type>())
        return failure();
      AffineMap m = type.getLayout().getAffineMap();
      if (m.getNumResults() != 1 && !m.isIdentity())
        return failure();
      // Optional<uint64_t> sizeInBytes = getMemRefSizeInBytes(type);
      // if (!sizeInBytes.has_value() || (sizeInBytes.value() > 256 * 4))
      //   return failure();
    }

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    if ((layout == LAYOUT_1D) || (layout == LAYOUT_2DS))
      return failure();

    // All users except zlow.unstick must be affine.load, so that zlow.unstick
    // will be dangling and can be totally removed at the end of this pass.
    SmallVector<affine::AffineLoadOp, 4> affineLoads;
    for (Operation *user : cpuMemRef.getUsers()) {
      if (user == op)
        continue;
      if (auto affineLoad = llvm::dyn_cast<affine::AffineLoadOp>(user))
        affineLoads.emplace_back(affineLoad);
      else
        return failure();
    }
    if (affineLoads.size() == 0)
      return failure();

    // 2. Rewrite
    MultiDialectBuilder<AffineBuilder> create(rewriter, loc);
    for (affine::AffineLoadOp loadOp : affineLoads) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(loadOp);
      ValueRange indices = loadOp.getIndices();
      // Load a dfl16 directly from zTensor and convert it to fp32.
      Value loadDLF16 = create.affine.load(zMemRef, indices);
      Value toFP32 = rewriter.create<ZLowConvertDLF16ToF32Op>(loc, loadDLF16);
      rewriter.replaceOp(loadOp, {toFP32});
    }

    rewriter.eraseOp(unstickOp);
    return success();
  }
};

class DLF16ConversionForStorePattern : public OpRewritePattern<ZLowStickOp> {
public:
  using OpRewritePattern<ZLowStickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ZLowStickOp stickOp, PatternRewriter &rewriter) const override {
    Location loc = stickOp.getLoc();

    Operation *op = stickOp.getOperation();
    Value cpuMemRef = stickOp.getX();
    Value zMemRef = stickOp.getOut();
    std::string layout = stickOp.getLayout().value().str();

    // 1. Match

    // Only support fp32 and identity affine layout in the CPU MemRef.
    if (auto type = dyn_cast<MemRefType>(cpuMemRef.getType())) {
      if (!type.getElementType().isa<Float32Type>())
        return failure();
      AffineMap m = type.getLayout().getAffineMap();
      if (m.getNumResults() != 1 && !m.isIdentity())
        return failure();
      // Optional<uint64_t> sizeInBytes = getMemRefSizeInBytes(type);
      // if (!sizeInBytes.has_value() || (sizeInBytes.value() > 256 * 4))
      //   return failure();
    }

    // Do not support layout 1D and 2DS since their access index functions are
    // incorrect: https://github.com/onnx/onnx-mlir/issues/1940
    if ((layout == LAYOUT_1D) || (layout == LAYOUT_2DS))
      return failure();

    // All users except zlow.stick must be affine.load, so that zlow.stick
    // will be dangling and can be totally removed at the end of this pass.
    SmallVector<affine::AffineStoreOp, 4> affineStores;
    for (Operation *user : cpuMemRef.getUsers()) {
      if (user == op)
        continue;
      if (auto affineStore = llvm::dyn_cast<affine::AffineStoreOp>(user))
        affineStores.emplace_back(affineStore);
      else
        return failure();
    }
    if (affineStores.size() == 0)
      return failure();

    // 2. Rewrite
    // Move up the allocation of zMemRef so that it dominates the following
    // stores.
    zMemRef.getDefiningOp()->moveAfter(cpuMemRef.getDefiningOp());

    // Replace AffineStoreOp.
    MultiDialectBuilder<AffineBuilder> create(rewriter, loc);
    for (affine::AffineStoreOp storeOp : affineStores) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(storeOp);
      ValueRange indices = storeOp.getIndices();
      Value f32 = storeOp.getValue();
      // Convert the value to dlf16 and store it to zTensor directly.
      Value dlf16 = rewriter.create<ZLowConvertF32ToDLF16Op>(loc, f32);
      create.affine.store(dlf16, zMemRef, indices);
      // Remove the old AffineStore.
      rewriter.eraseOp(storeOp);
    }

    rewriter.eraseOp(stickOp);
    return success();
  }
};
/*!
 *  Function pass that optimizes ZLowIR.
 */
class ZLowInsertDLF16ConversionPass
    : public PassWrapper<ZLowInsertDLF16ConversionPass,
          OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "zlow-insert-dlf16-conversion";
  }

  StringRef getDescription() const override {
    return "Replacing zlow.unstick and zlow.stick by inserting dlf16 "
           "conversion directly into affine-for loops";
  }

  void runOnOperation() override {
    Operation *function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<DLF16ConversionForLoadPattern>(&getContext());
    patterns.insert<DLF16ConversionForStorePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowInsertDLF16ConversionPass() {
  return std::make_unique<ZLowInsertDLF16ConversionPass>();
}

} // namespace zlow
} // namespace onnx_mlir
