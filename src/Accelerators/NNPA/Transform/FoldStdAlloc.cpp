//===-------- FoldStdAlloc.cpp - Fold std.alloc ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This pass replaces a std.alloc by a constant when applicable.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Pass/NNPAPasses.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

/// This pass replaces a std.alloc by a constant if all the elements in the
/// returned MemRef of the std.alloc are constants.
///
/// Example:
/// The following code:
///   %c0 = constant 0 : i64
///   %c1 = constant 1 : i64
///   %c2 = constant 2 : i64
///
///   %c7 = constant 7 : i64
///   %c8 = constant 8 : i64
///   %c9 = constant 9 : i64
///
///   %0 = alloc() : memref<3xi64>
///   krnl.store %c7, %0[%c0] : memref<3xi64>
///   krnl.store %c8, %0[%c1] : memref<3xi64>
///   krnl.store %c9, %0[%c2] : memref<3xi64>
///
/// will be replaced by:
///   "krnl.global"() {name = "constant_fold_std_alloc_0",
///                    shape = [3],
///                    value = dense<[7, 8, 9]> : tensor<3xi64>}
///                  : () -> memref<3xi64

static int constantFoldStdAllocID = 0;

/// Get a constant value from a ConstantOp.
static int64_t getConstantValue(arith::ConstantOp constOp) {
  if (IntegerAttr attr = constOp.getValue().dyn_cast<IntegerAttr>()) {
    int64_t val = attr.cast<IntegerAttr>().getInt();
    return val;
  } else {
    llvm_unreachable("Only support IntegerAttr");
  }
}

/// Fold std.alloc.
/// This only supports static 1D MemRefs with integer type.
class FoldStdAlloc : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();

    Value memRef = allocOp.getResult();
    MemRefType memRefType = memRef.getType().dyn_cast<MemRefType>();
    Type elementType = memRefType.getElementType();

    // 1. Match

    // The MemRef type returned by the AllocOp must be normalized.
    if (!memRefType.getLayout().isIdentity())
      return failure();

    // Only support static 1D MemRefs with integer type.
    if ((memRefType.getRank() != 1) || (memRefType.getShape()[0] == -1) ||
        (!elementType.isa<IntegerType>()))
      return failure();

    // The number of elements in the returned MemRef.
    int numElements = memRefType.getShape()[0];

    // Collect all stores to the memref. The store index and value in each store
    // must be constants.
    llvm::SmallDenseSet<Operation *, 2> storeOps;
    // Collect all related dealloc ops.
    llvm::SmallDenseSet<Operation *, 2> deallocOps;
    // A map from the store index to store value. Using std::map to have the map
    // autotically sorted.
    std::map<int64_t, int64_t> indexToValueMap;
    for (Operation *user : memRef.getUsers()) {
      // Collect DeallocOp.
      if (isa<memref::DeallocOp>(user)) {
        deallocOps.insert(user);
        continue;
      }

      // Only support KrnlStoreOp/AffineStoreOp/StoreOp
      if (!(isa<KrnlStoreOp>(user) || isa<AffineStoreOp>(user) ||
              isa<memref::StoreOp>(user)))
        continue;

      // StoreOp must be in the same block as AllocOp.
      if (user->getBlock() != allocOp.getOperation()->getBlock())
        return failure();

      Value storeIndex, storeVal;
      if (isa<KrnlStoreOp>(user)) {
        KrnlStoreOp op = llvm::dyn_cast<KrnlStoreOp>(user);
        storeIndex = *op.getIndices().begin();
        storeVal = op.getValueToStore();
      } else if (isa<memref::StoreOp>(user)) {
        memref::StoreOp op = llvm::dyn_cast<memref::StoreOp>(user);
        storeIndex = *op.getIndices().begin();
        storeVal = op.getValueToStore();
      } else { // AffineStoreOp
        AffineStoreOp op = llvm::dyn_cast<AffineStoreOp>(user);
        SmallVector<Value, 2> indices(op.getMapOperands());
        auto maybeExpandedMap =
            expandAffineMap(rewriter, loc, op.getAffineMap(), indices);
        if (!maybeExpandedMap)
          continue;
        storeIndex = maybeExpandedMap.getValue()[0];
        storeVal = op.getValueToStore();
      }

      int64_t idx, val;
      // Check the store index.
      if (arith::ConstantOp op =
              storeIndex.getDefiningOp<arith::ConstantOp>()) {
        idx = getConstantValue(op);
      } else
        return failure();

      // Check the store value.
      if (arith::ConstantOp op = storeVal.getDefiningOp<arith::ConstantOp>()) {
        val = getConstantValue(op);
      } else
        return failure();

      indexToValueMap.insert(std::pair<int64_t, int64_t>(idx, val));
      storeOps.insert(user);
    }

    // There must be exactly N stores to N different locations, where N is the
    // number of elements.
    if ((int)storeOps.size() != numElements)
      return failure();
    if ((int)indexToValueMap.size() != numElements)
      return failure();

    // 2. Rewrite.
    // We replace std.alloc by std.constant whose attribute is constructed from
    // the store values.

    SmallVector<int64_t, 4> data;
    for (int i = 0; i < numElements; ++i)
      data.emplace_back(indexToValueMap[i]);

    RankedTensorType tensorType =
        RankedTensorType::get(memRefType.getShape(), elementType);
    DenseElementsAttr dataAttr = DenseElementsAttr::get<int64_t>(
        tensorType, llvm::makeArrayRef<int64_t>(data));

    KrnlGlobalOp resOp = rewriter.create<KrnlGlobalOp>(loc, memRefType,
        /*shape=*/
        rewriter.getI64ArrayAttr({numElements}),
        /*name=*/
        rewriter.getStringAttr("constant_fold_std_alloc_" +
                               std::to_string(constantFoldStdAllocID)),
        /*value=*/dataAttr,
        /*offset=*/nullptr,
        /*alignment=*/nullptr);
    constantFoldStdAllocID++;

    // Erase all the stores.
    for (Operation *op : storeOps)
      rewriter.eraseOp(op);
    // Erase all the dealloc.
    for (Operation *op : deallocOps)
      rewriter.eraseOp(op);

    rewriter.replaceOp(allocOp, resOp.getResult());

    return success();
  }
};

/*!
 *  Function pass that folds std.alloc.
 */
class FoldStdAllocPass
    : public PassWrapper<FoldStdAllocPass, OperationPass<FuncOp>> {
public:
  StringRef getArgument() const override { return "fold-std-alloc"; }

  StringRef getDescription() const override {
    return "Fold std.alloc to constants.";
  }

  void runOnOperation() override {
    auto function = getOperation();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<FoldStdAlloc>(&getContext());

    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

std::unique_ptr<Pass> onnx_mlir::createFoldStdAllocPass() {
  return std::make_unique<FoldStdAllocPass>();
}
