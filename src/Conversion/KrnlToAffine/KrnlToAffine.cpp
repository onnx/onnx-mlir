/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

// EDSC intrinsics (which include all builder methods too).
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlTerminator operation.
//===----------------------------------------------------------------------===//

class KrnlTerminatorLowering : public OpRewritePattern<KrnlTerminatorOp> {
public:
  using OpRewritePattern<KrnlTerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlTerminatorOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AffineYieldOp>(op);
    return success();
  }
};

void lowerIterateOp(KrnlIterateOp &iterateOp, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &refToOps) {
  builder.setInsertionPointAfter(iterateOp);
  SmallVector<std::pair<Value, AffineForOp>, 4> currentNestedForOps;
  auto boundMapAttrs =
      iterateOp->getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
          .getValue();
  auto operandItr =
      iterateOp.operand_begin() + iterateOp.getNumOptimizedLoops();
  for (size_t boundIdx = 0; boundIdx < boundMapAttrs.size(); boundIdx += 2) {
    // Consume input loop operand, at this stage, do not do anything with it.
    auto unoptimizedLoopRef = *(operandItr++);

    // Organize operands into lower/upper bounds in affine.for ready formats.
    llvm::SmallVector<Value, 4> lbOperands, ubOperands;
    AffineMap lbMap, ubMap;
    for (int boundType = 0; boundType < 2; boundType++) {
      auto &operands = boundType == 0 ? lbOperands : ubOperands;
      auto &map = boundType == 0 ? lbMap : ubMap;
      map =
          boundMapAttrs[boundIdx + boundType].cast<AffineMapAttr>().getValue();
      operands.insert(
          operands.end(), operandItr, operandItr + map.getNumInputs());
      std::advance(operandItr, map.getNumInputs());
    }
    auto forOp = builder.create<AffineForOp>(
        iterateOp.getLoc(), lbOperands, lbMap, ubOperands, ubMap);

    currentNestedForOps.emplace_back(std::make_pair(unoptimizedLoopRef, forOp));
    builder.setInsertionPoint(currentNestedForOps.back().second.getBody(),
        currentNestedForOps.back().second.getBody()->begin());
  }

  // Replace induction variable references from those introduced by a
  // single krnl.iterate to those introduced by multiple affine.for
  // operations.
  for (int64_t i = 0; i < (int64_t)currentNestedForOps.size() - 1; i++) {
    auto iterateIV = iterateOp.bodyRegion().front().getArgument(0);
    auto forIV = currentNestedForOps[i].second.getBody()->getArgument(0);
    iterateIV.replaceAllUsesWith(forIV);
    iterateOp.bodyRegion().front().eraseArgument(0);
  }

  // Pop krnl.iterate body region block arguments, leave the last one
  // for convenience (it'll be taken care of by region inlining).
  while (iterateOp.bodyRegion().front().getNumArguments() > 1)
    iterateOp.bodyRegion().front().eraseArgument(0);

  if (currentNestedForOps.empty()) {
    // If no loops are involved, simply move operations from within iterateOp
    // body region to the parent region of iterateOp.
    builder.setInsertionPointAfter(iterateOp);
    iterateOp.bodyRegion().walk([&](Operation *op) {
      if (!op->isKnownTerminator())
        op->replaceAllUsesWith(builder.clone(*op));
    });
  } else {
    // Transfer krnl.iterate region to innermost for op.
    auto innermostForOp = currentNestedForOps.back().second;
    innermostForOp.region().getBlocks().clear();
    auto &innerMostRegion = innermostForOp.region();
    innerMostRegion.getBlocks().splice(
        innerMostRegion.end(), iterateOp.bodyRegion().getBlocks());
  }

  for (const auto &pair : currentNestedForOps)
    refToOps.try_emplace(pair.first, pair.second);
}

//===----------------------------------------------------------------------===//
// ConvertKrnlToAffinePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the krnl dialect operations.
/// At this stage the dialect will contain standard operations as well like
/// add and multiply, this pass will leave these operations intact.
namespace {
struct ConvertKrnlToAffinePass
    : public PassWrapper<ConvertKrnlToAffinePass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

LogicalResult interpretOperation(Operation *op, OpBuilder &builder,
    llvm::SmallDenseMap<Value, AffineForOp, 4> &loopRefToOp,
    llvm::SmallPtrSetImpl<Operation *> &opsToErase) {
  // Recursively interpret nested operations.
  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
      for (auto itr = blockOps.begin(); itr != blockOps.end();)
        if (failed(interpretOperation(
                &(*itr), builder, loopRefToOp, opsToErase))) {
          return failure();
        } else {
          ++itr;
        }
    }

  if (auto defineOp = dyn_cast_or_null<KrnlDefineLoopsOp>(op)) {
    // Collect users of defineLoops operations that are iterate operations.
    std::vector<KrnlIterateOp> iterateOps;
    for (auto result : op->getResults())
      for (auto *user : result.getUsers())
        if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(user))
          if (std::find(iterateOps.begin(), iterateOps.end(), iterateOp) ==
              iterateOps.end())
            iterateOps.push_back(dyn_cast<KrnlIterateOp>(user));

    // Lower iterate operations and record the mapping between loop references
    // and affine for loop operations in loopRefToOp map.
    if (!iterateOps.empty()) {
      for (auto opToLower : iterateOps) {
        if (opsToErase.count(opToLower) == 0) {
          lowerIterateOp(opToLower, builder, loopRefToOp);
          opsToErase.insert(opToLower);
        }
      }
    }
    opsToErase.insert(op);
    return success();
  } else if (auto iterateOp = dyn_cast_or_null<KrnlIterateOp>(op)) {
    // If an iterateOp has no unoptimized loop references, then we need to lower
    // them manually.
    if (opsToErase.count(op) == 0) {
      lowerIterateOp(iterateOp, builder, loopRefToOp);
      opsToErase.insert(iterateOp);
    }
    return success();
  } else if (auto blockOp = dyn_cast_or_null<KrnlBlockOp>(op)) {
    SmallVector<AffineForOp, 2> tiledLoops;
    SmallVector<AffineForOp, 1> loopsToTile = {loopRefToOp[blockOp.loop()]};
    if (failed(tilePerfectlyNested(
            loopsToTile, blockOp.tile_sizeAttr().getInt(), &tiledLoops))) {
      return failure();
    }
    assert(tiledLoops.size() == 2);
    assert(blockOp.getNumResults() == 2);

    // Record the tiled loop references, and their corresponding tiled
    // for loops in loopRefToLoop.
    loopRefToOp[blockOp.getResult(0)] = tiledLoops[0];
    loopRefToOp[blockOp.getResult(1)] = tiledLoops[1];

    opsToErase.insert(op);
    return success();
  } else if (auto permuteOp = dyn_cast_or_null<KrnlPermuteOp>(op)) {
    // Collect loops to permute.
    SmallVector<AffineForOp, 4> loopsToPermute;
    std::transform(permuteOp.operand_begin(), permuteOp.operand_end(),
        std::back_inserter(loopsToPermute),
        [&](const Value &val) { return loopRefToOp[val]; });

    // Construct permutation map from integer array attribute.
    SmallVector<unsigned int, 4> permuteMap;
    for (const auto &attr : permuteOp.map().getAsRange<IntegerAttr>())
      permuteMap.emplace_back(attr.getValue().getSExtValue());

    // Perform loop permutation.
    permuteLoops(loopsToPermute, permuteMap);

    opsToErase.insert(op);
    return success();
  } else if (auto unrollOp = dyn_cast_or_null<KrnlUnrollOp>(op)) {
    // Unroll the affine for loop fully.
    auto loopRef = unrollOp.loop();
    loopUnrollFull(loopRefToOp[loopRef]);

    opsToErase.insert(op);
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlLoad operation.
//===----------------------------------------------------------------------===//

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class KrnlLoadLowering : public OpRewritePattern<KrnlLoadOp> {
public:
  using OpRewritePattern<KrnlLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlLoadOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlLoadOpAdaptor operandAdaptor = KrnlLoadOpAdaptor(op);

    // Prepare inputs.
    Value memref = operandAdaptor.memref();
    SmallVector<Value, 4> indices = operandAdaptor.indices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<AffineLoadOp>(op, memref, indices);
    else
      rewriter.replaceOpWithNewOp<LoadOp>(op, memref, indices);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: KrnlStore operation.
//===----------------------------------------------------------------------===//

/// KrnlStore will be lowered to std.store or affine.store, depending on whether
/// the access indices are all affine maps or not.
class KrnlStoreLowering : public OpRewritePattern<KrnlStoreOp> {
public:
  using OpRewritePattern<KrnlStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlStoreOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlStoreOpAdaptor operandAdaptor = KrnlStoreOpAdaptor(op);

    // Prepare inputs.
    Value value = operandAdaptor.value();
    Value memref = operandAdaptor.memref();
    SmallVector<Value, 4> indices = operandAdaptor.indices();

    // Check whether all indices are affine maps or not.
    bool affineIndices =
        !llvm::any_of(indices, [](Value v) { return !isValidDim(v); });

    if (affineIndices)
      rewriter.replaceOpWithNewOp<AffineStoreOp>(op, value, memref, indices);
    else
      rewriter.replaceOpWithNewOp<StoreOp>(op, value, memref, indices);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl MatMul operation.
//===----------------------------------------------------------------------===//

using namespace mlir::edsc;
using namespace mlir::edsc::ops;
using namespace mlir::edsc::intrinsics;

#if 0
void genIfThenElseWithoutParams(PatternRewriter &rewriter, Value condition,
    Block *&thenBlock, Block *&elseBlock) {

  // Split current block in the if-conditional block, and the end block.
  Block *ifBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  Block *endBlock = rewriter.splitBlock(ifBlock, opPosition);

  // Then / Else bock.
  thenBlock =
      buildInNewBlock({}, [&](ValueRange args) { std_br(endBlock, {}); });
  elseBlock =
      buildInNewBlock({}, [&](ValueRange args) { std_br(endBlock, {}); });

  // Add the conditional
  appendToBlock(ifBlock, [&](ValueRange) {
    std_cond_br(condition, thenBlock, {}, elseBlock, {});
  });
}
#endif

// KrnlMatmul will be lowered to vector and affine expressions
class KrnlMatmulLowering : public OpRewritePattern<KrnlMatMulOp> {
public:
  using OpRewritePattern<KrnlMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlMatMulOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);

    // Operands and types.
    Type elementType =
        operandAdaptor.A().getType().cast<MemRefType>().getElementType();
    bool simdize = op.simdize();
    bool zeroC = op.zeroC();
    // Init scope and emit constants.
    ScopedContext scope(rewriter, op.getLoc());
    // Generate the test for full/partial blocks
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    MemRefBoundsCapture aBounds(A), bBounds(B), cBounds(C);
    Value nBlock(cBounds.ub(0)), mBlock(cBounds.ub(1)), kBlock(aBounds.ub(1));
    Value nUB(operandAdaptor.nUpperBound()), mUB(operandAdaptor.mUpperBound()),
        kUB(operandAdaptor.kUpperBound()), nGI(operandAdaptor.nGlobalIndex()),
        mGI(operandAdaptor.mGlobalIndex()), kGI(operandAdaptor.kGlobalIndex());
    using namespace edsc::op;

    Value nTrip = nUB - nGI;
    nTrip = std_select(sge(nTrip, nBlock), nBlock, nTrip);
    Value mTrip = mUB - mGI;
    mTrip = std_select(sge(mTrip, mBlock), mBlock, mTrip);
    Value kTrip = kUB - kGI;
    kTrip = std_select(sge(kTrip, kBlock), kBlock, kTrip);

    if (simdize) {
      if (zeroC) {
        Value test = eq(kGI, std_constant_index(0));
        printf("hi alex, start\n");
        Block *thenBlock, *elseBlock;
        genIfThenElseWithoutParams(rewriter, test, thenBlock, elseBlock);
        appendToBlock(thenBlock,
            [&](ValueRange) { genSimd(rewriter, op, elementType, true, false); });
        appendToBlock(elseBlock,
            [&](ValueRange) { genSimd(rewriter, op, elementType, false, true); });
        printf("hi alex, end\n");
      } else {
        genSimd(rewriter, op, elementType, false, false);
      }
    } else {
      genScalar(rewriter, op, elementType, zeroC, false);
    }
    rewriter.eraseOp(op);
    printf("hi alex, done removing op\n");

    return success();
  }

private:
  void genScalar(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      bool zeroC, bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A = operandAdaptor.A();
    Value B = operandAdaptor.B();
    Value C = operandAdaptor.C();

    // Get the EDSC variables, and loop dimensions.
    AffineIndexedValue AA(A), BB(B), CC(C); // Obj we can load and store into.
    MemRefBoundsCapture aBounds(A), bBounds(B), cBounds(C); // Get bounds.
    Value N(cBounds.ub(0)), M(cBounds.ub(1)), K(aBounds.ub(1));
    MemRefType CTmpType = MemRefType::get({}, elementType);

    // For i, j loops.
    using namespace edsc::op;
    Value zero = std_constant_index(0);
    Value fZero(std_constant_float(llvm::APFloat(0.0f), rewriter.getF32Type()));
    Value i, j;
    // clang-format off
    affineLoopNestBuilder({zero, zero}, {N, M}, {1, 1}, [&](ValueRange ivs) {
      // Defines induction variables, and possibly initialize C.
      i = ivs[0]; 
      j = ivs[1];
      // Alloc and init temp c storage.
      Value TmpC = std_alloca(CTmpType);
      AffineIndexedValue TTmpC(TmpC);
      if (zeroC) {
        TTmpC() = fZero;
      } else {
        TTmpC() = CC(i, j);
      }
      // Sum over k.
      affineLoopBuilder(zero, K, 1, [&](Value k) {
        TTmpC() = AA(i, k) * BB(k, j) + TTmpC();
        //TTmpC() = std_fmaf(AA(i, k), BB(k, j), TTmpC());
      });
      // Store temp result into C(i, j)
      CC(i, j) = TTmpC();
    });
    // clang-format on
    if (unrollJam) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto lj = getForInductionVarOwner(j);
      LogicalResult res = loopUnrollJamByFactor(lj, 4);
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genSimd(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      bool zeroC, bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A = operandAdaptor.A();
    Value B = operandAdaptor.B();
    Value C = operandAdaptor.C();
    MemRefType AType = A.getType().cast<MemRefType>();
    MemRefType CType = C.getType().cast<MemRefType>();
    // Find literal value for sizes.
    int64_t NLit = CType.getShape()[0];
    int64_t MLit = CType.getShape()[1];
    int64_t KLit = AType.getShape()[1];
    // Typecast B, C to memrefs of K x vector<M>, N x vector<M>.
    VectorType vecType = VectorType::get({MLit}, elementType);
    MemRefType BVecType = MemRefType::get({KLit}, vecType);
    MemRefType CVecType = MemRefType::get({NLit}, vecType);
    MemRefType CTmpType = MemRefType::get({}, vecType);
    Value BVec = vector_type_cast(BVecType, B);
    Value CVec = vector_type_cast(CVecType, C);
    // Get the EDSC variables, and loop dimensions.
    AffineIndexedValue AA(A), BB(B), CC(C), BBVec(BVec), CCvec(CVec);
    MemRefBoundsCapture aBounds(A), bBounds(B), cBounds(C); // Get bounds.
    Value N(cBounds.ub(0)), M(cBounds.ub(1)), K(aBounds.ub(1));
    // Iterates over the I indices (j are simd dim).
    Value ii;
    using namespace edsc::op;
    Value zero = std_constant_index(0);
    Value fZero(std_constant_float(llvm::APFloat(0.0f), rewriter.getF32Type()));
    // clang-format off
    affineLoopBuilder(zero, M, 1, [&](Value i) {
      ii = i; // Saved for unroll and jam.
      // Alloca temp vector TmpC and save C(i)/0.0 into it.
      Value TmpC = std_alloca(CTmpType);
      AffineIndexedValue TTmpC(TmpC);
      if (zeroC) {
        TTmpC() = vector_broadcast(vecType, fZero);
      } else {
        TTmpC() = CCvec(i);
      }
      // Sum over k.
      affineLoopBuilder(zero, K, 1, [&](Value k) {
        TTmpC() = vector_fma(vector_broadcast(vecType, AA(i, k)), BBVec(k), TTmpC());
      });
      // Store temp result into C(i)
      CCvec(i) = TTmpC();
    });
    // clang-format on
    if (unrollJam) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto li = getForInductionVarOwner(ii);
      LogicalResult res = loopUnrollJamByFactor(li, MLit);
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genIfThenElseWithoutParams(PatternRewriter &rewriter, Value condition,
      function_ref<void(ValueRange)> thenFn,
      function_ref<void(ValueRange)> elseFn) const {

    // Split current block in the if-conditional block, and the end block.
    Block *ifBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *endBlock = rewriter.splitBlock(ifBlock, opPosition);

    // Then / Else bock.
    Block *thenBlock = buildInNewBlock({}, [&](ValueRange args) {
      thenFn(args);
      std_br(endBlock, {});
    });
    Block *elseBlock = buildInNewBlock({}, [&](ValueRange args) {
      elseFn(args);
      std_br(endBlock, {});
    });

    // Add the conditional
    appendToBlock(ifBlock, [&](ValueRange args) {
      std_cond_br(condition, thenBlock, {}, elseBlock, {});
    });
  }
  void genIfThenElseWithoutParams(PatternRewriter &rewriter, Value condition,
      Block *&thenBlock, Block *&elseBlock) const {

    // Split current block in the if-conditional block, and the end block.
    Block *ifBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *endBlock = rewriter.splitBlock(ifBlock, opPosition);

    // Then / Else bock.
    thenBlock = buildInNewBlock({}, [&](ValueRange) { std_br(endBlock, {}); });
    elseBlock = buildInNewBlock({}, [&](ValueRange) { std_br(endBlock, {}); });

    // Add the conditional
    appendToBlock(ifBlock, [&](ValueRange) {
      std_cond_br(condition, thenBlock, {}, elseBlock, {});
    });
  }
};

void ConvertKrnlToAffinePass::runOnFunction() {
  OpBuilder builder(&getContext());
  mlir::Operation *funcOp = getFunction();

  // Interpret krnl dialect operations while looping recursively through
  // operations within the current function, note that erasing operations
  // while iterating is tricky because it can invalidate the iterator, so we
  // collect the operations to be erased in a small ptr set `opsToErase`, and
  // only erase after iteration completes.
  llvm::SmallDenseMap<Value, AffineForOp, 4> loopRefToOp;
  llvm::SmallPtrSet<Operation *, 4> opsToErase;
  if (failed(interpretOperation(funcOp, builder, loopRefToOp, opsToErase))) {
    signalPassFailure();
    return;
  }

  // Remove lowered operations topologically; if ops are not removed
  // topologically, memory error will occur.
  size_t numOpsToRemove = opsToErase.size();
  // Given N operations to remove topologically, and that we remove
  // at least one operation during each pass through opsToErase, we
  // can only have a maximum of N passes through opsToErase.
  for (size_t i = 0; i < numOpsToRemove; i++) {
    for (auto op : opsToErase) {
      if (op->use_empty()) {
        op->erase();
        opsToErase.erase(op);
        // Restart, itr has been invalidated.
        break;
      }
    }
    if (opsToErase.empty())
      break;
  }
  assert(opsToErase.empty());

  ConversionTarget target(getContext());
  target.addIllegalOp<KrnlTerminatorOp>();
  // krnl.dim operations must be lowered prior to this pass.
  target.addIllegalOp<KrnlDimOp>();
  target.addIllegalOp<KrnlMatMulOp>();
  target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect,
      mlir::vector::VectorDialect>();
  OwningRewritePatternList patterns;
  patterns.insert<KrnlTerminatorLowering>(&getContext());
  patterns.insert<KrnlLoadLowering>(&getContext());
  patterns.insert<KrnlStoreLowering>(&getContext());
  patterns.insert<KrnlMatmulLowering>(&getContext());
  DenseSet<Operation *> unconverted;
  if (failed(applyPartialConversion(
          getFunction(), target, std::move(patterns), &unconverted))) {
    printf("alex: failure\n");
    signalPassFailure();
  }
  printf("alex start function dumping\n");
  funcOp->dump();
  printf("alex done function dumping\n");
}
} // namespace

std::unique_ptr<Pass> mlir::createConvertKrnlToAffinePass() {
  return std::make_unique<ConvertKrnlToAffinePass>();
}
