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
#include "src/Dialect/ONNX/IndexExpr.hpp"
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
    LogicalResult res = loopUnrollFull(loopRefToOp[loopRef]);
    assert(res.succeeded() && "full unrolling failed");
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

// KrnlMatmul will be lowered to vector and affine expressions
class KrnlMatmulLowering : public OpRewritePattern<KrnlMatMulOp> {
public:
  using OpRewritePattern<KrnlMatMulOp>::OpRewritePattern;

  PredicateIndexExpr isFullTile(
      IndexExpr UB, IndexExpr block, IndexExpr GI) const {
    // Determine if the current tile is full. It is full if the begining of
    // the tile (nGI) is smaller or equal to UB - bloc, namely
    //   PredicateIndexExpr nIsFullTile = (nGI <= (nUB - nBlock));
    // However, if UB is divisible by Block, then its full no matter what.
    if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
      // Last tile is guaranteed to be full because UB is divisable by block.
      return PredicateIndexExpr(true);
    }
    return GI <= (UB - block);
  }

  IndexExpr trip(IndexExpr UB, IndexExpr block, IndexExpr GI) const {
    // Trip count in general: min(UB - GI, Block).
    //   IndexExpr nTrip = IndexExpr::min(nUB - nGI, nBlock);
    if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
      // Last tile is guaranteed to be full, so trip is always full.
      return block;
    }
    return IndexExpr::min(UB - GI, block);
  }

  IndexExpr partialTrip(IndexExpr UB, IndexExpr block, IndexExpr GI) const {
    // Trip count for partial tiles: leftover = UB - GI in general. If UB is
    // known at compile time, then without loss of generality, leftover = (UB-
    // GI) % Block, and since GI is by definition a multiple of Block (GI is
    // index at begining of tile), then leftover = UB % Block.
    //   IndexExpr nPartialTrip = nUB.isLiteral() ? nUB % nBlock : nUB - nGI;
    if (UB.isLiteral()) {
      IndexExpr partialTrip = UB % block;
      assert(partialTrip.isLiteral() && "op on 2 literals has to be literal");
      assert(partialTrip.getLiteral() > 0 &&
             "here with zero partial trip, we have a problem");
      return partialTrip;
    }
    // don't have to take the mod since we know we have a partial tile already.
    return UB - GI;
  }

  LogicalResult matchAndRewrite(
      KrnlMatMulOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);

    // Operands and types.
    Type elementType =
        operandAdaptor.A().getType().cast<MemRefType>().getElementType();
    bool simdize = op.simdize();
    // Init scope and emit constants.
    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());

    // Generate the test for full/partial blocks
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    MemRefBoundIndexCapture aBounds(A), bBounds(B), cBounds(C);
    IndexExpr nBlock(cBounds.getDim(0)), mBlock(cBounds.getDim(1)),
        kBlock(aBounds.getDim(1));
    assert(nBlock.isLiteral() && "n dim expected to be compile time constant");
    assert(mBlock.isLiteral() && "m dim expected to be compile time constant");
    assert(kBlock.isLiteral() && "k dim expected to be compile time constant");
    SymbolIndexExpr nUB(operandAdaptor.nUpperBound()),
        mUB(operandAdaptor.mUpperBound()), kUB(operandAdaptor.kUpperBound()),
        nGI(operandAdaptor.nGlobalIndex()), mGI(operandAdaptor.mGlobalIndex()),
        kGI(operandAdaptor.kGlobalIndex());

    PredicateIndexExpr nIsFullTile = isFullTile(nUB, nBlock, nGI);
    PredicateIndexExpr mIsFullTile = isFullTile(mUB, mBlock, mGI);
    PredicateIndexExpr kIsFullTile = isFullTile(kUB, kBlock, kGI);
    PredicateIndexExpr fullTiles = (nIsFullTile & mIsFullTile) & kIsFullTile;

    nBlock.debugPrint("N block");
    nUB.debugPrint("N UB");
    nGI.debugPrint("N GI");
    nIsFullTile.debugPrint("n full tile");

    mBlock.debugPrint("M block");
    mUB.debugPrint("M UB");
    mGI.debugPrint("M GI");
    mIsFullTile.debugPrint("m full tile");

    kBlock.debugPrint("K block");
    kUB.debugPrint("K UB");
    kGI.debugPrint("K GI");
    kIsFullTile.debugPrint("k full tile");

    fullTiles.debugPrint("full tile");

    using namespace edsc::op;

    if (simdize) {
      // SIMD code generator.
      // clang-format off
      genIfThenElseWithoutParams(rewriter, !fullTiles,
        /* has some partial tiles */ [&](ValueRange) {
        // Trip regardless of full/partial for N & K
        IndexExpr nTrip = trip(nUB, nBlock, nGI); // May or may not be full.
        IndexExpr kTrip = trip(kUB, kBlock, kGI); // May or may not be full.
        // Test if SIMD dim (M) is full.
        genIfThenElseWithoutParams(rewriter, mIsFullTile,
          /* full SIMD */ [&](ValueRange) {
          genSimd(rewriter, op, elementType, nTrip, mBlock, kTrip, false);
        }, /* else partial SIMD */ [&](ValueRange) {
          IndexExpr mPartialTrip = partialTrip(mUB, mBlock, mGI);
          if (mPartialTrip.isLiteral() && mPartialTrip.getLiteral() >=2) {
            // has a known trip count along the simd dimension of at least 2
            // elements, use simd again.
            genSimd(rewriter, op, elementType, nTrip, mPartialTrip, kTrip, false);
          } else {
            genScalar(rewriter, op, elementType, nTrip, mPartialTrip, kTrip, false);
          }
        });
      }, /* else full */ [&](ValueRange) {
        genSimd(rewriter, op, elementType, nBlock, mBlock, kBlock,
            /*unroll&jam*/true);
      });
      // clang-format on
    } else {
      // Scalar code generator.
      // clang-format off
      genIfThenElseWithoutParams(rewriter, !fullTiles,
        /* partial */ [&](ValueRange) {
        IndexExpr nTrip = trip(nUB, nBlock, nGI); // May or may not be full.
        IndexExpr mTrip = trip(mUB, mBlock, mGI); // May or may not be full.
        IndexExpr kTrip = trip(kUB, kBlock, kGI); // May or may not be full.
        genScalar(rewriter, op, elementType, nTrip, mTrip, kTrip, false);
      }, /* else  full */ [&](ValueRange) {
        genScalar(rewriter, op, elementType, nBlock, mBlock, kBlock, true);
      });
      // clang-format on
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  void genScalar(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      IndexExpr N, IndexExpr M, IndexExpr K, bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor(op);
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());

    // Get the EDSC variables, and loop dimensions.
    AffineIndexedValue AA(A), BB(B), CC(C); // Obj we can load and store into.
    MemRefType CTmpType = MemRefType::get({}, elementType);

    // For i, j loops.
    using namespace edsc::op;
    Value zero = std_constant_index(0);
    Value i, j;
    // clang-format off
    affineLoopNestBuilder({zero, zero}, {N.getValue(), M.getValue()},
        {1, 1}, [&](ValueRange ivs) {
      // Defines induction variables, and possibly initialize C.
      i = ivs[0];
      j = ivs[1];
      // Alloc and init temp c storage.
      Value TmpC = std_alloca(CTmpType);
      AffineIndexedValue TTmpC(TmpC);
      TTmpC() = CC(i, j);
      // Sum over k.
      affineLoopBuilder(zero, K.getValue(), 1, [&](Value k) {
        TTmpC() = AA(i, k) * BB(k, j) + TTmpC();
        //TTmpC() = std_fmaf(AA(i, k), BB(k, j), TTmpC());
      });
      // Store temp result into C(i, j)
      CC(i, j) = TTmpC();
    });
    // clang-format on
    if (unrollJam && M.isLiteral()) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto lj = getForInductionVarOwner(j);
      LogicalResult res = loopUnrollJamByFactor(lj, M.getLiteral());
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genSimd(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      IndexExpr nTrip, IndexExpr mTrip, IndexExpr kTrip, bool unrollJam) const {
    // can simdize only if K is compile time
    assert(mTrip.isLiteral() &&
           "can only simdize with compile time blocking factor on simd axis");
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A = operandAdaptor.A();
    Value B = operandAdaptor.B();
    Value C = operandAdaptor.C();
    MemRefType AType = A.getType().cast<MemRefType>();
    MemRefType CType = C.getType().cast<MemRefType>();
    // Find literal value for sizes of the array. These sizes are compile time
    // constant since its related to the tiling size, decided by the compiler.
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
    // Iterates over the I indices (j are simd dim).
    Value ii;
    using namespace edsc::op;
    Value zero = std_constant_index(0);
    // clang-format off
    affineLoopBuilder(zero, nTrip.getValue(), 1, [&](Value i) {
      ii = i; // Saved for unroll and jam.
      // Alloca temp vector TmpC and save C(i)/0.0 into it.
      Value TmpC = std_alloca(CTmpType);
      AffineIndexedValue TTmpC(TmpC);
      TTmpC() = CCvec(i);
      // Sum over k.
      affineLoopBuilder(zero, kTrip.getValue(), 1, [&](Value k) {
        TTmpC() = vector_fma(vector_broadcast(vecType, AA(i, k)), BBVec(k), TTmpC());
      });
      // Store temp result into C(i)
      Value tmpResults = TTmpC();
      int64_t mTripLit = mTrip.getLiteral();
      if (MLit != mTripLit) {
        // create vector constant
        SmallVector<int64_t, 8> mask;
        for(int64_t i=0; i<MLit; i++)
          mask.emplace_back((i<mTripLit) ? i : MLit+i);
        // permute
        Value originalCvec = CCvec(i);
        tmpResults = rewriter.create<vector::ShuffleOp>(op.getLoc(),
          tmpResults, originalCvec, mask);
      }
      CCvec(i) = tmpResults;
    });
    // clang-format on
    if (unrollJam && nTrip.isLiteral()) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto li = getForInductionVarOwner(ii);
      LogicalResult res = loopUnrollJamByFactor(li, nTrip.getLiteral());
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genIfThenElseWithoutParams(PatternRewriter &rewriter,
      IndexExpr condition, function_ref<void(ValueRange)> thenFn,
      function_ref<void(ValueRange)> elseFn) const {
    Block *ifBlock = rewriter.getInsertionBlock();
    // Handle branches known at compile time.
    if (condition.isLiteral()) {
      if (condition.getLiteral() != 0) {
        // Issue only the then path directly.
        appendToBlock(ifBlock, [&](ValueRange args) { thenFn(args); });
      } else {
        appendToBlock(ifBlock, [&](ValueRange args) { elseFn(args); });
      }
      return;
    }
    // Split current block in the if-conditional block, and the end block.
    auto opPosition = rewriter.getInsertionPoint();
    Block *endBlock = rewriter.splitBlock(ifBlock, opPosition);
    // Construct the empty Then / Else bock.
    Block *thenBlock =
        buildInNewBlock({}, [&](ValueRange args) { std_br(endBlock, {}); });
    Block *elseBlock =
        buildInNewBlock({}, [&](ValueRange args) { std_br(endBlock, {}); });
    // Add the conditional to the If block
    appendToBlock(ifBlock, [&](ValueRange args) {
      std_cond_br(condition.getValue(), thenBlock, {}, elseBlock, {});
    });
    // Has to add the then/else code after fully creating the blocks
    appendToBlock(thenBlock, [&](ValueRange args) { thenFn(args); });
    appendToBlock(elseBlock, [&](ValueRange args) { elseFn(args); });
  }
};

// KrnlCopyToBuffer will be lowered to vector and affine expressions
class KrnlCopyToBufferLowering : public OpRewritePattern<KrnlCopyToBufferOp> {
public:
  using OpRewritePattern<KrnlCopyToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlCopyToBufferOp op, PatternRewriter &rewriter) const override {
    KrnlCopyToBufferOpAdaptor operandAdaptor = KrnlCopyToBufferOpAdaptor(op);
    Value buffMemref(operandAdaptor.bufferMemref());
    Value sourceMemref(operandAdaptor.memref());
    ValueRange starts(operandAdaptor.starts());
    ValueRange sizes(operandAdaptor.sizes());
    Value padVal(operandAdaptor.padValue());
    auto sourceShape = sourceMemref.getType().cast<MemRefType>().getShape();
    int64_t rank = sourceShape.size();
    assert(starts.size() == rank && "starts rank differs from memref");
    assert(sizes.size() == rank && "sizes rank differs from memref");
    assert(buffMemref.getType().cast<MemRefType>().getShape().size() == rank &&
           "buffer and memref should have the same rank");

    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());
    SmallVector<IndexExpr, 4> startIndices, sizeIndices, padUBIndices;
    MemRefBoundIndexCapture buffBounds(buffMemref);
    getIndexExprList<SymbolIndexExpr>(starts, startIndices);
    getIndexExprList<SymbolIndexExpr>(sizes, sizeIndices);
    ArrayAttributeIndexCapture padCapture(op.padsAttr(), 1);
    LiteralIndexExpr one(1);
    for (long i = 0; i < rank; ++i) {
      IndexExpr amount = padCapture.getLiteral(i);
      int64_t amountVal = amount.getLiteral(); // Will assert if undefined.
      IndexExpr UB = buffBounds.getDim(i);
      int64_t UBval = UB.getLiteral(); // Will assert if not literal.
      if (amountVal == 1) {
        // Add pad % 1... namely no pad, put one.
        padUBIndices.emplace_back(one);
      } else if (amountVal == 0 || amountVal == UBval) {
        // Full pad (0 pad is the same as full pad).
        if (sizeIndices[i].isLiteral() &&
            sizeIndices[i].getLiteral() == UBval) {
          // We are already filling the whole buffer with data, no need to pad.
          padUBIndices.emplace_back(one);
        } else {
          // We will fill to the end, namely UB.
          padUBIndices.emplace_back(UB);
        }
      } else {
        assert(amountVal > 0 && amountVal < UBval && "out of range pad");
        IndexExpr newUB = (sizeIndices[i].ceilDiv(amount)) * amount;
        padUBIndices.emplace_back(newUB);
      }
    }
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zero(0);
    genCopyLoops(buffMemref, sourceMemref, padVal, zero, startIndices,
        sizeIndices, padUBIndices, loopIndices, 0, rank, false);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(Value buffMemref, Value sourceMemref, Value padVal,
      IndexExpr zero, SmallVectorImpl<IndexExpr> &startIndices,
      SmallVectorImpl<IndexExpr> &sizeIndices,
      SmallVectorImpl<IndexExpr> &padUBIndices,
      SmallVectorImpl<Value> &loopIndices, int64_t i, int64_t rank,
      bool padPhase) const {
    if (i == rank) {
      // create new scope and import index expressions
      IndexExprScope currScope;
      SmallVector<IndexExpr, 4> currLoopIndices, currStartIndices;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      getIndexExprList<SymbolIndexExpr>(startIndices, currStartIndices);
      Value sourceVal;
      if (!padPhase) {
        SmallVector<IndexExpr, 4> currLoadIndices;
        for (long i = 0; i < rank; ++i) {
          currLoadIndices.emplace_back(
              currLoopIndices[i] + currStartIndices[i]);
        }
        sourceVal = krnl_load(sourceMemref, currLoadIndices);
      } else {
        sourceVal = padVal;
      }
      krnl_store(sourceVal, buffMemref, currLoopIndices);
    } else {
      using namespace edsc::op;
      Value size = sizeIndices[i].getValue();
      if (!sizeIndices[i].isLiteralAndIdenticalTo(0)) {
        // Loop to copy the data.
        affineLoopBuilder(zero.getValue(), size, 1, [&](Value index) {
          loopIndices.emplace_back(index);
          genCopyLoops(buffMemref, sourceMemref, padVal, zero, startIndices,
              sizeIndices, padUBIndices, loopIndices, i + 1, rank,
              /*no pad phase*/ false);
          loopIndices.pop_back_n(1);
        });
      }
      if (!padUBIndices[i].isLiteralAndIdenticalTo(1)) {
        // Need some padding at this level.
        affineLoopBuilder(
            size, padUBIndices[i].getValue(), 1, [&](Value index) {
              loopIndices.emplace_back(index);
              genCopyLoops(buffMemref, sourceMemref, padVal, zero, startIndices,
                  sizeIndices, padUBIndices, loopIndices, i + 1, rank,
                  /*pad phase*/ true);
              loopIndices.pop_back_n(1);
            });
      }
      // For next level up of padding, if any, will not copy data anymore
      sizeIndices[i] = zero;
    }
  }
};

// KrnlCopyFromBuffer will be lowered to vector and affine expressions
class KrnlCopyFromBufferLowering
    : public OpRewritePattern<KrnlCopyFromBufferOp> {
public:
  using OpRewritePattern<KrnlCopyFromBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlCopyFromBufferOp op, PatternRewriter &rewriter) const override {
    KrnlCopyFromBufferOpAdaptor operandAdaptor =
        KrnlCopyFromBufferOpAdaptor(op);
    Value buffMemref(operandAdaptor.bufferMemref());
    Value sourceMemref(operandAdaptor.memref());
    ValueRange starts(operandAdaptor.starts());
    ValueRange sizes(operandAdaptor.sizes());
    auto sourceShape = sourceMemref.getType().cast<MemRefType>().getShape();
    int64_t rank = sourceShape.size();
    assert(starts.size() == rank && "starts rank differs from memref");
    assert(sizes.size() == rank && "sizes rank differs from memref");
    assert(buffMemref.getType().cast<MemRefType>().getShape().size() == rank &&
           "buffer and memref should have the same rank");

    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());
    SmallVector<IndexExpr, 4> startIndices, sizeIndices;
    MemRefBoundIndexCapture buffBounds(buffMemref);
    getIndexExprList<SymbolIndexExpr>(starts, startIndices);
    getIndexExprList<SymbolIndexExpr>(sizes, sizeIndices);
    LiteralIndexExpr one(1);
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zero(0);
    genCopyLoops(buffMemref, sourceMemref, zero, startIndices, sizeIndices,
        loopIndices, 0, rank);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(Value buffMemref, Value sourceMemref, IndexExpr zero,
      SmallVectorImpl<IndexExpr> &startIndices,
      SmallVectorImpl<IndexExpr> &sizeIndices,
      SmallVectorImpl<Value> &loopIndices, int64_t i, int64_t rank) const {
    if (i == rank) {
      // create new scope and import index expressions
      IndexExprScope currScope;
      SmallVector<IndexExpr, 4> currLoopIndices, currStartIndices;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      getIndexExprList<SymbolIndexExpr>(startIndices, currStartIndices);
      SmallVector<IndexExpr, 4> currStoreIndices;
      for (long i = 0; i < rank; ++i) {
        currStoreIndices.emplace_back(currLoopIndices[i] + currStartIndices[i]);
      }
      Value sourceVal = krnl_load(sourceMemref, currLoopIndices);
      krnl_store(sourceVal, buffMemref, currStoreIndices);
    } else {
      using namespace edsc::op;
      Value size = sizeIndices[i].getValue();
      if (!sizeIndices[i].isLiteralAndIdenticalTo(0)) {
        // Loop to copy the data.
        affineLoopBuilder(zero.getValue(), size, 1, [&](Value index) {
          loopIndices.emplace_back(index);
          genCopyLoops(buffMemref, sourceMemref, zero, startIndices,
              sizeIndices, loopIndices, i + 1, rank);
          loopIndices.pop_back_n(1);
        });
      }
    }
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
  target.addIllegalOp<KrnlCopyToBufferOp>();
  target.addIllegalOp<KrnlCopyFromBufferOp>();
  target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect,
      mlir::vector::VectorDialect>();
  OwningRewritePatternList patterns;
  patterns.insert<KrnlTerminatorLowering>(&getContext());
  patterns.insert<KrnlLoadLowering>(&getContext());
  patterns.insert<KrnlStoreLowering>(&getContext());
  patterns.insert<KrnlMatmulLowering>(&getContext());
  patterns.insert<KrnlCopyToBufferLowering>(&getContext());
  patterns.insert<KrnlCopyFromBufferLowering>(&getContext());
  DenseSet<Operation *> unconverted;
  if (failed(applyPartialConversion(
          getFunction(), target, std::move(patterns), &unconverted))) {
    signalPassFailure();
  }
}
} // namespace

std::unique_ptr<Pass> mlir::createConvertKrnlToAffinePass() {
  return std::make_unique<ConvertKrnlToAffinePass>();
}
