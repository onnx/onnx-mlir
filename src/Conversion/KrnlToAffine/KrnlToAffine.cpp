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
#include "mlir/IR/IntegerSet.h"
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

static void affineLoopBuilder(ValueRange lbOperands, AffineMap &lbMap,
    ValueRange ubOperands, AffineMap &ubMap, int64_t step,
    function_ref<void(Value)> bodyBuilderFn) {
  // Fetch the builder and location.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();

  // Create the actual loop and call the body builder, if provided, after
  // updating the scoped context.
  builder.create<AffineForOp>(loc, lbOperands, lbMap, ubOperands, ubMap, step,
      llvm::None,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
        }
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });
}

static void affineLoopBuilder(IndexExpr lb, IndexExpr ub, int64_t step,
    function_ref<void(Value)> bodyBuilderFn) {
  AffineMap lbMap, ubMap;
  SmallVector<Value, 8> lbOperands, ubOperands;
  lb.getAffineMapAndOperands(lbMap, lbOperands);
  ub.getAffineMapAndOperands(ubMap, ubOperands);
  affineLoopBuilder(lbOperands, lbMap, ubOperands, ubMap, step, bodyBuilderFn);
}

// Affine expressions compared to >= 0
static IndexExpr isFullTile(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Determine if the current tile is full. It is full if the begining of
  // the tile (nGI) is smaller or equal to UB - bloc, namely
  //   PredicateIndexExpr nIsFullTile = (nGI <= (nUB - nBlock));
  // However, if UB is divisible by Block, then its full no matter what.
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full because UB is divisable by block.
    return LiteralIndexExpr(1); // 1 >= 0 is true
  }
  // true if GI <= (UB - block), namely UB - block - GI >= 0
  IndexExpr res = UB - block - GI;
  return res;
}

static IndexExpr trip(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Trip count in general: min(UB - GI, Block).
  UB.debugPrint("trip up");
  block.debugPrint("trip block");
  GI.debugPrint("trip GI");
  //   IndexExpr nTrip = IndexExpr::min(nUB - nGI, nBlock);
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full, so trip is always full.
    return block;
  }
  return IndexExpr::min(UB - GI, block);
}

static IndexExpr partialTrip(IndexExpr UB, IndexExpr block, IndexExpr GI) {
  // Trip count for partial tiles: leftover = UB - GI in general. If UB is
  // known at compile time, then without loss of generality, leftover = (UB-
  // GI) % Block, and since GI is by definition a multiple of Block (GI is
  // index at begining of tile), then leftover = UB % Block.
  //   IndexExpr nPartialTrip = nUB.isLiteral() ? nUB % nBlock : nUB - nGI;
  if (UB.isLiteral()) {
    IndexExpr partialTrip = UB % block;
    assert(partialTrip.isLiteral() && "op on 2 literals has to be literal");
    return partialTrip;
  }
  // don't have to take the mod since we know we have a partial tile already.
  return UB - GI;
}

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
    // Init scope and emit constants.
    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());

    // Gather A, B, C tile sizes.
    SmallVector<IndexExpr, 2> ATileSize, BTileSize, CTileSize;
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());
    MemRefBoundIndexCapture aBounds(A), bBounds(B), cBounds(C);
    // Tile sizes for A/B/C are determined by their memref unless explicitly
    // specified by an optional argument. That allows A/B/C memrefs to be
    // padded if needed for SIMD/unroll and jam, for example.
    aBounds.getLiteralList(ATileSize);
    ArrayAttributeIndexCapture ASizeCapture(op.ATileSizeAttr());
    if (ASizeCapture.size()) {
      int memSize0 = ATileSize[0].getLiteral();
      int memSize1 = ATileSize[1].getLiteral();
      ATileSize = {ASizeCapture.getLiteral(0), ASizeCapture.getLiteral(1)};
      assert(ATileSize[0].getLiteral() <= memSize0 &&
             "A tile size cannot be larger than the A buffer size (0 dim)");
      assert(ATileSize[1].getLiteral() <= memSize1 &&
             "A tile size cannot be larger than the A buffer size (1 dim)");
    }
    bBounds.getLiteralList(BTileSize);
    ArrayAttributeIndexCapture BSizeCapture(op.BTileSizeAttr());
    if (BSizeCapture.size()) {
      int memSize0 = BTileSize[0].getLiteral();
      int memSize1 = BTileSize[1].getLiteral();
      BTileSize = {BSizeCapture.getLiteral(0), BSizeCapture.getLiteral(1)};
      assert(BTileSize[0].getLiteral() <= memSize0 &&
             "B tile size cannot be larger than the B buffer size (0 dim)");
      assert(BTileSize[1].getLiteral() <= memSize1 &&
             "B tile size cannot be larger than the B buffer size (1 dim)");
    }
    cBounds.getLiteralList(CTileSize);
    ArrayAttributeIndexCapture CSizeCapture(op.CTileSizeAttr());
    if (CSizeCapture.size()) {
      int memSize0 = CTileSize[0].getLiteral();
      int memSize1 = CTileSize[1].getLiteral();
      CTileSize = {CSizeCapture.getLiteral(0), CSizeCapture.getLiteral(1)};
      assert(CTileSize[0].getLiteral() <= memSize0 &&
             "C tile size cannot be larger than the C buffer size (0 dim)");
      assert(CTileSize[1].getLiteral() <= memSize1 &&
             "C tile size cannot be larger than the C buffer size (1 dim)");
    }
    ATileSize[0].debugPrint("a tile size 0");
    ATileSize[1].debugPrint("a tile size 1");
    BTileSize[0].debugPrint("b tile size 0");
    BTileSize[1].debugPrint("b tile size 1");
    CTileSize[0].debugPrint("c tile size 0");
    CTileSize[1].debugPrint("c tile size 1");
    // Check consitency. If the dimensions were padded differently for
    // A, B, or C, and the optional A/B/C TileSize attributes were given,
    // we take the these optional sizes into consideration. Meaning, we
    // don't really care about the padded dimensions, we only care about
    // the actual data.
    assert(ATileSize[0].getLiteral() == CTileSize[0].getLiteral() &&
           "N dim mismatch");
    assert(ATileSize[1].getLiteral() == BTileSize[0].getLiteral() &&
           "K dim mismatch");
    assert(BTileSize[1].getLiteral() == CTileSize[1].getLiteral() &&
           "M dim mismatch");

    // Gather N, M, K compute tile size. This is the size of the computations,
    // if the tile is full. Because computation in the buffers could be further
    // subtiled, the default size can be overridden from the tile sizes using
    // the computeTileSize attribute. Tiles may not be full if they are at the
    // outer boundaries of the original data.
    IndexExpr nComputeTileSize = CTileSize[0];
    IndexExpr mComputeTileSize = CTileSize[1];
    IndexExpr kComputeTileSize = ATileSize[1];
    ArrayAttributeIndexCapture computeSizeCapture(op.computeTileSizeAttr());
    if (computeSizeCapture.size()) {
      int64_t nSize = nComputeTileSize.getLiteral();
      int64_t mSize = mComputeTileSize.getLiteral();
      int64_t kSize = kComputeTileSize.getLiteral();
      nComputeTileSize = computeSizeCapture.getLiteral(0);
      mComputeTileSize = computeSizeCapture.getLiteral(1);
      kComputeTileSize = computeSizeCapture.getLiteral(2);
      assert(nComputeTileSize.getLiteral() <= nSize &&
             "read n TileSize cannot be larger than the n size");
      assert(mComputeTileSize.getLiteral() <= mSize &&
             "read m TileSize cannot be larger than the n size");
      assert(kComputeTileSize.getLiteral() <= kSize &&
             "read k TileSize cannot be larger than the n size");
    }
    // A,B,C tile dimensions must be multiples of n/m/k compute tile sizes. This
    // means: no partial tiles allowed when organizing the sub-tiling within the
    // buffers. This is not a statement about data, as A, B, C buffers can be
    // padded, as long as their logical sizes are given explicitly.
    assert(ATileSize[0].getLiteral() % nComputeTileSize.getLiteral() == 0 &&
           "A tile size not multiple of n compute tile size");
    assert(ATileSize[1].getLiteral() % kComputeTileSize.getLiteral() == 0 &&
           "A tile size not multiple of k compute tile size");
    assert(BTileSize[0].getLiteral() % kComputeTileSize.getLiteral() == 0 &&
           "B tile size not multiple of k compute tile size");
    assert(BTileSize[1].getLiteral() % mComputeTileSize.getLiteral() == 0 &&
           "B tile size not multiple of m compute tile size");
    assert(CTileSize[0].getLiteral() % nComputeTileSize.getLiteral() == 0 &&
           "C tile size not multiple of n compute tile size");
    assert(CTileSize[1].getLiteral() % mComputeTileSize.getLiteral() == 0 &&
           "C tile size not multiple of m compute tile size");

    // Now get global start indices, which would define the first element of the
    // tiles in the original computations.
    DimIndexExpr nGlobalStart(operandAdaptor.nGlobalStart()),
        mGlobalStart(operandAdaptor.mGlobalStart()),
        kGlobalStart(operandAdaptor.kGlobalStart());
    // And get the global upper bound of the original computations.
    DimIndexExpr nGlobalUB(operandAdaptor.nGlobalUB()),
        mGlobalUB(operandAdaptor.mGlobalUB()),
        kGlobalUB(operandAdaptor.kGlobalUB());

    // Compute the start for each matrices (A[NxK], B[KxM], C[NxM]). Starts are
    // the offset in the buffers A, B, and C.
    IndexExpr AStart0(nGlobalStart % ATileSize[0]);
    IndexExpr AStart1(kGlobalStart % ATileSize[1]);
    IndexExpr BStart0(kGlobalStart % BTileSize[0]);
    IndexExpr BStart1(mGlobalStart % BTileSize[1]);
    IndexExpr CStart0(nGlobalStart % CTileSize[0]);
    IndexExpr CStart1(mGlobalStart % CTileSize[1]);

    // Simdize along M for the full compute tile
    IndexExpr vectorLen = mComputeTileSize;
    IndexExpr BVecStart1 = BStart1.ceilDiv(vectorLen);
    IndexExpr CVecStart1 = CStart1.ceilDiv(vectorLen);

    // Now determine if we have full/partial tiles. This is determined by the
    // outer dimensions of the original computations, as by definition tiling
    // within the buffer always results in full tiles. In other words, partial
    // tiles only occurs because of "runing out" of the original data.
    IndexExpr nIsFullTile =
        isFullTile(nGlobalUB, nComputeTileSize, nGlobalStart);
    IndexExpr mIsFullTile =
        isFullTile(mGlobalUB, mComputeTileSize, mGlobalStart);
    IndexExpr kIsFullTile =
        isFullTile(kGlobalUB, kComputeTileSize, kGlobalStart);
    SmallVector<IndexExpr, 4> allFullTiles = {
        nIsFullTile, mIsFullTile, kIsFullTile};
    SmallVector<IndexExpr, 4> mFullTiles = {mIsFullTile};
    // And if the tiles are not full, determine how many elements to compute.
    // With overcompute, this could be relaxed.
    IndexExpr nTrip = trip(
        nGlobalUB, nComputeTileSize, nGlobalStart); // May or may not be full.
    IndexExpr mTrip = trip(
        mGlobalUB, mComputeTileSize, mGlobalStart); // May or may not be full.
    IndexExpr kTrip = trip(
        kGlobalUB, kComputeTileSize, kGlobalStart); // May or may not be full.
    IndexExpr mPartialTrip =
        partialTrip(mGlobalUB, mComputeTileSize, mGlobalStart);

    using namespace edsc::op;

    if (simdize) {
      // SIMD code generator.
      // clang-format off
      genIfThenElseWithoutParams(rewriter, allFullTiles,
        /* then full */ [&](ValueRange) {
        genSimd(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
          BVecStart1,  CStart0,  CVecStart1, nComputeTileSize, mComputeTileSize, 
          kComputeTileSize,  vectorLen, false); // true
      }, /* has some partial tiles */ [&](ValueRange) {
        // Trip regardless of full/partial for N & K
        // Test if SIMD dim (M) is full.
        genIfThenElseWithoutParams(rewriter, mFullTiles,
          /* full SIMD */ [&](ValueRange) {
          genSimd(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
            BVecStart1,  CStart0,  CVecStart1, nTrip, mComputeTileSize, kTrip, 
            vectorLen, false);
        }, /* else partial SIMD */ [&](ValueRange) {
          if (mPartialTrip.isLiteral() && mPartialTrip.getLiteral() >=2) {
            // has a known trip count along the simd dimension of at least 2
            // elements, use simd again.
            genSimd(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
              BVecStart1,  CStart0,  CVecStart1, nTrip, mPartialTrip, kTrip, 
              vectorLen, false);
          } else {
            genScalar(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
              BStart1,  CStart0,  CStart1, nTrip, mPartialTrip, kTrip, false);
          }
        });
      });
      // clang-format on
    } else {
      // Scalar code generator.
      // clang-format off
      genIfThenElseWithoutParams(rewriter, allFullTiles,
        /* then full */ [&](ValueRange) {
        genScalar(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
          BStart1,  CStart0,  CStart1, nComputeTileSize, mComputeTileSize, 
          kComputeTileSize, false); // true
      }, /* else partial */ [&](ValueRange) {
        genScalar(rewriter, op, elementType, AStart0,  AStart1,  BStart0,
          BStart1,  CStart0,  CStart1, nTrip, mTrip, kTrip, false);
      });
      // clang-format on
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  void genScalar(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      IndexExpr AStart0, IndexExpr AStart1, IndexExpr BStart0,
      IndexExpr BStart1, IndexExpr CStart0, IndexExpr CStart1, IndexExpr N,
      IndexExpr M, IndexExpr K, bool unrollJam) const {
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor(op);
    Value A(operandAdaptor.A()), B(operandAdaptor.B()), C(operandAdaptor.C());

    // IndexExprScope newScope;
    // SymbolIndexExpr NN(N), MM(M), KK(K);
    // DimIndexExpr nnStart(nStart), mmStart(mStart), kkStart(kStart);

    // Get the EDSC variables, and loop dimensions.
    AffineIndexedValue AA(A), BB(B), CC(C); // Obj we can load and store into.
    MemRefType CTmpType = MemRefType::get({}, elementType);

    // For i, j loops.
    using namespace edsc::op;
    LiteralIndexExpr zero(0);
    Value jSaved;
    // clang-format off
    affineLoopBuilder(zero, N, 1, [&](Value i) {
      affineLoopBuilder(zero, M, 1, [&](Value j) {
        // Defines induction variables, and possibly initialize C.
        jSaved = j;
        // Alloc and init temp c storage.
        Value TmpC = std_alloca(CTmpType);
        AffineIndexedValue TTmpC(TmpC);
        TTmpC() = CC(i + CStart0.getValue(), j + CStart1.getValue());
        // Sum over k.
        affineLoopBuilder(zero, K, 1, [&](Value k) {
          TTmpC() = AA(i + AStart0.getValue(), k + AStart1.getValue()) *
            BB(k + BStart0.getValue(), j + BStart1.getValue()) +
            TTmpC();
        });
        // Store temp result into C(i, j)
        CC(i + CStart0.getValue(), j + CStart1.getValue()) = TTmpC();
      });
    });
    // clang-format on
    if (unrollJam && M.isLiteral()) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto lj = getForInductionVarOwner(jSaved);
      LogicalResult res = loopUnrollJamByFactor(lj, M.getLiteral());
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genSimd(PatternRewriter &rewriter, KrnlMatMulOp op, Type elementType,
      IndexExpr AStart0, IndexExpr AStart1, IndexExpr BStart0,
      IndexExpr BVecStart1, IndexExpr CStart0, IndexExpr CVecStart1,
      IndexExpr nTrip, IndexExpr mTrip, IndexExpr kTrip, IndexExpr vectorLen,
      bool unrollJam) const {
    // can simdize only if K is compile time
    assert(mTrip.isLiteral() &&
           "can only simdize with compile time blocking factor on simd axis");
    // Get operands.
    KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
    Value A = operandAdaptor.A();
    Value B = operandAdaptor.B();
    Value C = operandAdaptor.C();
    MemRefType BType = B.getType().cast<MemRefType>();
    MemRefType CType = C.getType().cast<MemRefType>();

    // Generate the vector type conversions.
    int64_t VL = vectorLen.getLiteral();
    VectorType vecType = VectorType::get({VL}, elementType);
    MemRefType CTmpType = MemRefType::get({}, vecType);

    // Find literal value for sizes of the array. These sizes are compile time
    // constant since its related to the tiling size, decided by the compiler.
    int64_t KLitB = BType.getShape()[0];
    int64_t MLitB = BType.getShape()[1];
    assert(MLitB % VL == 0 && "expected M dim to be a multiple of VL");
    // Typecast B to memrefs of K x (M/VL) x vector<VL>.
    MemRefType BVecType = MemRefType::get({KLitB, MLitB / VL}, vecType);
    Value BVec =
        rewriter.create<KrnlVectorTypeCastOp>(op.getLoc(), BVecType, B);

    int64_t NLitC = CType.getShape()[0];
    int64_t MLitC = CType.getShape()[1];
    assert(MLitC % VL == 0 && "expected M dim to be a multiple of VL");
    // Typecast C to memrefs of N x (M/VL) x vector<VL>.
    MemRefType CVecType = MemRefType::get({NLitC, MLitC / VL}, vecType);
    Value CVec =
        rewriter.create<KrnlVectorTypeCastOp>(op.getLoc(), CVecType, C);

    // IndexExprScope newScope;
    // SymbolIndexExpr nnTrip(nTrip), mmTrip(mTrip), kkTrip(kTrip);
    // DimIndexExpr nnStart(nStart), mmStart(mStart), kkStart(kStart);

    // Get the EDSC variables, and loop dimensions.
    AffineIndexedValue AA(A), BB(B), CC(C), BBVec(BVec), CCvec(CVec);
    // Iterates over the I indices (j are simd dim).
    Value iSaved;
    using namespace edsc::op;
    LiteralIndexExpr zero(0);
    // clang-format off
    affineLoopBuilder(zero, nTrip, 1, [&](Value i) {
      iSaved = i; // Saved for unroll and jam.
      // Alloca temp vector TmpC and save C(i)/0.0 into it.
      Value TmpC = std_alloca(CTmpType);
      AffineIndexedValue TTmpC(TmpC);
      TTmpC() = CCvec(i + CStart0.getValue(), CVecStart1.getValue());
      // Sum over k.
      affineLoopBuilder(zero, kTrip, 1, [&](Value k) {
        Value a = AA(i + AStart0.getValue(), k + AStart1.getValue());
        Value va = vector_broadcast(vecType, a);
        Value vb = BBVec(k + BStart0.getValue(), BVecStart1.getValue());
        TTmpC() = vector_fma(va, vb, TTmpC());
      });
      // Store temp result into C(i)
      Value tmpResults = TTmpC();
      int64_t mTripLit = mTrip.getLiteral();
      if (mTripLit != VL) {
        // create vector constant
        SmallVector<int64_t, 8> mask;
        for(int64_t i=0; i<VL; i++)
          mask.emplace_back((i<mTripLit) ? i : VL+i);
        // permute
        Value originalCvec = CCvec(i + CStart0.getValue(), CVecStart1.getValue());
        tmpResults = rewriter.create<vector::ShuffleOp>(op.getLoc(),
          tmpResults, originalCvec, mask);
      }
      CCvec(i + CStart0.getValue(), CVecStart1.getValue()) = tmpResults;
    });
    // clang-format on
    if (unrollJam && nTrip.isLiteral()) {
      // Unroll and jam. Seems to support only one operation at this time.
      auto li = getForInductionVarOwner(iSaved);
      LogicalResult res = loopUnrollJamByFactor(li, nTrip.getLiteral());
      assert(res.succeeded() && "failed to optimize");
    }
  }

  void genIfThenElseWithoutParams(PatternRewriter &rewriter,
      SmallVectorImpl<IndexExpr> &conditions,
      function_ref<void(ValueRange)> thenFn,
      function_ref<void(ValueRange)> elseFn) const {

    IndexExprScope &scope = IndexExprScope::getCurrentScope();
    int64_t rank = conditions.size();
    SmallVector<bool, 4> isEq(rank, false);
    SmallVector<AffineExpr, 4> affineCond;
    bool allTrue = true;
    bool allFalse = true;
    for (IndexExpr i : conditions) {
      assert(i.isAffine() && "conditions expected to be affine");
      affineCond.emplace_back(i.getAffineExpr());
      if (i.isLiteral()) {
        if (i.getLiteral() < 0) // Inequality is expr >= 0, test if false.
          allTrue = false;
        if (i.getLiteral() >= 0) // Inequality is expr >= 0, test if true.
          allFalse = false;
      } else {
        allTrue = false;
        allFalse = false;
      }
    }
    auto inset = IntegerSet::get(
        scope.getNumDims(), scope.getNumSymbols(), affineCond, isEq);
    SmallVector<Value, 8> dimAndSymbolList;
    scope.getDimAndSymbolList(dimAndSymbolList);
    auto ifOp = rewriter.create<AffineIfOp>(
        scope.getLoc(), inset, dimAndSymbolList, true);
    Block *thenBlock = ifOp.getThenBlock();
    Block *elseBlock = ifOp.getElseBlock();
    if (!allFalse) {
      appendToBlock(thenBlock, [&](ValueRange args) { thenFn(args); });
    }
    if (!allTrue) {
      appendToBlock(elseBlock, [&](ValueRange args) { elseFn(args); });
    }
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl Copy to Buffer operation.
//===----------------------------------------------------------------------===//

class KrnlCopyToBufferLowering : public OpRewritePattern<KrnlCopyToBufferOp> {
public:
  using OpRewritePattern<KrnlCopyToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlCopyToBufferOp op, PatternRewriter &rewriter) const override {
    // Get info from operands.
    KrnlCopyToBufferOpAdaptor operandAdaptor = KrnlCopyToBufferOpAdaptor(op);
    Value buffMemref(operandAdaptor.bufferMemref());
    Value sourceMemref(operandAdaptor.memref());
    ValueRange startVals(operandAdaptor.starts());
    Value padVal(operandAdaptor.padValue());
    int64_t rank = sourceMemref.getType().cast<MemRefType>().getShape().size();
    assert(startVals.size() == rank && "starts rank differs from memref");
    assert(buffMemref.getType().cast<MemRefType>().getShape().size() == rank &&
           "buffer and memref should have the same rank");

    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());
    SmallVector<IndexExpr, 4> starts, bufferReadUBs, bufferPadUBs;
    MemRefBoundIndexCapture buffBounds(buffMemref);
    MemRefBoundIndexCapture sourceBounds(sourceMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    ArrayAttributeIndexCapture padCapture(op.padToNextAttr(), 1);
    assert((padCapture.size() == 0 || padCapture.size() == rank) &&
           "optional padToNext rank differs from memref");
    ArrayAttributeIndexCapture readSizeCapture(op.tileSizeAttr());
    assert((readSizeCapture.size() == 0 || readSizeCapture.size() == rank) &&
           "optional readTileSize rank differs from memref");
    // Overread not currently used, will if we simdize reads or
    // unroll and jam loops.
    // ArrayAttributeIndexCapture overCapture(op.overreadToNextAttr(), 1);

    // Determine here bufferReadUBs, which determine how many values of source
    // memeref to copy into the buffer. Also determine bufferPadUBs, which is
    // the upper bound past bufferReadUBs that must be padded.
    LiteralIndexExpr zero(0);
    for (long i = 0; i < rank; ++i) {
      // Compute how many values to read.
      IndexExpr sourceBound = sourceBounds.getSymbol(i); // Source memref size.
      IndexExpr blockSize = buffBounds.getSymbol(i);     // Buffer memref size.
      if (readSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = readSizeCapture.getLiteral(i); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "readTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI = starts[i]; // Global index in source memref of tile.
      IndexExpr bufferRead = trip(sourceBound, blockSize, startGI);
      bufferRead.debugPrint("buffer read");
      bufferReadUBs.emplace_back(bufferRead);
      // Determine the UB until which to pad
      IndexExpr padToNext = padCapture.getLiteral(i);
      int64_t padToNextLit =
          padToNext.getLiteral(); // Will assert if undefined.
      int64_t blockSizeLit = blockSize.getLiteral(); // Will assert if not lit.
      if (bufferRead.isLiteralAndIdenticalTo(blockSizeLit)) {
        // Read the full buffer already, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (bufferRead.isLiteral() &&
                 bufferRead.getLiteral() % padToNextLit == 0) {
        // We are already reading to the end of a line.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == 1) {
        // Add pad % 1... namely no pad, nothing to do.
        bufferPadUBs.emplace_back(zero);
      } else if (padToNextLit == blockSizeLit) {
        // Pad to end.
        bufferPadUBs.emplace_back(blockSize);
      } else {
        assert(padToNextLit > 1 && padToNextLit < blockSizeLit &&
               "out of range padToLit");
        IndexExpr newPadUB = (bufferRead.ceilDiv(padToNext)) * padToNext;
        bufferPadUBs.emplace_back(newPadUB);
      }
    }
    SmallVector<Value, 4> loopIndices;
    genCopyLoops(buffMemref, sourceMemref, padVal, zero, starts, bufferReadUBs,
        bufferPadUBs, loopIndices, 0, rank, false);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(Value buffMemref, Value sourceMemref, Value padVal,
      IndexExpr zero, SmallVectorImpl<IndexExpr> &starts,
      SmallVectorImpl<IndexExpr> &readUBs, SmallVectorImpl<IndexExpr> &padUBs,
      SmallVectorImpl<Value> &loopIndices, int64_t i, int64_t rank,
      bool padPhase) const {
    if (i == rank) {
      // create new scope and import index expressions
      IndexExprScope currScope;
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      if (!padPhase) {
        SmallVector<IndexExpr, 4> currLoadIndices;
        getIndexExprList<DimIndexExpr>(starts, currStarts);
        for (long i = 0; i < rank; ++i) {
          currLoadIndices.emplace_back(currLoopIndices[i] + currStarts[i]);
        }
        Value sourceVal = krnl_load(sourceMemref, currLoadIndices);
        krnl_store(sourceVal, buffMemref, currLoopIndices);
      } else {
        krnl_store(padVal, buffMemref, currLoopIndices);
      }
    } else {
      using namespace edsc::op;
      Value readUBVal = readUBs[i].getValue();
      if (readUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to read, skip.
      } else {
        affineLoopBuilder(zero, readUBs[i], 1, [&](Value index) {
          loopIndices.emplace_back(index);
          genCopyLoops(buffMemref, sourceMemref, padVal, zero, starts, readUBs,
              padUBs, loopIndices, i + 1, rank,
              /*no pad phase*/ false);
          loopIndices.pop_back_n(1);
        });
      }
      if (padUBs[i].isLiteralAndIdenticalTo(0)) {
        // No padding needed.
      } else {
        affineLoopBuilder(readUBs[i], padUBs[i], 1, [&](Value index) {
          loopIndices.emplace_back(index);
          genCopyLoops(buffMemref, sourceMemref, padVal, zero, starts, readUBs,
              padUBs, loopIndices, i + 1, rank,
              /*pad phase*/ true);
          loopIndices.pop_back_n(1);
        });
      }
      // For next level up of padding, if any, will not copy data anymore
      readUBs[i] = zero;
    }
  }
};

//===----------------------------------------------------------------------===//
// Krnl to Affine Rewrite Patterns: Krnl Copy from Buffer operation.
//===----------------------------------------------------------------------===//

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
    ValueRange startVals(operandAdaptor.starts());
    int64_t rank = sourceMemref.getType().cast<MemRefType>().getShape().size();
    assert(startVals.size() == rank && "starts rank differs from memref");
    assert(buffMemref.getType().cast<MemRefType>().getShape().size() == rank &&
           "buffer and memref should have the same rank");
    ArrayAttributeIndexCapture writeSizeCapture(op.tileSizeAttr());
    assert((writeSizeCapture.size() == 0 || writeSizeCapture.size() == rank) &&
           "optional writeTileSize rank differs from memref");

    ScopedContext scope(rewriter, op.getLoc());
    IndexExprScope indexScope(rewriter, op.getLoc());
    SmallVector<IndexExpr, 4> starts, bufferWriteUBs;
    MemRefBoundIndexCapture buffBounds(buffMemref);
    MemRefBoundIndexCapture sourceBounds(sourceMemref);
    getIndexExprList<DimIndexExpr>(startVals, starts);
    SmallVector<Value, 4> loopIndices;
    LiteralIndexExpr zero(0);

    for (long i = 0; i < rank; ++i) {
      // Compute how many values to read.
      IndexExpr sourceBound = sourceBounds.getSymbol(i); // Source memref size.
      IndexExpr blockSize = buffBounds.getSymbol(i);     // Buffer memref size.
      if (writeSizeCapture.size()) {
        int64_t memSize = blockSize.getLiteral();
        blockSize = writeSizeCapture.getLiteral(i); // Size from param.
        assert(blockSize.getLiteral() <= memSize &&
               "writeTileSize cannot be larger than the buffer size");
      }
      IndexExpr startGI = starts[i]; // Global index in source memref of tile.
      IndexExpr bufferWrite = trip(sourceBound, blockSize, startGI);
      bufferWrite.debugPrint("buffer wrote");
      bufferWriteUBs.emplace_back(bufferWrite);
    }
    genCopyLoops(buffMemref, sourceMemref, zero, starts, bufferWriteUBs,
        loopIndices, 0, rank);
    rewriter.eraseOp(op);
    return success();
  }

  void genCopyLoops(Value buffMemref, Value sourceMemref, IndexExpr zero,
      SmallVectorImpl<IndexExpr> &starts, SmallVectorImpl<IndexExpr> &writeUBs,
      SmallVectorImpl<Value> &loopIndices, int64_t i, int64_t rank) const {
    if (i == rank) {
      // create new scope and import index expressions
      IndexExprScope currScope;
      SmallVector<IndexExpr, 4> currLoopIndices, currStarts;
      getIndexExprList<DimIndexExpr>(loopIndices, currLoopIndices);
      getIndexExprList<SymbolIndexExpr>(starts, currStarts);
      SmallVector<IndexExpr, 4> currStoreIndices;
      for (long i = 0; i < rank; ++i) {
        currStoreIndices.emplace_back(currLoopIndices[i] + currStarts[i]);
      }
      Value sourceVal = krnl_load(buffMemref, currLoopIndices);
      krnl_store(sourceVal, sourceMemref, currStoreIndices);
    } else {
      using namespace edsc::op;
      if (writeUBs[i].isLiteralAndIdenticalTo(0)) {
        // Nothing to write.
      } else {
        // Loop to copy the data.
        affineLoopBuilder(zero, writeUBs[i], 1, [&](Value index) {
          loopIndices.emplace_back(index);
          genCopyLoops(buffMemref, sourceMemref, zero, starts, writeUBs,
              loopIndices, i + 1, rank);
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
  target.addLegalOp<KrnlVectorTypeCastOp>();
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
