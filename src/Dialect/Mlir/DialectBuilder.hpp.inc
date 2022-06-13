//===---- DialectBuilder.hpp.inc - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

// Implementation of GenericAffineBuilder
template <class LOAD_OP, class STORE_OP>
mlir::Value GenericAffineBuilder<LOAD_OP, STORE_OP>::load(
    mlir::Value memref, mlir::ValueRange indices) const {
  return b.create<LOAD_OP>(loc, memref, indices);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::store(
    mlir::Value val, mlir::Value memref, mlir::ValueRange indices) const {
  b.create<STORE_OP>(loc, val, memref, indices);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forIE(IndexExpr lb,
    IndexExpr ub, int64_t step,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::Value)> builderFn)
    const {
  // Transform IndexExpressions into value maps and list of operands.
  mlir::AffineMap lbMap, ubMap;
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;
  lb.getAffineMapAndOperands(lbMap, lbOperands);
  ub.getAffineMapAndOperands(ubMap, ubOperands);
  // Create affine for.
  b.create<mlir::AffineForOp>(loc, lbOperands, lbMap, ubOperands, ubMap, step,
      mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value index,
          mlir::ValueRange args) {
        GenericAffineBuilder createAffine(b, loc);
        builderFn(createAffine, index);
        createAffine.yield();
      });
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forIE(
    llvm::SmallVectorImpl<IndexExpr> &lbs,
    llvm::SmallVectorImpl<IndexExpr> &ubs,
    llvm::SmallVectorImpl<int64_t> &steps,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
        builderFn) const {
  assert(lbs.size() == ubs.size() && "expected identical sizes");
  assert(lbs.size() == steps.size() && "expected identical sizes");
  llvm::SmallVector<mlir::Value> loopIndices;
  recursionForIE(lbs, ubs, steps, loopIndices, builderFn);
}

// This if then else construct has no arguments to the blocks.
template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::ifThenElse(
    IndexExprScope &scope, llvm::SmallVectorImpl<IndexExpr> &conditions,
    mlir::function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
    mlir::function_ref<void(GenericAffineBuilder &createAffine)> elseFn) const {
  int64_t rank = conditions.size();
  llvm::SmallVector<mlir::AffineExpr, 4> affineCond;
  bool allTrue = true;
  bool allFalse = true;
  for (IndexExpr c : conditions) {
    assert(c.isAffine() && "conditions expected to be affine");
    affineCond.emplace_back(c.getAffineExpr());
    if (c.isLiteral()) {
      if (c.getLiteral() < 0) // Inequality is expr >= 0, test if false.
        allTrue = false;
      if (c.getLiteral() >= 0) // Inequality is expr >= 0, test if true.
        allFalse = false;
    } else {
      allTrue = allFalse = false;
    }
  }
  llvm::SmallVector<bool, 4> isEq(rank, false);
  auto inset = mlir::IntegerSet::get(
      scope.getNumDims(), scope.getNumSymbols(), affineCond, isEq);
  llvm::SmallVector<mlir::Value, 8> dimAndSymbolList;
  scope.getDimAndSymbolList(dimAndSymbolList);
  auto ifOp = b.create<mlir::AffineIfOp>(loc, inset, dimAndSymbolList, true);
  mlir::Block *thenBlock = ifOp.getThenBlock();
  mlir::Block *elseBlock = ifOp.getElseBlock();
  if (!allFalse) {
    appendToBlock(thenBlock, [&](mlir::ValueRange args) {
      GenericAffineBuilder createAffine(b, loc);
      thenFn(createAffine);
    });
  }
  if (!allTrue) {
    appendToBlock(elseBlock, [&](mlir::ValueRange args) {
      GenericAffineBuilder createAffine(b, loc);
      elseFn(createAffine);
    });
  }
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::yield() const {
  b.create<mlir::AffineYieldOp>(loc);
}

// Support for multiple forIE loops.
template <class LOAD_OP, class STORE_OP>
void GenericAffineBuilder<LOAD_OP, STORE_OP>::recursionForIE(
    llvm::SmallVectorImpl<IndexExpr> &lbs,
    llvm::SmallVectorImpl<IndexExpr> &ubs,
    llvm::SmallVectorImpl<int64_t> &steps,
    llvm::SmallVectorImpl<mlir::Value> &loopIndices,
    mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
        builderFn) const {
  int d = loopIndices.size();
  if (d < (int)lbs.size()) {
    // Issue a loop and recurse again.
    forIE(lbs[d], ubs[d], steps[d],
        [&](GenericAffineBuilder &createAffine, mlir::Value i) {
          loopIndices.emplace_back(i);
          recursionForIE(lbs, ubs, steps, loopIndices, builderFn);
        });
  } else {
    // Call lambda function
    GenericAffineBuilder createAffine(b, loc);
    builderFn(createAffine, loopIndices);
  }
}

// Support for adding blocks.
template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::appendToBlock(
    mlir::Block *block,
    mlir::function_ref<void(mlir::ValueRange)> builderFn) const {
  mlir::OpBuilder::InsertionGuard guard(b);
  if (block->empty() ||
      !block->back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
    b.setInsertionPointToEnd(block);
  } else
    b.setInsertionPoint(&block->back());
  builderFn(block->getArguments());
}
