//===-- MLIRDialectBuilder.hpp.inc - Helper functions for MLIR dialects
//----===//
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
Value GenericAffineBuilder<LOAD_OP, STORE_OP>::load(
    Value memref, ValueRange indices) const {
  return b.create<LOAD_OP>(loc, memref, indices);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::store(
    Value val, Value memref, ValueRange indices) const {
  b.create<STORE_OP>(loc, val, memref, indices);
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forIE(IndexExpr lb,
    IndexExpr ub, int64_t step,
    function_ref<void(GenericAffineBuilder &, Value)> builderFn) const {
  // Transform IndexExpressions into value maps and list of operands.
  AffineMap lbMap, ubMap;
  SmallVector<Value, 8> lbOperands, ubOperands;
  lb.getAffineMapAndOperands(lbMap, lbOperands);
  ub.getAffineMapAndOperands(ubMap, ubOperands);
  // Create affine for.
  b.create<AffineForOp>(loc, lbOperands, lbMap, ubOperands, ubMap, step,
      ValueRange{},
      [&](OpBuilder &b, Location loc, Value index, ValueRange args) {
        GenericAffineBuilder createAffine(b, loc);
        builderFn(createAffine, index);
        createAffine.yield();
      });
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::forIE(
    SmallVectorImpl<IndexExpr> &lbs, SmallVectorImpl<IndexExpr> &ubs,
    SmallVectorImpl<int64_t> &steps,
    function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn) const {
  assert(lbs.size() == ubs.size() && "expected identical sizes");
  assert(lbs.size() == steps.size() && "expected identical sizes");
  SmallVector<Value> loopIndices;
  recursionForIE(lbs, ubs, steps, loopIndices, builderFn);
}

// This if then else construct has no arguments to the blocks.
template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::ifThenElse(
    IndexExprScope &scope, SmallVectorImpl<IndexExpr> &conditions,
    function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
    function_ref<void(GenericAffineBuilder &createAffine)> elseFn) const {
  int64_t rank = conditions.size();
  SmallVector<AffineExpr, 4> affineCond;
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
  SmallVector<bool, 4> isEq(rank, false);
  auto inset = IntegerSet::get(
      scope.getNumDims(), scope.getNumSymbols(), affineCond, isEq);
  SmallVector<Value, 8> dimAndSymbolList;
  scope.getDimAndSymbolList(dimAndSymbolList);
  auto ifOp = b.create<AffineIfOp>(loc, inset, dimAndSymbolList, true);
  Block *thenBlock = ifOp.getThenBlock();
  Block *elseBlock = ifOp.getElseBlock();
  if (!allFalse) {
    appendToBlock(thenBlock, [&](ValueRange args) {
      GenericAffineBuilder createAffine(b, loc);
      thenFn(createAffine);
    });
  }
  if (!allTrue) {
    appendToBlock(elseBlock, [&](ValueRange args) {
      GenericAffineBuilder createAffine(b, loc);
      elseFn(createAffine);
    });
  }
}

template <class LOAD_OP, class STORE_OP>
inline void GenericAffineBuilder<LOAD_OP, STORE_OP>::yield() const {
  b.create<AffineYieldOp>(loc);
}

// Support for multiple forIE loops.
template <class LOAD_OP, class STORE_OP>
void GenericAffineBuilder<LOAD_OP, STORE_OP>::recursionForIE(
    SmallVectorImpl<IndexExpr> &lbs, SmallVectorImpl<IndexExpr> &ubs,
    SmallVectorImpl<int64_t> &steps, SmallVectorImpl<Value> &loopIndices,
    function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn) const {
  int d = loopIndices.size();
  if (d < (int)lbs.size()) {
    // Issue a loop and recurse again.
    forIE(lbs[d], ubs[d], steps[d],
        [&](GenericAffineBuilder &createAffine, Value i) {
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
    Block *block, function_ref<void(ValueRange)> builderFn) const {
  OpBuilder::InsertionGuard guard(b);
  if (block->empty() ||
      !block->back().mightHaveTrait<OpTrait::IsTerminator>()) {
    b.setInsertionPointToEnd(block);
  } else
    b.setInsertionPoint(&block->back());
  builderFn(block->getArguments());
}
