/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------IndexExpr.cpp - Index expression---------------------=== //
//
// copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculation using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExprDetail.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "index-expr"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexExprScope constructors.
//===----------------------------------------------------------------------===//

// Initial scope.
IndexExprScope::IndexExprScope(OpBuilder *rewriter, Location loc)
    : dims(), symbols(), rewriter(rewriter), parentScope(getCurrentScopePtr()),
      loc(loc), container() {
#if DETAILED_DEBUG_OF_SCOPE
  LLVM_DEBUG(
      llvm::dbgs() << "IES: build scope: " << ((long long)this) << "\n";);
#endif
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(const DialectBuilder &db)
    : IndexExprScope(&db.getBuilder(), db.getLoc()) {}

// Nested scopes.
IndexExprScope::IndexExprScope(
    OpBuilder *innerRewriter, IndexExprScope *enclosingScope)
    : dims(), symbols(), rewriter(innerRewriter),
      parentScope(enclosingScope ? enclosingScope : getCurrentScopePtr()),
      loc(parentScope->loc), container() {
#if DETAILED_DEBUG_OF_SCOPE
  LLVM_DEBUG(
      llvm::dbgs() << "IES: build scope: " << ((long long)this) << "\n";);
#endif
  // Check the provided enclosing scope is the current one.
  assert(parentScope == getCurrentScopePtr() &&
         "provided parent scope was not the enclosing active scope");
  // Set location.
  // assert(parentScope && "Use this constructor only for nested scopes");
  // loc = parentScope->loc;
  // Install new inner scope as current one.
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(
    const DialectBuilder &innerDb, IndexExprScope *enclosingScope)
    : IndexExprScope(&innerDb.getBuilder(), enclosingScope) {}

IndexExprScope::~IndexExprScope() {
  // Free the memory of each IndexExprImpl in scope's container.
  for (IndexExprImpl *obj : container)
    delete obj;
  container.clear();
  // no need to clear the cached copies as they are also in the container.
  getCurrentScopePtr() = parentScope;
#if DETAILED_DEBUG_OF_SCOPE
  LLVM_DEBUG(
      llvm::dbgs() << "IES: delete scope: " << ((long long)this) << "\n";);
#endif
}

/*static*/ IndexExprScope &IndexExprScope::getCurrentScope() {
  IndexExprScope *currScope = getCurrentScopePtr();
  assert(currScope != nullptr && "expected nonnull scope");
  return *currScope;
}

//===----------------------------------------------------------------------===//
// IndexExprScope builder for IndexExpr.
//===----------------------------------------------------------------------===//

void IndexExprScope::addIndexExprImpl(IndexExprImpl *obj) {
  container.emplace_back(obj);
}

//===----------------------------------------------------------------------===//
// IndexExprScope support for dim and symbol lists in affine exprs.
//===----------------------------------------------------------------------===//

int IndexExprScope::indexInList(
    SmallVectorImpl<Value> const &list, Value const &value) const {
  int num = list.size();
  for (int i = 0; i < num; ++i) {
    if (list[i] == value)
      return i;
  }
  return -1; // Minus one indicates not found.
}

int IndexExprScope::addDim(Value const value) {
  assert(indexInList(symbols, value) == -1 &&
         "Cannot have a dim that is already a symbol");
  int i = indexInList(dims, value);
  if (i != -1)
    return i; // We already have this dim, reuse it.
  // Else, new dim, add at the end.
  dims.emplace_back(value);
  return dims.size() - 1;
}

int IndexExprScope::addSymbol(Value const value) {
  assert(indexInList(dims, value) == -1 &&
         "Cannot have a symbol that is already a dim");
  int i = indexInList(symbols, value);
  if (i != -1)
    return i; // We already have this symbol, reuse it.
  // Else, new symbol, add at the end.
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

//===----------------------------------------------------------------------===//
// IndexExprScope getters.
//===----------------------------------------------------------------------===//

bool IndexExprScope::isCurrentScope() const {
  return getCurrentScopePtr() == this;
}

bool IndexExprScope::isEnclosingScope() const {
  for (IndexExprScope *s = getCurrentScopePtr()->parentScope; s;
       s = s->parentScope) {
    if (s == this)
      return true;
  }
  return false;
}

void IndexExprScope::getDimAndSymbolList(SmallVectorImpl<Value> &list) const {
  list.clear();
  for (auto dim : dims)
    list.emplace_back(dim);
  for (auto sym : symbols)
    list.emplace_back(sym);
}

OpBuilder &IndexExprScope::getRewriter() const {
  assert(rewriter && "Should have a valid pointer");
  return *rewriter;
}

OpBuilder *IndexExprScope::getRewriterPtr() const { return rewriter; }

//===----------------------------------------------------------------------===//
// IndexExprScope Debug.
//===----------------------------------------------------------------------===//

void IndexExprScope::debugPrint(const std::string &msg) const {
  LLVM_DEBUG(llvm::dbgs() << "Scope " << msg.c_str() << " 0x" << (long long)this
                          << ": with parent scope 0x" << (long long)parentScope
                          << " and " << dims.size() << " dims and "
                          << symbols.size() << " symbols\n";);
}

//===----------------------------------------------------------------------===//
// IndexExpr copy and setters.
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::deepCopy() const {
  // Create new implementation and set scope to current scope (don't copy it).
  IndexExprImpl *newImplObj = new IndexExprImpl();
  assert(newImplObj && "failed to allocate IndexExpr implementation");
  // Copy all of hte other fields (preserving current scope).
  newImplObj->copy(getObjPtr());
  return IndexExpr(newImplObj);
}

//===----------------------------------------------------------------------===//
// IndexExpr queries.
//===----------------------------------------------------------------------===//

bool IndexExpr::isDefined() const {
  assert(!getObj().isDefined() || hasScope());
  return getObj().isDefined();
}

// Undefined: its ok to have no impl object associated with it.
bool IndexExpr::isUndefined() const {
  return !indexExprObj || !getObj().isDefined();
}

bool IndexExpr::isLiteral() const { return getObj().isLiteral(); }

bool IndexExpr::isFloat() const { return getObj().isFloatType(); }

bool IndexExpr::areFloat(IndexExpr b) const {
  if (isFloat() && b.isFloat())
    return true;
  if (!isFloat() && !b.isFloat())
    return false;
  llvm_unreachable("expected both float or both integer, got some of both");
}

bool IndexExpr::isQuestionmark() const { return getObj().isQuestionmark(); }

bool IndexExpr::isAffine() const { return getObj().isAffine(); }

bool IndexExpr::isSymbol() const { return getObj().isSymbol(); }

bool IndexExpr::isDim() const { return getObj().isDim(); }

bool IndexExpr::isPredType() const { return getObj().isPredType(); }

bool IndexExpr::isIndexType() const { return getObj().isIndexType(); }

bool IndexExpr::hasAffineExpr() const { return getObj().hasAffineExpr(); }

bool IndexExpr::hasValue() const { return getObj().hasValue(); }

//===----------------------------------------------------------------------===//
// IndexExpr list queries.
//===----------------------------------------------------------------------===//

bool IndexExpr::isLiteralAndIdenticalTo(int b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteralAndIdenticalTo((int64_t)b);
}

bool IndexExpr::isLiteralAndIdenticalTo(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() == b);
}

bool IndexExpr::isLiteralAndIdenticalTo(double b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getFloatLiteral() == b);
}

bool IndexExpr::isLiteralAndIdenticalTo(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!b.isLiteral())
    return false;
  if (areFloat(b))
    return isLiteralAndIdenticalTo(b.getFloatLiteral());
  return isLiteralAndIdenticalTo(b.getLiteral());
}

bool IndexExpr::isLiteralAndDifferentThan(int b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteralAndDifferentThan((int64_t)b);
}

bool IndexExpr::isLiteralAndDifferentThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() != b);
}

bool IndexExpr::isLiteralAndDifferentThan(double b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getFloatLiteral() != b);
}

bool IndexExpr::isLiteralAndDifferentThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!b.isLiteral())
    return false;
  if (areFloat(b))
    return isLiteralAndDifferentThan(b.getFloatLiteral());
  return isLiteralAndDifferentThan(b.getLiteral());
}

bool IndexExpr::isLiteralAndGreaterThan(int b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteralAndGreaterThan((int64_t)b);
}

bool IndexExpr::isLiteralAndGreaterThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() > b);
}

bool IndexExpr::isLiteralAndGreaterThan(double b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getFloatLiteral() > b);
}

bool IndexExpr::isLiteralAndGreaterThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!b.isLiteral())
    return false;
  if (areFloat(b))
    return isLiteralAndGreaterThan(b.getFloatLiteral());
  return isLiteralAndGreaterThan(b.getLiteral());
}

bool IndexExpr::isLiteralAndSmallerThan(int b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteralAndSmallerThan((int64_t)b);
}

bool IndexExpr::isLiteralAndSmallerThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() < b);
}

bool IndexExpr::isLiteralAndSmallerThan(double b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getFloatLiteral() < b);
}

bool IndexExpr::isLiteralAndSmallerThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!b.isLiteral())
    return false;
  if (areFloat(b))
    return isLiteralAndSmallerThan(b.getFloatLiteral());
  return isLiteralAndSmallerThan(b.getLiteral());
}

// All element in list are literals.
/*static*/ bool IndexExpr::isLiteral(ArrayRef<IndexExpr> list) {
  for (IndexExpr i : list)
    if (!i.isLiteral())
      return false;
  return true;
}

// All element in list are literals and non-negative (i.e. >= 0).
/*static*/ bool IndexExpr::isNonNegativeLiteral(ArrayRef<IndexExpr> list) {
  for (IndexExpr i : list)
    if (!i.isLiteral() || i.getLiteral() < 0)
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// IndexExpr private queries.
//===----------------------------------------------------------------------===//

bool IndexExpr::hasScope() const { return getObj().hasScope(); }

bool IndexExpr::isInCurrentScope() const { return getScope().isCurrentScope(); }

bool IndexExpr::canBeUsedInScope() const {
  if (isInCurrentScope())
    return true;
  if (isLiteral()) {
    return getScope().isEnclosingScope();
  }
  switch (getKind()) {
  case IndexExprKind::NonAffine:
  case IndexExprKind::Predicate:
    // Its ok to use a non-affine index expressions from enclosing scopes.
    assert(hasValue() && "must have value to be used from enclosing scopes");
    return getScope().isEnclosingScope();
    break;
  case IndexExprKind::Questionmark:
    return true;
    break;
  case IndexExprKind::Affine:
  case IndexExprKind::Dim:
  case IndexExprKind::Symbol:
    // Because affine/dim/symbols are specific to a current scope, they have to
    // be converted to the current scope before being used. They cannot be used
    // out of current scope.
    return false;
  }
  llvm_unreachable("unknown kind");
}

//===----------------------------------------------------------------------===//
// IndexExpr public getter.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const {
  assert(!isFloat() && "attempt to get int value of a float index expr");
  return getObj().getLiteral();
}

double IndexExpr::getFloatLiteral() const {
  assert(isFloat() && "attempt to get float value of a int index expr");
  return getObj().getFloatLiteral();
}

int64_t IndexExpr::getQuestionmark() const {
  return getObj().getQuestionmark();
}

AffineExpr IndexExpr::getAffineExpr() const {
  assert(!isFloat() && "attempt to get affine expr of a float index expr");
  return getObj().getAffineExpr();
}

Value IndexExpr::getValue() const { return getObj().getValue(); }

void IndexExpr::getAffineMapAndOperands(
    AffineMap &map, SmallVectorImpl<Value> &operands) const {
  assert(!isFloat() && "attempt to get affine map of a float index expr");
  getObj().getAffineMapAndOperands(map, operands);
}

int64_t IndexExpr::getShape(bool uniqueQuestionMark) const {
  if (isLiteral()) {
    int64_t val = getLiteral();
    assert(val >= 0 && "expected positive shape values only");
    return val;
  }
  if (uniqueQuestionMark)
    return getQuestionmark();
  return ShapedType::kDynamic;
}

//===----------------------------------------------------------------------===//
// IndexExpr private getter.
//===----------------------------------------------------------------------===//

IndexExprScope *IndexExpr::getScopePtr() const {
  return getObj().getScopePtr();
}

IndexExprImpl &IndexExpr::getObj() const { return *getObjPtr(); }

IndexExprImpl *IndexExpr::getObjPtr() const {
  assert(indexExprObj);
  return indexExprObj;
}

IndexExprKind IndexExpr::getKind() const { return getObj().getKind(); }

//===----------------------------------------------------------------------===//
// IndexExpr Debug.
//===----------------------------------------------------------------------===//

void IndexExpr::debugPrint(const std::string &msg) const {
  LLVM_DEBUG({
    if (!indexExprObj)
      llvm::dbgs() << msg.c_str() << " undefined\n";
    else
      indexExprObj->debugPrint(msg);
  });
}

void IndexExpr::debugPrint(
    const std::string &msg, const ArrayRef<IndexExpr> list) {
  LLVM_DEBUG({
    int s = list.size();
    llvm::dbgs() << msg.c_str() << " (" << s << " elements)\n";
    for (int i = 0; i < s; ++i) {
      std::string element = "  " + std::to_string(i) + ": ";
      list[i].debugPrint(element);
    }
  });
}

//===----------------------------------------------------------------------===//
// Helpers for IndexExpressions
//===----------------------------------------------------------------------===//

/*static*/ void IndexExpr::getLiteral(
    ArrayRef<IndexExpr> indexExprArray, SmallVectorImpl<int64_t> &intList) {
  intList.clear();
  llvm::transform(indexExprArray, std::back_inserter(intList),
      [](IndexExpr expr) { return expr.getLiteral(); });
}

/*static*/ void IndexExpr::getShape(ArrayRef<IndexExpr> indexExprArray,
    SmallVectorImpl<int64_t> &intDimList, bool uniqueQuestionMark) {
  intDimList.clear();
  for (IndexExpr expr : indexExprArray)
    intDimList.emplace_back(expr.getShape(uniqueQuestionMark));
}

/*static*/ void IndexExpr::getDynSymbols(
    ArrayRef<IndexExpr> indexExprArray,         // Input list.
    llvm::SmallVectorImpl<Value> &dynSymbols) { // Symbol for dyn ref.
  // For each dyn shape, enqueue its value in dynamic symbol list.
  dynSymbols.clear();
  for (IndexExpr expr : indexExprArray) {
    if (!expr.isLiteral())
      dynSymbols.emplace_back(expr.getValue());
  }
}

/*static*/ void IndexExpr::getOpOrFoldResults(
    ArrayRef<IndexExpr> indexExprArray,
    SmallVectorImpl<OpFoldResult> &resList) {
  resList.clear();
  for (IndexExpr expr : indexExprArray) {
    if (expr.isLiteral()) {
      assert(!expr.isFloat() && "missing support for float");
      auto val = expr.getRewriter().getIndexAttr(expr.getLiteral());
      resList.emplace_back(val);
    } else
      resList.emplace_back(expr.getValue());
  }
}

/*static*/ void IndexExpr::getValues(
    ArrayRef<IndexExpr> indexExprArray, SmallVectorImpl<Value> &valueList) {
  valueList.clear();
  for (IndexExpr expr : indexExprArray)
    valueList.emplace_back(expr.getValue());
}

/* static*/ void IndexExpr::getAffineMapAndOperands(
    ArrayRef<IndexExpr> indexExprArray, AffineMap &map,
    SmallVectorImpl<Value> &operands) {
  assert(indexExprArray.size() > 0 && "expected at least one index expr");
  SmallVector<AffineExpr, 8> affineExprList;
  for (IndexExpr expr : indexExprArray) {
    AffineMap tmpMap;
    SmallVector<Value, 8> tmpOperands;
    expr.getAffineMapAndOperands(tmpMap, tmpOperands);
    operands = tmpOperands;
    // Enqueue the affine expressions defined by this temp map.
    for (AffineExpr affineExpr : tmpMap.getResults()) {
      affineExprList.emplace_back(affineExpr);
    }
  }

  // Now can generate a common map with all the results
  unsigned dimCount = indexExprArray[0].getScope().getNumDims();
  unsigned symCount = indexExprArray[0].getScope().getNumSymbols();
  map = AffineMap::get(dimCount, symCount, affineExprList,
      indexExprArray[0].getRewriter().getContext());
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Local support function, use double to also represent the int value for the
// neutral values.
static bool isIdentical(const IndexExpr litExpr, double dval) {
  if (litExpr.isFloat())
    return litExpr.isLiteralAndIdenticalTo(dval);
  int64_t ival = dval;
  return litExpr.isLiteralAndIdenticalTo(ival);
}

// Used for add/sub/mult/ceilDiv/floorDiv.
// Add/sub: B does not need to be a literal for the result to be affine.
// All the other ones (mul, div*, mod) require the B to be a literal.
IndexExpr IndexExpr::binaryOp(IndexExpr const b, bool propagateIntoMinMax,
    bool affineWithLitB, bool hasNeutralA, bool hasNeutralB, double neutralVal,
    F2 litFct, F2 affineExprFct, F2 valueFct) const {
  assert(litFct && "expect lit function");
  assert(valueFct && "expect value function");
  assert(canBeUsedInScope() && "a cannot be used in current scope");
  assert(b.canBeUsedInScope() && "b cannot be used in current scope");
  // Literal integer if a and b are literals. Affine if canBeAffine is true,
  // both a and b are affine, and possibly a and/or b are also constant.
  bool resIsLit = isLiteral() && b.isLiteral();
  bool canBeAffine = (affineExprFct != nullptr);
  bool resIsAffine = resIsLit || (canBeAffine && isAffine() && b.isAffine() &&
                                     (!affineWithLitB || b.isLiteral()));
  if (resIsAffine)
    // Test if we have a neutral value.
    if (hasNeutralA && isIdentical(*this, neutralVal))
      return b.deepCopy(); // Copy of the other value (use same questionmark).
  if (hasNeutralB && isIdentical(b, neutralVal)) {
    return deepCopy(); // Copy of the other value (use same questionmark).
  }
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this, b);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values &
    // Affine expr undefined. Result is float if both inputs are float.
    return QuestionmarkIndexExpr(areFloat(b));
  if (resIsAffine)
    // Use affine values.
    return affineExprFct(*this, b);
  // See if we have a min/max on one side that we can propagate into.
  if (canBeAffine && propagateIntoMinMax) {
    Value valA = this->getValue();
    bool hasMinMaxA = valA.getDefiningOp<affine::AffineMinOp>() ||
                      valA.getDefiningOp<affine::AffineMaxOp>();
    Value valB = b.getValue();
    bool hasMinMaxB = valB.getDefiningOp<affine::AffineMinOp>() ||
                      valB.getDefiningOp<affine::AffineMaxOp>();
    // Can handle only cases where either a or b are min/max and the other one
    // is affine.
    if ((hasMinMaxA && !hasMinMaxB && b.isAffine()) ||
        (!hasMinMaxA && hasMinMaxB && this->isAffine())) {
      // Of the two inputs, find out the one with the min/max.
      IndexExpr minMaxIE = hasMinMaxA ? *this : b;
      // Retrieve the map and list of dim/symbols in the current scope
      bool isMin;
      llvm::SmallVector<Value, 8> vals;
      AffineMap map;
      assert(minMaxIE.retrieveAffineMinMax(isMin, vals, map) && "expected one");
      // Perform the affineExprFct for each min/max terms.
      llvm::SmallVector<IndexExpr, 4> updatedMinMaxExprs;
      for (AffineExpr affineExpr : map.getResults()) {
        IndexExpr oldAffineExpr = AffineIndexExpr(affineExpr);
        IndexExpr newAffineExpr;
        if (hasMinMaxA)
          newAffineExpr = affineExprFct(oldAffineExpr, b);
        else
          newAffineExpr = affineExprFct(*this, oldAffineExpr);
        updatedMinMaxExprs.emplace_back(newAffineExpr);
      }
      // Create new operation.
      if (isMin) {
        return IndexExpr::min(updatedMinMaxExprs);
      }
      return IndexExpr::max(updatedMinMaxExprs);
    }
  }
  // Use values.
  return valueFct(*this, b);
}

// Used for ceil/floor
IndexExpr IndexExpr::unaryOp(
    bool resIsFloat, F1 litFct, F1 affineExprFct, F1 valueFct) const {
  assert(litFct && "expect lit function");
  assert(valueFct && "expect value function");
  assert(canBeUsedInScope() && "a cannot be used in current scope");
  // Literal integer if is literal. Affine if canBeAffine is true,
  // and a is affine.
  bool resIsLit = isLiteral();
  bool canBeAffine = (affineExprFct != nullptr);
  bool resIsAffine = resIsLit || (canBeAffine && isAffine());

  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values &
    // Affine expr undefined.
    return QuestionmarkIndexExpr(resIsFloat);
  if (resIsAffine) {
    // Use affine values.
    return affineExprFct(*this);
  }
  // Use values.
  return valueFct(*this);
}

// Integer version/
IndexExpr IndexExpr::compareOp(
    arith::CmpIPredicate comparePred, IndexExpr const b) const {
  F2 litFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t aaa = aa.getLiteral();
    int64_t bbb = bb.getLiteral();
    switch (comparePred) {
    case arith::CmpIPredicate::eq:
      if (aaa == bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpIPredicate::ne:
      if (aaa != bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpIPredicate::slt:
      if (aaa < bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpIPredicate::sle:
      if (aaa <= bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpIPredicate::sgt:
      if (aaa > bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpIPredicate::sge:
      if (aaa >= bbb)
        return PredicateIndexExpr(true);
      break;
    default:
      llvm_unreachable("unknown or illegal (unsigned) compare operator");
    }
    return PredicateIndexExpr(false);
  };
  F2 valueFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    Value compare = aa.getRewriter().create<arith::CmpIOp>(
        aa.getLoc(), comparePred, aa.getValue(), bb.getValue());
    return PredicateIndexExpr(compare);
  };

  // Cannot have affine results, disable and pass null lambda function.
  // Ignore possible neutral values.
  assert(!areFloat(b) && "integer compare");
  return binaryOp(
      b, false, false, false, false, 0.0, litFct, nullptr, valueFct);
}

// Floating point version.
IndexExpr IndexExpr::compareOp(
    arith::CmpFPredicate comparePred, IndexExpr const b) const {
  // Float version.
  F2 litFloatFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    double aaa = aa.getFloatLiteral();
    double bbb = bb.getFloatLiteral();
    switch (comparePred) {
    case arith::CmpFPredicate::OEQ:
      if (aaa == bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpFPredicate::ONE:
      if (aaa != bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpFPredicate::OLT:
      if (aaa < bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpFPredicate::OLE:
      if (aaa <= bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpFPredicate::OGT:
      if (aaa > bbb)
        return PredicateIndexExpr(true);
      break;
    case arith::CmpFPredicate::OGE:
      if (aaa >= bbb)
        return PredicateIndexExpr(true);
      break;
    default:
      llvm_unreachable("unknown or illegal (unsigned) compare operator");
    }
    return PredicateIndexExpr(false);
  };
  F2 valueFloatFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    Value compare = aa.getRewriter().create<arith::CmpFOp>(
        aa.getLoc(), comparePred, aa.getValue(), bb.getValue());
    return PredicateIndexExpr(compare);
  };

  // Cannot have affine results, disable and pass null lambda function.
  // Ignore possible neutral values.
  assert(areFloat(b) && "float compare");
  return binaryOp(
      b, false, false, false, false, 0.0, litFloatFct, nullptr, valueFloatFct);
}

// Conjunction of two conditions: And
IndexExpr IndexExpr::operator&(IndexExpr const b) const {
  if (isLiteral()) {
    if (getLiteral() == 0)
      // false & b -> false
      return PredicateIndexExpr(false);
    // true & b -> b
    return b.deepCopy();
  }
  if (b.isLiteral()) {
    if (b.getLiteral() == 0)
      // a & false -> false
      return PredicateIndexExpr(false);
    // a & true -> a
    return deepCopy();
  }
  if (isQuestionmark() || b.isQuestionmark())
    return QuestionmarkIndexExpr(/*isFloat*/ false);
  // Not literals or questionmark, we must have predicates.
  assert(isPredType() && "expected predicate index expression");
  assert(b.isPredType() && "expected predicate index expression");
  MathBuilder createMath(getRewriter(), getLoc());
  return PredicateIndexExpr(createMath.andi(getValue(), b.getValue()));
}

// Conjunction of two conditions: Or
IndexExpr IndexExpr::operator|(IndexExpr const b) const {
  if (isLiteral()) {
    if (getLiteral() != 0)
      // true | b -> true
      return PredicateIndexExpr(true);
    // false | b -> b
    return b.deepCopy();
  }
  if (b.isLiteral()) {
    if (b.getLiteral() != 0)
      // a & true -> true
      return PredicateIndexExpr(true);
    // a & false -> a
    return deepCopy();
  }
  if (isQuestionmark() || b.isQuestionmark())
    return QuestionmarkIndexExpr(/*isFloat*/ false);
  // Not literals or questionmark, we must have predicates.
  assert(isPredType() && "expected predicate index expression");
  assert(b.isPredType() && "expected predicate index expression");
  MathBuilder createMath(getRewriter(), getLoc());
  return PredicateIndexExpr(createMath.ori(getValue(), b.getValue()));
}

IndexExpr IndexExpr::operator!() const {
  return (*this == PredicateIndexExpr(false));
}

// The affine reduction lambda function processes the whole list and must init
// the result. Literal and Values treat one operation at a time
/* static*/ IndexExpr IndexExpr::reductionOp(
    ArrayRef<IndexExpr> vals, F2Self litRed, Flist affineRed, F2Self valueRed) {
  // If no values, result is undefined.
  int size = vals.size();
  if (size == 0)
    return UndefinedIndexExpr();

  // Set the output to the first value.
  IndexExpr res = vals[0].deepCopy();
  // If list has one element, we are done. Literal/Affine... will be the same
  // as this single element.
  if (vals.size() == 1)
    return res;
  // Have multiple values, need to do some checks.
  bool resIsLit = (litRed != nullptr);
  bool resIsAffine = (affineRed != nullptr);
  for (int i = 0; i < size; ++i) {
    if (!vals[i].isLiteral())
      resIsLit = false;
    if (!vals[i].isAffine())
      resIsAffine = false;
    assert(vals[i].canBeUsedInScope() && "incompatible contexts");
  }
  if (resIsLit) {
    // Process int literals, if we only have literal values.
    // Result was set to first element, which by default is literal/affine.
    // This will be the correct result for the output.
    for (int i = 1; i < size; ++i) {
      litRed(res, vals[i]);
    }
    return res;
  }
  if (vals[0].isShapeInferencePass()) {
    // Just set as undefined
    return QuestionmarkIndexExpr(vals[0].isFloat());
  }
  if (resIsAffine) {
    // Affine handles the hole list
    return affineRed(res, vals);
  }
  // Process value, one item at a time.
  for (int i = 1; i < size; ++i) {
    valueRed(res, vals[i]);
  }
  return res;
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops.
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::operator+(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getLiteral() + bb.getLiteral());
  };
  F2 litFloatFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getFloatLiteral() + bb.getFloatLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() + bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.add(aa.getValue(), bb.getValue()));
  };
  // Neutral value: a + 0 = a, 0 + b = b.
  if (areFloat(b))
    return binaryOp(
        b, false, false, true, true, 0.0, litFloatFct, nullptr, valueFct);
  return binaryOp(
      b, true, false, true, true, 0.0, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator-(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getLiteral() - bb.getLiteral());
  };
  F2 litFloatFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getFloatLiteral() - bb.getFloatLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() - bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.sub(aa.getValue(), bb.getValue()));
  };
  // Neutral value: a - 0 = a.
  if (areFloat(b))
    return binaryOp(
        b, false, false, false, true, 0.0, litFloatFct, nullptr, valueFct);
  return binaryOp(
      b, true, false, false, true, 0.0, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator*(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getLiteral() * bb.getLiteral());
  };
  F2 litFloatFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LitIE(aa.getFloatLiteral() * bb.getFloatLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() * bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.mul(aa.getValue(), bb.getValue()));
  };
  // Neutral value: a * 1 = a, 1 * b = b.
  if (areFloat(b))
    return binaryOp(
        b, false, false, true, true, 1.0, litFloatFct, nullptr, valueFct);
  // For affine, requires one to be a literal, and in "b" (argument).
  if (isLiteral())
    return b.binaryOp(
        *this, false, true, true, true, 1.0, litFct, affineExprFct, valueFct);
  return binaryOp(
      b, false, true, true, true, 1.0, litFct, affineExprFct, valueFct);
}

// Int operator
IndexExpr IndexExpr::floorDiv(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval =
        std::floor((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return LitIE(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval > 1)
      return AffineIndexExpr(aa.getAffineExpr().floorDiv(bval));
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(
        createMath.floorDiv(aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(
        createMath.floorDiv(aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  // Neutral value: a / 1 = a.
  assert(!areFloat(b) && "floor div only supports int");
  return binaryOp(
      b, false, true, false, true, 1.0, litFct, affineExprFct, valueFct);
}

// Int operator
IndexExpr IndexExpr::ceilDiv(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = std::ceil((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return LitIE(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval > 1)
      return AffineIndexExpr(aa.getAffineExpr().ceilDiv(bval));
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.ceilDiv(aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.ceilDiv(aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  // Neutral value: a / 1 = a.
  assert(!areFloat(b) && "ceil div only supports int");
  return binaryOp(
      b, false, true, false, true, 1.0, litFct, affineExprFct, valueFct);
}

// Int operator
IndexExpr IndexExpr::operator%(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = llvm::mod(aa.getLiteral(), bb.getLiteral());
    return LitIE(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval >= 0)
      return AffineIndexExpr(aa.getAffineExpr() % bval);
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.rem(aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.rem(aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  // Neutral value: ignore here that x % x = 0.
  assert(!areFloat(b) && "mod only supports int");
  return binaryOp(
      b, false, true, false, false, 1.0, litFct, affineExprFct, valueFct);
}

// Float operator
IndexExpr IndexExpr::operator/(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    double rval = aa.getFloatLiteral() / bb.getFloatLiteral();
    return LitIE(rval);
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.div(aa.getValue(), bb.getValue()));
  };
  // Neutral value: x / 1 = x.
  assert(areFloat(b) && "float only; int: use ceilDiv or floorDiv");
  // Note: there are no affine functions for float, so affineWithLitB==true or
  // false is irrelevant.
  return binaryOp(b, false, false, false, true, 1.0, litFct, nullptr, valueFct);
}

// Float operator.
IndexExpr IndexExpr::ceil() const {
  F1 litFct = [](IndexExpr const aa) -> IndexExpr {
    double rval = std::ceil(aa.getFloatLiteral());
    return LitIE(rval);
  };
  F1 valueFct = [](IndexExpr const aa) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.ceil(aa.getValue()));
  };

  // Neutral value: none.
  assert(isFloat() && "float only; int: use ceilDiv or floorDiv");
  return unaryOp(/*resIsFloat*/ true, litFct, nullptr, valueFct);
}

// Float operator.
IndexExpr IndexExpr::floor() const {
  F1 litFct = [](IndexExpr const aa) -> IndexExpr {
    double rval = std::floor(aa.getFloatLiteral());
    return LitIE(rval);
  };
  F1 valueFct = [](IndexExpr const aa) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    Value floorVal = createMath.floor(aa.getValue());
    return NonAffineIndexExpr(floorVal);
  };

  // Neutral value: none.
  assert(isFloat() && "float only; int: use ceilDiv or floorDiv");
  return unaryOp(/*resIsFloat*/ true, litFct, nullptr, valueFct);
}

// Int operator.
IndexExpr IndexExpr::convertToFloat() const {
  F1 litFct = [](IndexExpr const aa) -> IndexExpr {
    double rval = (double)aa.getLiteral();
    return LitIE(rval);
  };
  F1 valueFct = [](IndexExpr const aa) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    Type f32Ty = aa.getRewriter().getF32Type();
    return NonAffineIndexExpr(createMath.cast(f32Ty, aa.getValue()));
  };

  // Neutral value: none.
  assert(!isFloat() && "convert to float expect an int as input");
  return unaryOp(/*resIsFloat*/ true, litFct, nullptr, valueFct);
}

// Float operator
IndexExpr IndexExpr::convertToIndex() const {
  F1 litFct = [](IndexExpr const aa) -> IndexExpr {
    int64_t rval = (int64_t)aa.getFloatLiteral();
    return LitIE(rval);
  };
  F1 valueFct = [](IndexExpr const aa) -> IndexExpr {
    MathBuilder createMath(aa.getRewriter(), aa.getLoc());
    return NonAffineIndexExpr(createMath.castToIndex(aa.getValue()));
  };

  // Neutral value: none.
  assert(isFloat() && "convert to int expect a float as input");
  return unaryOp(/*resIsFloat*/ false, litFct, nullptr, valueFct);
}

IndexExpr IndexExpr::clamp(IndexExpr const min, IndexExpr const max) const {
  // Functions below unconditionally override rr with the clipped value of
  // val.
  F3 litFct = [](IndexExpr const val, IndexExpr const min,
                  IndexExpr const max) -> IndexExpr {
    // assume signed compares
    if (val.isLiteralAndSmallerThan(min))
      return min;
    if (val.isLiteralAndGreaterThan(max))
      return max;
    return val;
  };
  // Int or float.
  F3 valueFct = [](IndexExpr const val, IndexExpr const min,
                    IndexExpr const max) {
    IndexExpr res1 = select(val < min, min, val);
    IndexExpr res2 = select(res1 > max, max, res1);
    return res2;
  };

  assert(canBeUsedInScope() && "incompatible scope");
  assert(min.canBeUsedInScope() && "min incompatible scope");
  assert(max.canBeUsedInScope() && "max incompatible scope");

  // Literal integer if a, b, and c are literals. Output is not affine (unless
  // all 3 are literals).
  bool resIsLit = isLiteral() && min.isLiteral() && max.isLiteral();
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this, min, max);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values &
    // Affine expr undefined.
    return QuestionmarkIndexExpr(isFloat());
  // Use values.
  return valueFct(*this, min, max);
}

/*static*/ IndexExpr IndexExpr::select(IndexExpr const compare,
    IndexExpr const trueVal, IndexExpr const falseVal) {
  assert(compare.canBeUsedInScope() && "compare incompatible scope");
  assert(trueVal.canBeUsedInScope() && "trueVal incompatible scope");
  assert(falseVal.canBeUsedInScope() && "falseVal incompatible scope");
  // When compare result is literal, just feed forward the right value.
  // Do not deep copy the question mark to keep it unchanged.
  if (compare.isLiteral()) {
    if (compare.getLiteral()) {
      if (trueVal.isQuestionmark())
        return trueVal;
      return trueVal.deepCopy();
    }
    if (falseVal.isQuestionmark())
      return falseVal;
    return falseVal.deepCopy();
  }
  // Dynamic value, just set as undefined during shape inference pass.
  if (compare.isShapeInferencePass())
    return QuestionmarkIndexExpr(trueVal.isFloat());
  // Generate code for the select.
  MathBuilder createMath(compare.getRewriter(), compare.getLoc());
  Value results = createMath.select(
      compare.getValue(), trueVal.getValue(), falseVal.getValue());
  return NonAffineIndexExpr(results);
}

/*static*/ IndexExpr IndexExpr::min(ArrayRef<IndexExpr> vals) {
  // Res is already an literal int, we are reducing into it.
  F2Self litFct = [](IndexExpr res, IndexExpr const aa) -> IndexExpr {
    if (aa.isLiteralAndSmallerThan(res))
      res.getObj().setLiteral(aa.getObj());
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            ArrayRef<IndexExpr> vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    // Important to get the affine expressions before getting the
    // dims/symbols.
    for (IndexExpr vv : vvals) {
      affineExprs.emplace_back(vv.getAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprScope &scope = vvals[0].getScope();
    int dimNum = scope.getNumDims();
    int symNum = scope.getNumSymbols();
    auto mapContext = scope.getRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    scope.getDimAndSymbolList(dimAndSymList);
    Value minVal = scope.getRewriter().create<affine::AffineMinOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsKind(minVal, IndexExprKind::NonAffine);
    return res;
  };
  // Res is already defined, we are reducing into it.
  // Integer version.
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
    // Could use arith::min op.
    IndexExpr compareIE = aa < res;
    IndexExpr selectIE = select(compareIE, aa, res);
    res.getObj().initAsKind(selectIE.getValue(), IndexExprKind::NonAffine);
    return res;
  };

  // Empty, treat as integer.
  if (vals.size() > 0 && vals[0].isFloat())
    return reductionOp(vals, litFct, nullptr, valueFct);
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::min(
    IndexExpr const first, IndexExpr const second) {
  SmallVector<IndexExpr, 2> list = {first, second};
  return min(list);
}

/*static*/ IndexExpr IndexExpr::min(
    IndexExpr const first, int64_t const second) {
  SmallVector<IndexExpr, 2> list = {first, LitIE(second)};
  return min(list);
}

/*static*/ IndexExpr IndexExpr::max(ArrayRef<IndexExpr> vals) {
  // Res is already an literal int, we are reducing into it.
  F2Self litFct = [](IndexExpr res, IndexExpr const aa) -> IndexExpr {
    if (aa.isLiteralAndGreaterThan(res))
      res.getObj().setLiteral(aa.getObj());
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            ArrayRef<IndexExpr> vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    // Important to get the affine expressions before getting the
    // dims/symbols.
    for (IndexExpr vv : vvals) {
      affineExprs.emplace_back(vv.getAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprScope &scope = vvals[0].getScope();
    int dimNum = scope.getNumDims();
    int symNum = scope.getNumSymbols();
    auto mapContext = scope.getRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    scope.getDimAndSymbolList(dimAndSymList);
    Value minVal = scope.getRewriter().create<affine::AffineMaxOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsKind(minVal, IndexExprKind::NonAffine);
    return res;
  };
  // Res is already defined, we are reducing into it.
  // Integer and Float version.
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
    // Could use arith::max op.
    IndexExpr compareIE = aa > res;
    IndexExpr selectIE = select(compareIE, aa, res);
    res.getObj().initAsKind(selectIE.getValue(), IndexExprKind::NonAffine);
    return res;
  };

  // Empty, treat as integer.
  if (vals.size() > 0 && vals[0].isFloat())
    return reductionOp(vals, litFct, nullptr, valueFct);
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::max(
    IndexExpr const first, IndexExpr const second) {
  SmallVector<IndexExpr, 2> list = {first, second};
  return max(list);
}

/*static*/ IndexExpr IndexExpr::max(
    IndexExpr const first, int64_t const second) {
  SmallVector<IndexExpr, 2> list = {first, LitIE(second)};
  return max(list);
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

bool IndexExpr::retrieveAffineMinMax(
    bool &isMin, llvm::SmallVectorImpl<Value> &vals, AffineMap &map) const {
  Value val = this->getValue();
  auto minOp = val.getDefiningOp<affine::AffineMinOp>();
  auto maxOp = val.getDefiningOp<affine::AffineMaxOp>();
  // Expect here the defining op to be either min or max.
  if (minOp == nullptr && maxOp == nullptr)
    return false;
  isMin = minOp != nullptr;
  if (isMin)
    map = minOp.getAffineMap();
  else
    map = maxOp.getAffineMap();
  IndexExprScope &scope = this->getScope();
  scope.getDimAndSymbolList(vals);
  return true;
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::operator+(int64_t const b) const {
  return *this + LitIE(b);
}

IndexExpr IndexExpr::operator-(int64_t const b) const {
  return *this - LitIE(b);
}

IndexExpr IndexExpr::operator*(int64_t const b) const {
  return *this * LitIE(b);
}

IndexExpr IndexExpr::operator==(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::OEQ, b);
  return compareOp(arith::CmpIPredicate::eq, b);
}

IndexExpr IndexExpr::operator==(int64_t const b) const {
  return *this == LitIE(b);
}

IndexExpr IndexExpr::operator!=(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::ONE, b);
  return compareOp(arith::CmpIPredicate::ne, b);
}

IndexExpr IndexExpr::operator!=(int64_t const b) const {
  return *this != LitIE(b);
}

IndexExpr IndexExpr::operator<=(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::OLE, b);
  return compareOp(arith::CmpIPredicate::sle, b);
}

IndexExpr IndexExpr::operator<=(int64_t const b) const {
  return *this <= LitIE(b);
}

IndexExpr IndexExpr::operator<(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::OLT, b);
  return compareOp(arith::CmpIPredicate::slt, b);
}

IndexExpr IndexExpr::operator<(int64_t const b) const {
  return *this < LitIE(b);
}

IndexExpr IndexExpr::operator>=(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::OGE, b);
  return compareOp(arith::CmpIPredicate::sge, b);
}

IndexExpr IndexExpr::operator>=(int64_t const b) const {
  return *this >= LitIE(b);
}

IndexExpr IndexExpr::operator>(IndexExpr const b) const {
  if (areFloat(b))
    return compareOp(arith::CmpFPredicate::OGT, b);
  return compareOp(arith::CmpIPredicate::sgt, b);
}

IndexExpr IndexExpr::operator>(int64_t const b) const {
  return *this > LitIE(b);
}

IndexExpr IndexExpr::operator%(int64_t const b) const {
  return *this % LitIE(b);
}

IndexExpr IndexExpr::floorDiv(int64_t const b) const {
  return this->floorDiv(LitIE(b));
}

IndexExpr IndexExpr::ceilDiv(int64_t const b) const {
  return this->ceilDiv(LitIE(b));
}

IndexExpr IndexExpr::clamp(int64_t min, IndexExpr max) {
  return clamp(LitIE(min), max);
}

/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, IndexExpr const falseVal) {
  return select(compare, LitIE(trueVal), falseVal);
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, IndexExpr const trueVal, int64_t const falseVal) {
  return select(compare, trueVal, LitIE(falseVal));
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, int64_t const falseVal) {
  return select(compare, LitIE(trueVal), LitIE(falseVal));
}

IndexExpr IndexExpr::selectOrSelf(
    IndexExpr const compare, IndexExpr const trueVal) const {
  return select(compare, trueVal, *this);
}

IndexExpr IndexExpr::selectOrSelf(
    IndexExpr const compare, int64_t const trueVal) const {
  return select(compare, trueVal, *this);
}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing UndefinedIndexExpr.
//===----------------------------------------------------------------------===//

UndefinedIndexExpr::UndefinedIndexExpr() : IndexExpr() {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing LiteralIndexExpr.
//===----------------------------------------------------------------------===//

LiteralIndexExpr::LiteralIndexExpr(int const value) : IndexExpr() {
  init((int64_t)value);
}

LiteralIndexExpr::LiteralIndexExpr(unsigned int const value) : IndexExpr() {
  init((int64_t)value);
}

LiteralIndexExpr::LiteralIndexExpr(int64_t const value) : IndexExpr() {
  init(value);
}

LiteralIndexExpr::LiteralIndexExpr(uint64_t const value) : IndexExpr() {
  init((int64_t)value);
}

LiteralIndexExpr::LiteralIndexExpr(double const value) : IndexExpr() {
  init(value);
}

void LiteralIndexExpr::init(int64_t const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Affine);
}

void LiteralIndexExpr::init(double const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Affine);
}

LiteralIndexExpr::LiteralIndexExpr(IndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  if (o.isFloat())
    init(o.getFloatLiteral());
  else
    init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(UndefinedIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  if (o.isFloat())
    init(o.getFloatLiteral());
  else
    init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(LiteralIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  if (o.isFloat())
    init(o.getFloatLiteral());
  else
    init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(NonAffineIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  if (o.isFloat())
    init(o.getFloatLiteral());
  else
    init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(QuestionmarkIndexExpr const &o)
    : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  assert(!o.isFloat() && "question mark should not be float");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(PredicateIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  assert(!o.isFloat() && "predicate should not be float");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(AffineIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  assert(!o.isFloat() && "affine expressions should not be float");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(DimIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  assert(!o.isFloat() && "dim expressions should not be float");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(SymbolIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  assert(!o.isFloat() && "symbol expressions should not be float");
  init(o.getLiteral());
}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing NonAffineIndexExpr.
//===----------------------------------------------------------------------===//

NonAffineIndexExpr::NonAffineIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsKind(value, IndexExprKind::NonAffine);
}

NonAffineIndexExpr::NonAffineIndexExpr(IndexExprImpl *otherObjPtr)
    : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  // If the index expression is a literal,  just copy it.
  if (otherObjPtr->isLiteral()) {
    if (otherObjPtr->isFloatType()) {
      indexExprObj->initAsLiteral(
          otherObjPtr->getFloatLiteral(), IndexExprKind::Affine);
    } else {
      indexExprObj->initAsLiteral(
          otherObjPtr->getLiteral(), IndexExprKind::Affine);
    }
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark(
        otherObjPtr->getQuestionmark(), otherObjPtr->isFloatType());
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->copy(otherObjPtr);
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make a non-affine from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::NonAffine);
    return;
  }
  case IndexExprKind::Dim: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::NonAffine);
    return;
  }
  case IndexExprKind::Symbol: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::NonAffine);
    return;
  }
  }
  llvm_unreachable("bad path");
}

NonAffineIndexExpr::NonAffineIndexExpr(IndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(UndefinedIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(LiteralIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(NonAffineIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(QuestionmarkIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(PredicateIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(AffineIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(DimIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}
NonAffineIndexExpr::NonAffineIndexExpr(SymbolIndexExpr const &o)
    : NonAffineIndexExpr(o.getObjPtr()) {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing QuestionmarkIndexExpr.
//===----------------------------------------------------------------------===//

QuestionmarkIndexExpr::QuestionmarkIndexExpr(bool isFloatFlag) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsQuestionmark(isFloatFlag);
}

QuestionmarkIndexExpr::QuestionmarkIndexExpr(
    Value tensorOrMemref, int64_t index)
    : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsQuestionmark(tensorOrMemref, index);
}

QuestionmarkIndexExpr::QuestionmarkIndexExpr(IndexExprImpl *otherObjPtr)
    : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  // If the index expression is a question mark, just copy it.
  if (otherObjPtr->isQuestionmark()) {
    indexExprObj->initAsQuestionmark(
        otherObjPtr->getQuestionmark(), otherObjPtr->isFloatType());
    return;
  }
  // Don't care about otherObjPtr, just create a general question mark.
  indexExprObj->initAsQuestionmark(otherObjPtr->isFloatType());
}

QuestionmarkIndexExpr::QuestionmarkIndexExpr(IndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(UndefinedIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(LiteralIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(NonAffineIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(QuestionmarkIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(PredicateIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(AffineIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(DimIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(SymbolIndexExpr const &o)
    : QuestionmarkIndexExpr(o.getObjPtr()) {}

bool QuestionmarkIndexExpr::specificQuestionmark() const {
  assert((getKind() == IndexExprKind::Questionmark) &&
         "Expected QuestionMarkIndexExpr");
  return (getQuestionmark() != ShapedType::kDynamic);
}

bool QuestionmarkIndexExpr::sameQuestionmark(IndexExpr const &o) const {
  if (!o.isQuestionmark())
    return false;
  QuestionmarkIndexExpr oQM(o);
  return (specificQuestionmark() && oQM.specificQuestionmark() &&
          (getQuestionmark() == oQM.getQuestionmark()));
}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing PredicateIndexExpr.
//===----------------------------------------------------------------------===//

PredicateIndexExpr::PredicateIndexExpr(bool const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral((int64_t)value, IndexExprKind::Predicate);
}

PredicateIndexExpr::PredicateIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsKind(value, IndexExprKind::Predicate);
}

PredicateIndexExpr::PredicateIndexExpr(IndexExprImpl *otherObjPtr)
    : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  // TODO: what if other is questionmark?
  assert(!otherObjPtr->isFloatType() && "predicate not be float");
  // If the index expression is a literal,  just copy it.
  if (otherObjPtr->isLiteral()) {
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Predicate);
    return;
  }
  assert(otherObjPtr->getKind() == IndexExprKind::Predicate &&
         "can only make a predicate from another predicate");
  indexExprObj->copy(otherObjPtr);
}

PredicateIndexExpr::PredicateIndexExpr(IndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(UndefinedIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(LiteralIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(NonAffineIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(QuestionmarkIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(PredicateIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(AffineIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(DimIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}
PredicateIndexExpr::PredicateIndexExpr(SymbolIndexExpr const &o)
    : PredicateIndexExpr(o.getObjPtr()) {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing AffineIndexExpr.
//===----------------------------------------------------------------------===//

AffineIndexExpr::AffineIndexExpr(AffineExpr const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsAffineExpr(value);
}

AffineIndexExpr::AffineIndexExpr(IndexExprImpl *otherObjPtr) : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  // If the index expression is a literal,  just copy it.
  if (otherObjPtr->isLiteral()) {
    if (otherObjPtr->isFloatType()) {
      indexExprObj->initAsLiteral(
          otherObjPtr->getFloatLiteral(), IndexExprKind::Affine);
    } else {
      indexExprObj->initAsLiteral(
          otherObjPtr->getLiteral(), IndexExprKind::Affine);
    }
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherObjPtr->isInCurrentScope();
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark(
        otherObjPtr->getQuestionmark(), otherObjPtr->isFloatType());
    return;
  }
  case IndexExprKind::NonAffine: {
    llvm_unreachable("cannot make an affine from an non affine, affine are "
                     "made of literals, dims, and symbols");
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make an affine from a predicate");
  }
  case IndexExprKind::Affine: {
    assert(isSameScope && "cannot can only import literals, dims and symbols "
                          "from different scopes");
    indexExprObj->copy(otherObjPtr);
    return;
  }
  case IndexExprKind::Dim:
  case IndexExprKind::Symbol: {
    assert(isSameScope && "cannot can only import literals, dims and symbols "
                          "from different scopes");
    indexExprObj->initAsAffineExpr(otherObjPtr->getAffineExpr());
    return;
  }
  }
  llvm_unreachable("bad path");
}

AffineIndexExpr::AffineIndexExpr(IndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(UndefinedIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(LiteralIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(NonAffineIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(QuestionmarkIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(PredicateIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(AffineIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(DimIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}
AffineIndexExpr::AffineIndexExpr(SymbolIndexExpr const &o)
    : AffineIndexExpr(o.getObjPtr()) {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing DimIndexExpr.
//===----------------------------------------------------------------------===//

DimIndexExpr::DimIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsKind(value, IndexExprKind::Dim);
}

DimIndexExpr::DimIndexExpr(IndexExprImpl *otherObjPtr) : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  // If the index expression is a literal,  just copy it.
  assert(!otherObjPtr->isFloatType() && "dim cannot be float");
  if (otherObjPtr->isLiteral()) {
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherObjPtr->isInCurrentScope();
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark(
        otherObjPtr->getQuestionmark(), otherObjPtr->isFloatType());
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make an dim from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Dim: {
    // If replicated in the same scope, its not great but will not gen errors.
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Symbol: {
    assert(!isSameScope && "cannot make a dim from a symbol at the same scope");
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Dim);
    return;
  }
  }
  llvm_unreachable("bad path");
}

DimIndexExpr::DimIndexExpr(IndexExpr const &o) : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(UndefinedIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(LiteralIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(NonAffineIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(QuestionmarkIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(PredicateIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(AffineIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(DimIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}
DimIndexExpr::DimIndexExpr(SymbolIndexExpr const &o)
    : DimIndexExpr(o.getObjPtr()) {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing SymbolIndexExpr.
//===----------------------------------------------------------------------===//

SymbolIndexExpr::SymbolIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsKind(value, IndexExprKind::Symbol);
}

SymbolIndexExpr::SymbolIndexExpr(IndexExprImpl *otherObjPtr) : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
  assert(!otherObjPtr->isFloatType() && "dim cannot be float");
  // If the index expression is a literal,  just copy it.
  if (otherObjPtr->isLiteral()) {
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherObjPtr->isInCurrentScope();
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark(
        otherObjPtr->getQuestionmark(), otherObjPtr->isFloatType());
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make an symbol from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Dim: {
    assert(!isSameScope && "cannot make a symbol from a dim in the same scope");
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Symbol: {
    // If replicated in the same scope, its not great but will not gen errors.
    indexExprObj->initAsKind(otherObjPtr->getValue(), IndexExprKind::Symbol);
    return;
  }
  }
  llvm_unreachable("bad path");
}

SymbolIndexExpr::SymbolIndexExpr(IndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(UndefinedIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(LiteralIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(NonAffineIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(QuestionmarkIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(PredicateIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(AffineIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(DimIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}
SymbolIndexExpr::SymbolIndexExpr(SymbolIndexExpr const &o)
    : SymbolIndexExpr(o.getObjPtr()) {}

//===----------------------------------------------------------------------===//
// List helpers
//===----------------------------------------------------------------------===//

void getIndexExprListFromInt(
    ArrayRef<int64_t> inputList, llvm::SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (int64_t item : inputList)
    outputList.emplace_back(LitIE(item));
}

// Create a list of IndexExpr of kind LiteralIndexExpr/Questionmark from a
// shape.
void getIndexExprListFromShape(
    ArrayRef<int64_t> inputList, llvm::SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (int64_t item : inputList) {
    if (item == ShapedType::kDynamic)
      outputList.emplace_back(QuestionmarkIndexExpr(/*isFloat*/ false));
    else {
      assert(item >= 0 && "expected kDynamic, not -1");
      outputList.emplace_back(LitIE(item));
    }
  }
}

} // namespace onnx_mlir
