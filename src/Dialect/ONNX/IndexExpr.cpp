/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------IndexExpr.cpp - Index expression---------------------=== //
//
// copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculation using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/IndexExprDetail.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "index_expr"

using namespace mlir;

//===----------------------------------------------------------------------===//
// IndexExprScope constructors.
//===----------------------------------------------------------------------===//

// Initial scope.
IndexExprScope::IndexExprScope(OpBuilder *rewriter, Location loc)
    : dims(), symbols(), rewriter(rewriter), loc(loc),
      parentScope(getCurrentScopePtr()), container() {
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(DialectBuilder &db)
    : IndexExprScope(&db.getBuilder(), db.getLoc()) {}

// Nested scopes.
IndexExprScope::IndexExprScope(
    OpBuilder *innerRewriter, IndexExprScope *enclosingScope)
    : dims(), symbols(), rewriter(innerRewriter), loc(enclosingScope->loc),
      parentScope(enclosingScope), container() {
  // Check the enclosing scope is the current one.
  assert(enclosingScope == getCurrentScopePtr() &&
         "provided parent scope was not the previously active scope");
  // Install new inner scope as current one.
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(
    DialectBuilder &innerDb, IndexExprScope *enclosingScope)
    : IndexExprScope(&innerDb.getBuilder(), enclosingScope) {}

IndexExprScope::~IndexExprScope() {
  // Free the memory of each IndexExprImpl in scope's container.
  for (IndexExprImpl *obj : container)
    delete obj;
  container.clear();
  // no need to clear the cached copies as they are also in the container.
  getCurrentScopePtr() = parentScope;
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
  assert(newImplObj && "failed to allocate IndexExpr implemtation");
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

bool IndexExpr::isLiteralAndIdenticalTo(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() == b);
}

bool IndexExpr::isLiteralAndIdenticalTo(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  return b.isLiteral() && isLiteralAndIdenticalTo(b.getLiteral());
}

bool IndexExpr::isLiteralAndDifferentThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() != b);
}

bool IndexExpr::isLiteralAndDifferentThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  return b.isLiteral() && isLiteralAndDifferentThan(b.getLiteral());
}

bool IndexExpr::isLiteralAndGreaterThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() > b);
}

bool IndexExpr::isLiteralAndGreaterThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  return b.isLiteral() && isLiteralAndGreaterThan(b.getLiteral());
}

bool IndexExpr::isLiteralAndSmallerThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  return isLiteral() && (getLiteral() < b);
}

bool IndexExpr::isLiteralAndSmallerThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  return b.isLiteral() && isLiteralAndSmallerThan(b.getLiteral());
}

// All element in list are literals.
/*static*/ bool IndexExpr::isLiteral(SmallVectorImpl<IndexExpr> &list) {
  for (IndexExpr i : list)
    if (!i.isLiteral())
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
    // Its ok to use a nonafine index expressions from enclosing scopes.
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
  llvm_unreachable("unkown kind");
}

//===----------------------------------------------------------------------===//
// IndexExpr public getter.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const { return getObj().getLiteral(); }

AffineExpr IndexExpr::getAffineExpr() const { return getObj().getAffineExpr(); }

Value IndexExpr::getValue() const { return getObj().getValue(); }

void IndexExpr::getAffineMapAndOperands(
    AffineMap &map, SmallVectorImpl<Value> &operands) const {
  getObj().getAffineMapAndOperands(map, operands);
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
    llvm::dbgs() << msg.c_str();
    if (!isDefined()) {
      llvm::dbgs() << " undefined\n";
      return;
    }
    if (isLiteral())
      llvm::dbgs() << " literal(" << (long long)getLiteral() << ")";
    if (hasAffineExpr())
      llvm::dbgs() << " hasAffine";
    if (hasValue()) {
      llvm::dbgs() << " hasValue";
      auto op = getValue().getDefiningOp();
      if (op) {
        std::string str;
        llvm::raw_string_ostream os(str);
        op->print(os);
        llvm::dbgs() << "( \"" << str.c_str() << "\")";
      } else
        llvm::dbgs() << "(op not found)";
    }
    if (isAffine())
      llvm::dbgs() << " is affine";
    switch (getKind()) {
    case IndexExprKind::NonAffine:
      llvm::dbgs() << " kind(non-affine)";
      break;
    case IndexExprKind::Questionmark:
      llvm::dbgs() << " kind(questionmark)";
      break;
    case IndexExprKind::Predicate:
      llvm::dbgs() << " kind(predicate)";
      break;
    case IndexExprKind::Affine:
      llvm::dbgs() << " kind(affine)";
      break;
    case IndexExprKind::Dim:
      llvm::dbgs() << " kind(dim)";
      break;
    case IndexExprKind::Symbol:
      llvm::dbgs() << " kind(symbol)";
      break;
    }
    llvm::dbgs() << " scope(0x " << (long long unsigned)getScopePtr() << ")\n";
  });
}

void IndexExpr::debugPrint(
    const std::string &msg, const SmallVectorImpl<IndexExpr> &list) {
  LLVM_DEBUG({
    int s = list.size();
    llvm::dbgs() << msg.c_str() << " (" << s << "elements)\n";
    for (int i = 0; i < s; ++i) {
      std::string element = "  " + std::to_string(i) + ": ";
      list[i].debugPrint(element);
    }
  });
}

//===----------------------------------------------------------------------===//
// Helpers for IndexExpressions
//===----------------------------------------------------------------------===//

/*static*/ void IndexExpr::getShape(SmallVectorImpl<IndexExpr> &indexExprList,
    SmallVectorImpl<int64_t> &intDimList) {
  intDimList.clear();
  for (IndexExpr &expr : indexExprList) {
    if (expr.isLiteral()) {
      int64_t val = expr.getLiteral();
      assert(val >= 0 && "expected positive values only");
      intDimList.emplace_back(val);
    } else
      intDimList.emplace_back(-1);
  }
}

/*static*/ void IndexExpr::getValues(
    ArrayRef<IndexExpr> indexExprArray, SmallVectorImpl<Value> &valueList) {
  valueList.clear();
  for (IndexExpr const &expr : indexExprArray)
    valueList.emplace_back(expr.getValue());
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Used for add/sub/mult/ceilDiv/floorDiv
IndexExpr IndexExpr::binaryOp(IndexExpr const b, bool affineWithLitB,
    bool canBeAffine, F2 litFct, F2 affineExprFct, F2 valueFct) const {
  assert(canBeUsedInScope() && "a cannot be used in current scope");
  assert(b.canBeUsedInScope() && "b cannot be used in current scope");
  // Literal integer if a and b are literals. Affine if canBeAffine is true,
  // both a and b are affine, and possibly a and/or b are also constant.
  bool resIsLit = isLiteral() && b.isLiteral();
  bool resIsAffine = resIsLit || (canBeAffine && isAffine() && b.isAffine() &&
                                     (!affineWithLitB || b.isLiteral()));

  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this, b);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values &
    // Affine expr undefined.
    return QuestionmarkIndexExpr();
  if (resIsAffine)
    // Use affine values.
    return affineExprFct(*this, b);
  // Use values.
  return valueFct(*this, b);
}

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
  return binaryOp(b, false, false, litFct, nullptr, valueFct);
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
    return QuestionmarkIndexExpr();
  // Not literals or questionmark, we must have predicates.
  assert(isPredType() && "expected predicate index expression");
  assert(b.isPredType() && "expected predicate index expression");
  Value res =
      getRewriter().create<arith::AndIOp>(getLoc(), getValue(), b.getValue());
  return PredicateIndexExpr(res);
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
    return QuestionmarkIndexExpr();
  // Not literals or questionmark, we must have predicates.
  assert(isPredType() && "expected predicate index expression");
  assert(b.isPredType() && "expected predicate index expression");
  Value res =
      getRewriter().create<arith::OrIOp>(getLoc(), getValue(), b.getValue());
  return PredicateIndexExpr(res);
}

IndexExpr IndexExpr::operator!() const {
  return (*this == PredicateIndexExpr(false));
}

// The affine reduction lambda function processes the whole list and must init
// the result. Literal and Values treat one operation at a time
/* static*/ IndexExpr IndexExpr::reductionOp(SmallVectorImpl<IndexExpr> &vals,
    F2Self litRed, Flist affineRed, F2Self valueRed) {
  // If no values, result is undefined.
  int size = vals.size();
  if (size == 0) {
    return UndefinedIndexExpr();
  }
  // Set the output to the first value.
  IndexExpr res = vals[0].deepCopy();
  // If list has one element, we are done. Literal/Affine... will be the same
  // as this single element.
  if (vals.size() == 1)
    return res;
  // Have multiple values, need to do some checks.
  bool resIsLit = true;
  bool resIsAffine = true;
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
    return QuestionmarkIndexExpr();
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
    return LiteralIndexExpr(aa.getLiteral() + bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() + bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return NonAffineIndexExpr(aa.getRewriter().create<arith::AddIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return binaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator-(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LiteralIndexExpr(aa.getLiteral() - bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() - bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return NonAffineIndexExpr(aa.getRewriter().create<arith::SubIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return binaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator*(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return LiteralIndexExpr(aa.getLiteral() * bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return AffineIndexExpr(aa.getAffineExpr() * bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1)
      return aa.deepCopy();
    return NonAffineIndexExpr(aa.getRewriter().create<arith::MulIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Literal should be place in second argument; do so if a is a lit.
  if (isLiteral())
    return b.binaryOp(*this, true, true, litFct, affineExprFct, valueFct);
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::floorDiv(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = floor((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return LiteralIndexExpr(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return AffineIndexExpr(aa.getAffineExpr().floorDiv(bval));
    return NonAffineIndexExpr(aa.getRewriter().create<arith::FloorDivSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<arith::FloorDivSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::ceilDiv(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = ceil((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return LiteralIndexExpr(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return AffineIndexExpr(aa.getAffineExpr().ceilDiv(bval));
    return NonAffineIndexExpr(aa.getRewriter().create<arith::CeilDivSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<arith::CeilDivSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator%(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = mlir::mod(aa.getLiteral(), bb.getLiteral());
    return LiteralIndexExpr(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval >= 0)
      return AffineIndexExpr(aa.getAffineExpr() % bval);
    return NonAffineIndexExpr(aa.getRewriter().create<arith::RemSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<arith::RemSIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::clamp(IndexExpr const min, IndexExpr const max) const {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 litFct = [](IndexExpr const val, IndexExpr const min,
                  IndexExpr const max) -> IndexExpr {
    // assume signed compares
    int64_t smin = min.getLiteral();
    int64_t smax = max.getLiteral();
    int64_t res = val.getLiteral();
    if (res < smin)
      res = smin;
    if (res > smax)
      res = smax;
    return LiteralIndexExpr(res);
  };
  F3 valueFct = [](IndexExpr const val, IndexExpr const min,
                    IndexExpr const max) {
    IndexExpr res1 = select(val < min, min, val);
    IndexExpr res2 = select(res1 > max, max, res1);
    return res2;
  };

  assert(canBeUsedInScope() && "cannot be used in current scope");
  assert(min.canBeUsedInScope() && "min cannot be used in current scope");
  assert(max.canBeUsedInScope() && "max cannot be used in current scope");

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
    return QuestionmarkIndexExpr();
  // Use values.
  return valueFct(*this, min, max);
}

/*static*/ IndexExpr IndexExpr::select(IndexExpr const compare,
    IndexExpr const trueVal, IndexExpr const falseVal) {
  assert(
      compare.canBeUsedInScope() && "compare cannot be used in current scope");
  assert(
      trueVal.canBeUsedInScope() && "trueVal cannot be used in current scope");
  assert(falseVal.canBeUsedInScope() &&
         "falseVal cannot be used in current scope");
  // When compare result is literal, just feed forward the right value.
  if (compare.isLiteral()) {
    if (compare.getLiteral())
      return trueVal.deepCopy();
    return falseVal.deepCopy();
  }
  // Dynamic value, just set as undefined during shape inference pass.
  if (compare.isShapeInferencePass())
    return QuestionmarkIndexExpr();
  // Generate code for the select.
  Value results =
      compare.getRewriter().create<arith::SelectOp>(compare.getLoc(),
          compare.getValue(), trueVal.getValue(), falseVal.getValue());
  return NonAffineIndexExpr(results);
}

/*static*/ IndexExpr IndexExpr::min(SmallVectorImpl<IndexExpr> &vals) {
  // Res is already an literal int, we are reducing into it.
  F2Self litFct = [](IndexExpr res, IndexExpr const aa) -> IndexExpr {
    int64_t rrr = res.getLiteral();
    int64_t aaa = aa.getLiteral();
    if (aaa < rrr)
      res.getObj().intLit = aaa;
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            SmallVectorImpl<IndexExpr> &vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    // Important to get the affine expressions before getting the dims/symbols.
    for (IndexExpr &vv : vvals) {
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
    Value minVal = scope.getRewriter().create<AffineMinOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsKind(minVal, IndexExprKind::NonAffine);
    return res;
  };
  // Res is already defined, we are reducing into it.
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
    Value compareVal = res.getRewriter().create<arith::CmpIOp>(
        res.getLoc(), arith::CmpIPredicate::slt, aa.getValue(), res.getValue());
    Value resVal = res.getRewriter().create<arith::SelectOp>(
        res.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsKind(resVal, IndexExprKind::NonAffine);
    return res;
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::min(
    IndexExpr const first, IndexExpr const second) {
  SmallVector<IndexExpr, 2> list = {first, second};
  return min(list);
}

/*static*/ IndexExpr IndexExpr::min(
    IndexExpr const first, int64_t const second) {
  SmallVector<IndexExpr, 2> list = {first, LiteralIndexExpr(second)};
  return min(list);
}

/*static*/ IndexExpr IndexExpr::max(SmallVectorImpl<IndexExpr> &vals) {
  // Res is already an literal int, we are reducing into it.
  F2Self litFct = [](IndexExpr res, IndexExpr const aa) -> IndexExpr {
    int64_t rrr = res.getLiteral();
    int64_t aaa = aa.getLiteral();
    if (aaa > rrr)
      res.getObj().intLit = aaa;
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            SmallVectorImpl<IndexExpr> &vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    // Important to get the affine expressions before getting the dims/symbols.
    for (IndexExpr &vv : vvals) {
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
    Value minVal = scope.getRewriter().create<AffineMaxOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsKind(minVal, IndexExprKind::NonAffine);
    return res;
  };
  // Res is already defined, we are reducing into it.
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
    Value compareVal = res.getRewriter().create<arith::CmpIOp>(
        res.getLoc(), arith::CmpIPredicate::sgt, aa.getValue(), res.getValue());
    Value resVal = res.getRewriter().create<arith::SelectOp>(
        res.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsKind(resVal, IndexExprKind::NonAffine);
    return res;
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::max(
    IndexExpr const first, IndexExpr const second) {
  SmallVector<IndexExpr, 2> list = {first, second};
  return max(list);
}

/*static*/ IndexExpr IndexExpr::max(
    IndexExpr const first, int64_t const second) {
  SmallVector<IndexExpr, 2> list = {first, LiteralIndexExpr(second)};
  return max(list);
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::operator+(int64_t const b) const {
  return *this + LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator-(int64_t const b) const {
  return *this - LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator*(int64_t const b) const {
  return *this * LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator==(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::eq, b);
}

IndexExpr IndexExpr::operator==(int64_t const b) const {
  return *this == LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator!=(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::ne, b);
}

IndexExpr IndexExpr::operator!=(int64_t const b) const {
  return *this != LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator<=(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::sle, b);
}

IndexExpr IndexExpr::operator<=(int64_t const b) const {
  return *this <= LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator<(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::slt, b);
}

IndexExpr IndexExpr::operator<(int64_t const b) const {
  return *this < LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator>=(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::sge, b);
}

IndexExpr IndexExpr::operator>=(int64_t const b) const {
  return *this >= LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator>(IndexExpr const b) const {
  return compareOp(arith::CmpIPredicate::sgt, b);
}

IndexExpr IndexExpr::operator>(int64_t const b) const {
  return *this > LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator%(int64_t const b) const {
  return *this % LiteralIndexExpr(b);
}

IndexExpr IndexExpr::floorDiv(int64_t const b) const {
  return this->floorDiv(LiteralIndexExpr(b));
}

IndexExpr IndexExpr::ceilDiv(int64_t const b) const {
  return this->ceilDiv(LiteralIndexExpr(b));
}

IndexExpr IndexExpr::clamp(int64_t min, IndexExpr max) {
  return clamp(LiteralIndexExpr(min), max);
}

/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, IndexExpr const falseVal) {
  return select(compare, LiteralIndexExpr(trueVal), falseVal);
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, IndexExpr const trueVal, int64_t const falseVal) {
  return select(compare, trueVal, LiteralIndexExpr(falseVal));
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, int64_t const falseVal) {
  return select(compare, LiteralIndexExpr(trueVal), LiteralIndexExpr(falseVal));
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

LiteralIndexExpr::LiteralIndexExpr(int64_t const value) : IndexExpr() {
  init(value);
}

void LiteralIndexExpr::init(int64_t const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Affine);
}

LiteralIndexExpr::LiteralIndexExpr(IndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(UndefinedIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(LiteralIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(NonAffineIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(QuestionmarkIndexExpr const &o)
    : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(PredicateIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(AffineIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(DimIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
LiteralIndexExpr::LiteralIndexExpr(SymbolIndexExpr const &o) : IndexExpr() {
  assert(o.isLiteral() && "cannot make a literal from non literal");
  init(o.getLiteral());
}
//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing NonAffineIndexExpr.
//===----------------------------------------------------------------------===//

NonAffineIndexExpr::NonAffineIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
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
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
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

QuestionmarkIndexExpr::QuestionmarkIndexExpr() : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsQuestionmark();
}

// Don't care about otherIndexExpr as questionmarks have no real data.

QuestionmarkIndexExpr::QuestionmarkIndexExpr(IndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(UndefinedIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(LiteralIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(NonAffineIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(QuestionmarkIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(PredicateIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(AffineIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(DimIndexExpr const &o)
    : QuestionmarkIndexExpr() {}
QuestionmarkIndexExpr::QuestionmarkIndexExpr(SymbolIndexExpr const &o)
    : QuestionmarkIndexExpr() {}

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing PredicateIndexExpr.
//===----------------------------------------------------------------------===//

PredicateIndexExpr::PredicateIndexExpr(bool const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Predicate);
}

PredicateIndexExpr::PredicateIndexExpr(Value const value) : IndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
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
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
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
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherObjPtr->isInCurrentScope();
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
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
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
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
  if (otherObjPtr->isLiteral()) {
    indexExprObj->initAsLiteral(
        otherObjPtr->getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherObjPtr->isInCurrentScope();
  switch (otherObjPtr->getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
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
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsKind(value, IndexExprKind::Symbol);
}

SymbolIndexExpr::SymbolIndexExpr(IndexExprImpl *otherObjPtr) : IndexExpr() {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If undefined, nothing to do.
  if (!otherObjPtr)
    return;
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
    indexExprObj->initAsQuestionmark();
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
// Capturing Index Expressions: Array of values
//===----------------------------------------------------------------------===//

ArrayValueIndexCapture::ArrayValueIndexCapture(
    Value array, GetDenseVal fGetDenseVal, LoadVal fLoadVal)
    : array(array), hasDefault(false), fGetDenseArrayAttr(fGetDenseVal),
      fLoadVallFromArrayAtIndex(fLoadVal) {}

ArrayValueIndexCapture::ArrayValueIndexCapture(Value array,
    int64_t defaultLiteral, GetDenseVal fGetDenseVal, LoadVal fLoadVal)
    : array(array), defaultLiteral(defaultLiteral), hasDefault(true),
      fGetDenseArrayAttr(fGetDenseVal), fLoadVallFromArrayAtIndex(fLoadVal) {}

IndexExpr ArrayValueIndexCapture::getSymbol(uint64_t i) {
  // Check if we have an operand.
  if (array.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value if there is one.
    if (hasDefault)
      return LiteralIndexExpr(defaultLiteral);
    // Has no default: error
    return UndefinedIndexExpr();
  }
  // Check if we have an array of literals.
  assert(fGetDenseArrayAttr && "expected method to get a dense array");
  if (DenseElementsAttr attrArray = fGetDenseArrayAttr(array)) {
    // We extracted an dense attribute from definition of operand.
    int64_t dimSize;
    if (attrArray.getType().getRank() == 0)
      dimSize = 1;
    else
      dimSize = attrArray.getType().getDimSize(0);
    if ((int64_t)i >= dimSize) {
      // Request beyond available size.
      if (hasDefault)
        return LiteralIndexExpr(defaultLiteral);
      // Has no default: error
      return UndefinedIndexExpr();
    }
    auto attrVal = attrArray.getValues<Attribute>()[ArrayRef<uint64_t>({i})];
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return LiteralIndexExpr(attrInt);
  }

  // We must read value from an array.
  IndexExprScope &scope = IndexExprScope::getCurrentScope();
  if (scope.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return QuestionmarkIndexExpr();
  }
  // Emit code to read array.
  assert(fLoadVallFromArrayAtIndex && "expected method to load an array value");
  Value loadVal =
      fLoadVallFromArrayAtIndex(scope.getRewriter(), scope.getLoc(), array, i);
  return SymbolIndexExpr(loadVal);
}

bool ArrayValueIndexCapture::getSymbolList(
    int num, SmallVectorImpl<IndexExpr> &symbolList) {
  // Clear output.
  symbolList.clear();
  for (int i = 0; i < num; ++i) {
    IndexExpr sym = getSymbol(i);
    if (sym.isUndefined()) {
      symbolList.clear();
      return false;
    }
    symbolList.emplace_back(sym);
  }
  return true;
}

bool ArrayValueIndexCapture::getSymbolList(
    SmallVectorImpl<IndexExpr> &symbolList) {
  symbolList.clear();
  auto shapeType = array.getType().dyn_cast_or_null<ShapedType>();
  if (!shapeType)
    return false; // Assume error if its not a shape type.
  int rank = shapeType.getRank();
  assert(rank <= 1 && "Array value index capture supports const or 1D arrays");
  int num = (rank == 0) ? 1 : shapeType.getShape()[0];
  if (num == -1)
    return false; // Cannot read an unranked array.
  return getSymbolList(num, symbolList);
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions: Array of values
//===----------------------------------------------------------------------===//

ArrayAttributeIndexCapture::ArrayAttributeIndexCapture(ArrayAttr array)
    : array(array), arraySize((array) ? array.size() : 0), defaultLiteral(0),
      hasDefault(false) {}

ArrayAttributeIndexCapture::ArrayAttributeIndexCapture(
    ArrayAttr array, int64_t defaultLiteral)
    : array(array), arraySize((array) ? array.size() : 0),
      defaultLiteral(defaultLiteral), hasDefault(true) {}

IndexExpr ArrayAttributeIndexCapture::getLiteral(uint64_t i) {
  if (i < arraySize) {
    int64_t val = (array.getValue()[i]).cast<IntegerAttr>().getInt();
    return LiteralIndexExpr(val);
  }
  if (hasDefault)
    return LiteralIndexExpr(defaultLiteral);
  return UndefinedIndexExpr();
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions: MemRef Bounds
//===----------------------------------------------------------------------===//

MemRefBoundsIndexCapture::MemRefBoundsIndexCapture()
    : tensorOrMemref(nullptr), memRank(0) {}

MemRefBoundsIndexCapture::MemRefBoundsIndexCapture(Value tensorOrMemref)
    : tensorOrMemref(tensorOrMemref), memRank(0) {
  if (tensorOrMemref) {
    ShapedType shapedType =
        tensorOrMemref.getType().dyn_cast_or_null<ShapedType>();
    if (shapedType)
      memRank = shapedType.getShape().size();
  }
}

bool MemRefBoundsIndexCapture::isLiteral(int64_t i) {
  assert(tensorOrMemref && "Expected defined tensor or memref");
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  return (shape[i] >= 0);
}

int64_t MemRefBoundsIndexCapture::getShape(int64_t i) {
  assert(tensorOrMemref && "Expected defined tensor or memref");
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  return shape[i];
}

bool MemRefBoundsIndexCapture::areAllLiteral() {
  assert(tensorOrMemref && "Expected defined tensor or memref");
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  for (unsigned int i = 0; i < memRank; ++i)
    if (shape[i] < 0)
      return false;
  return true;
}

IndexExpr MemRefBoundsIndexCapture::getDim(uint64_t i) {
  return get<DimIndexExpr>(i);
}

IndexExpr MemRefBoundsIndexCapture::getSymbol(uint64_t i) {
  return get<SymbolIndexExpr>(i);
}

// Assert if not a literal.
IndexExpr MemRefBoundsIndexCapture::getLiteral(uint64_t i) {
  assert(tensorOrMemref && "Expected defined tensor or memref");
  assert(i < memRank && "out of bound access");
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  if (shape[i] >= 0) {
    // We have a constant dimension.
    int64_t intVal = shape[i];
    return LiteralIndexExpr(intVal);
  }
  llvm_unreachable("expected a literal");
}

void MemRefBoundsIndexCapture::getDimList(SmallVectorImpl<IndexExpr> &dimList) {
  getList<DimIndexExpr>(dimList);
}

void MemRefBoundsIndexCapture::getSymbolList(
    SmallVectorImpl<IndexExpr> &symbolList) {
  getList<SymbolIndexExpr>(symbolList);
}

void MemRefBoundsIndexCapture::getLiteralList(
    SmallVectorImpl<IndexExpr> &literalList) {
  // Clear output.
  literalList.clear();
  // Scan tensor or memref.
  for (unsigned int i = 0; i < memRank; ++i)
    literalList.emplace_back(getLiteral(i));
}

template <class INDEX>
IndexExpr MemRefBoundsIndexCapture::get(uint64_t i) {
  assert(tensorOrMemref && "Expected defined tensor or memref");
  assert(i < memRank && "index out of bound");

  Type type = tensorOrMemref.getType();
  ArrayRef<int64_t> shape = type.cast<ShapedType>().getShape();
  if (shape[i] >= 0) {
    // We have a constant dimension.
    int64_t intVal = shape[i];
    return LiteralIndexExpr(intVal);
  }
  // We have a dynamic dimension.
  IndexExprScope &scope = IndexExprScope::getCurrentScope();
  if (scope.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return QuestionmarkIndexExpr();
  }

  MemRefBuilder createMemRef(scope.getRewriter(), scope.getLoc());
  Value dynVal = createMemRef.dim(tensorOrMemref, i);
  return INDEX(dynVal);
}

template <class INDEX>
void MemRefBoundsIndexCapture::getList(SmallVectorImpl<IndexExpr> &list) {
  // Clear output.
  list.clear();
  // Scan tensor or memref.
  for (unsigned int i = 0; i < memRank; ++i)
    list.emplace_back(get<INDEX>(i));
}
