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

// both debug variables will be removed once debugging is complete.
#define DEBUG 0

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/IndexExprDetail.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// IndexExprScope constructors.
//===----------------------------------------------------------------------===//

IndexExprScope::IndexExprScope(OpBuilder *rewriter, Location loc)
    : dims(), symbols(), rewriter(rewriter), loc(loc),
      parentScope(getCurrentScopePtr()), container() {
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(OpBuilder &rewriter, Location loc)
    : IndexExprScope(&rewriter, loc){};

IndexExprScope::IndexExprScope()
    : dims(), symbols(), rewriter(getCurrentScope().rewriter),
      loc(getCurrentScope().loc), parentScope(getCurrentScopePtr()),
      container() {
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(IndexExprScope &explicitEnclosingScope)
    : IndexExprScope() {
  assert(&explicitEnclosingScope == parentScope &&
         "provided parent scope was not the previously active scope");
}

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

int IndexExprScope::addDim(Value const value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprScope::addSymbol(Value const value) {
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

//===----------------------------------------------------------------------===//
// IndexExprScope getters.
//===----------------------------------------------------------------------===//

bool IndexExprScope::isCurrentScope() { return getCurrentScopePtr() == this; }

bool IndexExprScope::isEnclosingScope() {
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
  assert(rewriter);
  return *rewriter;
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
  if (!isLiteral())
    return false;
  // We have a literal, now make sure they are the same
  return getLiteral() == b;
}

bool IndexExpr::isLiteralAndDifferentThan(int64_t b) const {
  // When dealing with non-literal, don't test and return false.
  if (!isLiteral())
    return false;
  // We have a literal, now make sure they are different
  return getLiteral() != b;
}

bool IndexExpr::isLiteralAndIdenticalTo(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!isLiteral() || !b.isLiteral())
    return false;
  // We have literals, now make sure they are the same
  return getLiteral() == b.getLiteral();
}

bool IndexExpr::isLiteralAndDifferentThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return false.
  if (!isLiteral() || !b.isLiteral())
    return false;
  // We have literals, now make sure they are different
  return getLiteral() != b.getLiteral();
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
    printf(" kind(predicate)");
    break;
  case IndexExprKind::Affine:
  case IndexExprKind::Dim:
  case IndexExprKind::Symbol:
    // Because affine/dim/symbols are specific to a current scope, they have to
    // be converted to the current scope before being used. They cannot be used
    // out of current scope.
    return false;
  default:
    break;
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
#if DEBUG
  printf("%s:", msg.c_str());
  if (isLiteral())
    printf(" literal(%lli)", getLiteral());
  if (hasAffineExpr())
    printf(" hasAffine");
  if (hasValue()) {
    printf(" hasValue");
    auto op = getValue().getDefiningOp();
    if (op) {
      std::string str;
      llvm::raw_string_ostream os(str);
      op->print(os);
      printf("( \"%s\" )", str.c_str());
    } else
      printf("(op not found)");
  }
  if (isAffine())
    printf(" is affine");
  switch (getKind()) {
  case IndexExprKind::NonAffine:
    printf(" kind(non-affine)");
    break;
  case IndexExprKind::Questionmark:
    printf(" kind(questionmark)");
    break;
  case IndexExprKind::Predicate:
    printf(" kind(predicate)");
    break;
  case IndexExprKind::Affine:
    printf(" kind(affine)");
    break;
  case IndexExprKind::Dim:
    printf(" kind(dim)");
    break;
  case IndexExprKind::Symbol:
    printf(" kind(symbol)");
    break;
  default:
    printf(" kind(unknown)");
    break;
  }
  printf(" scope(0x%llx)\n", (long long unsigned)getScopePtr());

#endif
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
    CmpIPredicate comparePred, IndexExpr const b) const {
  F2 litFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t aaa = aa.getLiteral();
    int64_t bbb = bb.getLiteral();
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (aaa == bbb)
        return PredicateIndexExpr(true);
      break;
    case CmpIPredicate::ne:
      if (aaa != bbb)
        return PredicateIndexExpr(true);
      break;
    case CmpIPredicate::slt:
      if (aaa < bbb)
        return PredicateIndexExpr(true);
      break;
    case CmpIPredicate::sle:
      if (aaa <= bbb)
        return PredicateIndexExpr(true);
      break;
    case CmpIPredicate::sgt:
      if (aaa > bbb)
        return PredicateIndexExpr(true);
      break;
    case CmpIPredicate::sge:
      if (aaa >= bbb)
        return PredicateIndexExpr(true);
      break;
    default:
      llvm_unreachable("unknown or illegal (unsigned) compare operator");
    }
    return PredicateIndexExpr(false);
  };
  F2 valueFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    Value compare = aa.getRewriter().create<CmpIOp>(
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
  Value res = getRewriter().create<AndOp>(getLoc(), getValue(), b.getValue());
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
  Value res = getRewriter().create<OrOp>(getLoc(), getValue(), b.getValue());
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
    return NonAffineIndexExpr(aa.getRewriter().create<AddIOp>(
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
    return NonAffineIndexExpr(aa.getRewriter().create<SubIOp>(
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
    return NonAffineIndexExpr(aa.getRewriter().create<MulIOp>(
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
    return NonAffineIndexExpr(aa.getRewriter().create<SignedFloorDivIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<SignedFloorDivIOp>(
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
    return NonAffineIndexExpr(aa.getRewriter().create<SignedCeilDivIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<SignedCeilDivIOp>(
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
    return NonAffineIndexExpr(aa.getRewriter().create<SignedRemIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return NonAffineIndexExpr(aa.getRewriter().create<SignedRemIOp>(
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
  Value results = compare.getRewriter().create<SelectOp>(compare.getLoc(),
      compare.getValue(), trueVal.getValue(), falseVal.getValue());
  return NonAffineIndexExpr(results);
}

/*static*/ IndexExpr IndexExpr::min(SmallVectorImpl<IndexExpr> &vals) {
  // Res is already an literal int, we are reducing into it.
  F2Self litFct = [](IndexExpr res, IndexExpr const aa) -> IndexExpr {
    int64_t rrr = res.getLiteral();
    int64_t aaa = aa.getLiteral();
    if (aaa < rrr)
      res.getObj().intLit += aaa;
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            SmallVectorImpl<IndexExpr> &vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
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
    Value compareVal = res.getRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::slt, aa.getValue(), res.getValue());
    Value resVal = aa.getRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsKind(res.getValue(), IndexExprKind::NonAffine);
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
      res.getObj().intLit += aaa;
    return res;
  };
  Flist affineExprFct = [&](IndexExpr res,
                            SmallVectorImpl<IndexExpr> &vvals) -> IndexExpr {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
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
    Value compareVal = res.getRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::sgt, aa.getValue(), res.getValue());
    Value resVal = aa.getRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsKind(res.getValue(), IndexExprKind::NonAffine);
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
  return compareOp(CmpIPredicate::eq, b);
}

IndexExpr IndexExpr::operator==(int64_t const b) const {
  return *this == LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator!=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::ne, b);
}

IndexExpr IndexExpr::operator!=(int64_t const b) const {
  return *this != LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator<=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::sle, b);
}

IndexExpr IndexExpr::operator<=(int64_t const b) const {
  return *this <= LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator<(IndexExpr const b) const {
  return compareOp(CmpIPredicate::slt, b);
}

IndexExpr IndexExpr::operator<(int64_t const b) const {
  return *this < LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator>=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::sge, b);
}

IndexExpr IndexExpr::operator>=(int64_t const b) const {
  return *this >= LiteralIndexExpr(b);
}

IndexExpr IndexExpr::operator>(IndexExpr const b) const {
  return compareOp(CmpIPredicate::sgt, b);
}

IndexExpr IndexExpr::operator>(int64_t const b) const {
  return *this > LiteralIndexExpr(b);
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
// IndexExpr Subclasses for constructing specific IndexExpr kinds.
//===----------------------------------------------------------------------===//

UndefinedIndexExpr::UndefinedIndexExpr() : IndexExpr() {}

LiteralIndexExpr::LiteralIndexExpr(int64_t const value) { init(value); }

LiteralIndexExpr::LiteralIndexExpr(IndexExpr const otherIndexExpr) {
  assert(
      otherIndexExpr.isLiteral() && "cannot make a literal from non literal");
  init(otherIndexExpr.getLiteral());
}

void LiteralIndexExpr::init(int64_t const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Affine);
}

NonAffineIndexExpr::NonAffineIndexExpr(Value const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsKind(value, IndexExprKind::NonAffine);
}

NonAffineIndexExpr::NonAffineIndexExpr(IndexExpr const otherIndexExpr) {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If the index expression is a literal,  just copy it.
  if (otherIndexExpr.isLiteral()) {
    indexExprObj->initAsLiteral(
        otherIndexExpr.getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->copy(otherIndexExpr.getObjPtr());
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make a non-affine from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(
        otherIndexExpr.getValue(), IndexExprKind::NonAffine);
    return;
  }
  case IndexExprKind::Dim: {
    indexExprObj->initAsKind(
        otherIndexExpr.getValue(), IndexExprKind::NonAffine);
    return;
  }
  case IndexExprKind::Symbol: {
    indexExprObj->initAsKind(
        otherIndexExpr.getValue(), IndexExprKind::NonAffine);
    return;
  }
  default:
    break;
  }
  llvm_unreachable("bad path");
}

QuestionmarkIndexExpr::QuestionmarkIndexExpr() {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsQuestionmark();
}

QuestionmarkIndexExpr::QuestionmarkIndexExpr(IndexExpr const otherIndexExpr)
    : QuestionmarkIndexExpr() {
  // Don't care about otherIndexExpr as questionmarks have no real data.
}

PredicateIndexExpr::PredicateIndexExpr(bool const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(value, IndexExprKind::Predicate);
}

PredicateIndexExpr::PredicateIndexExpr(Value const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsKind(value, IndexExprKind::Predicate);
}

PredicateIndexExpr::PredicateIndexExpr(IndexExpr const otherIndexExpr) {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If the index expression is a literal,  just copy it.
  if (otherIndexExpr.isLiteral()) {
    indexExprObj->initAsLiteral(
        otherIndexExpr.getLiteral(), IndexExprKind::Predicate);
    return;
  }
  assert(otherIndexExpr.getKind() == IndexExprKind::Predicate &&
         "can only make a predicate from another predicate");
  indexExprObj->copy(otherIndexExpr.getObjPtr());
}

AffineIndexExpr::AffineIndexExpr(AffineExpr const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsAffineExpr(value);
}

AffineIndexExpr::AffineIndexExpr(IndexExpr const otherIndexExpr) {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If the index expression is a literal,  just copy it.
  if (otherIndexExpr.isLiteral()) {
    indexExprObj->initAsLiteral(
        otherIndexExpr.getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.isInCurrentScope();
  switch (otherIndexExpr.getKind()) {
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
    indexExprObj->copy(otherIndexExpr.getObjPtr());
    return;
  }
  case IndexExprKind::Dim:
  case IndexExprKind::Symbol: {
    assert(isSameScope && "cannot can only import literals, dims and symbols "
                          "from different scopes");
    indexExprObj->initAsAffineExpr(otherIndexExpr.getAffineExpr());
    return;
  }
  default:
    break;
  }
  llvm_unreachable("bad path");
}

DimIndexExpr::DimIndexExpr(Value const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsKind(value, IndexExprKind::Dim);
}

DimIndexExpr::DimIndexExpr(IndexExpr const otherIndexExpr) {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If the index expression is a literal,  just copy it.
  if (otherIndexExpr.isLiteral()) {
    indexExprObj->initAsLiteral(
        otherIndexExpr.getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.isInCurrentScope();
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make an dim from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Dim: {
    // If replicated in the same scope, its not great but will not gen errors.
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Symbol: {
    assert(!isSameScope && "cannot make a dim from a symbol at the same scope");
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Dim);
    return;
  }
  default:
    break;
  }
  llvm_unreachable("bad path");
}

SymbolIndexExpr::SymbolIndexExpr(Value const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsKind(value, IndexExprKind::Symbol);
}

SymbolIndexExpr::SymbolIndexExpr(IndexExpr const otherIndexExpr) {
  // Create new IndexExpr implementation object.
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  // If the index expression is a literal,  just copy it.
  if (otherIndexExpr.isLiteral()) {
    indexExprObj->initAsLiteral(
        otherIndexExpr.getLiteral(), IndexExprKind::Affine);
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.isInCurrentScope();
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    indexExprObj->initAsQuestionmark();
    return;
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Predicate: {
    llvm_unreachable("cannot make an symbol from a predicate");
  }
  case IndexExprKind::Affine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Dim: {
    assert(!isSameScope && "cannot make a symbol from a dim in the same scope");
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Symbol: {
    // If replicated in the same scope, its not great but will not gen errors.
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Symbol);
    return;
  }
  default:
    break;
  }
  llvm_unreachable("bad path");
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions: Array of values
//===----------------------------------------------------------------------===//

ArrayValueIndexCapture::ArrayValueIndexCapture(Operation *op, Value array)
    : op(op), array(array), hasDefault(false) {
  assert(op && "expected an op");
}

ArrayValueIndexCapture::ArrayValueIndexCapture(
    Operation *op, Value array, int64_t defaultLiteral)
    : op(op), array(array), defaultLiteral(defaultLiteral), hasDefault(true) {
  assert(op && "expected an op");
}

IndexExpr ArrayValueIndexCapture::getSymbol(uint64_t i) {
  // Check if we have an operand.
  if (array.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value if there is one.
    if (hasDefault)
      return LiteralIndexExpr(defaultLiteral);
    // Has no default: error
    op->emitError("array value has no values");
    return UndefinedIndexExpr();
  }
  // Check if we have an array of literals.
  if (auto attrArray = getDenseElementAttributeFromValue(array)) {
    // We extracted an dense attribute from definition of operand.
    if (i >= attrArray.getType().getDimSize(0)) {
      // Request beyond available size.
      if (hasDefault)
        return LiteralIndexExpr(defaultLiteral);
      // Has no default: error
      op->emitError("request past array size");
      return UndefinedIndexExpr();
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
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
  Value indexVal = emitConstantOp(scope.getRewriter(), scope.getLoc(),
      scope.getRewriter().getIndexType(), i);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal =
      scope.getRewriter().create<KrnlLoadOp>(scope.getLoc(), array, memrefVal);
  return SymbolIndexExpr(loadVal);
}

void ArrayValueIndexCapture::getSymbolList(
    int num, SmallVectorImpl<IndexExpr> &symbolList) {
  // Clear output.
  symbolList.clear();
  for (int i = 0; i < num; ++i)
    symbolList.emplace_back(getSymbol(i));
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions: Array of values
//===----------------------------------------------------------------------===//

ArrayAttributeIndexCapture::ArrayAttributeIndexCapture(ArrayAttr array)
    : array(array), arraySize((array) ? array.size() : 0), hasDefault(false) {}

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

MemRefBoundIndexCapture::MemRefBoundIndexCapture(Value tensorOrMemref)
    : tensorOrMemref(tensorOrMemref) {
  memRank = tensorOrMemref.getType().cast<ShapedType>().getShape().size();
}

IndexExpr MemRefBoundIndexCapture::getDim(uint64_t i) {
  return get<DimIndexExpr>(i);
}

IndexExpr MemRefBoundIndexCapture::getSymbol(uint64_t i) {
  return get<SymbolIndexExpr>(i);
}

// Assert if not a literal.
IndexExpr MemRefBoundIndexCapture::getLiteral(uint64_t i) {
  assert(i<memRank && "out of bound access");
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  if (shape[i] >= 0) {
    // We have a constant dimension.
    int64_t intVal = shape[i];
    return LiteralIndexExpr(intVal);
  }
  llvm_unreachable("expected a literal");
}

void MemRefBoundIndexCapture::getDimList(SmallVectorImpl<IndexExpr> &dimList) {
  getList<DimIndexExpr>(dimList);
}

void MemRefBoundIndexCapture::getSymbolList(
    SmallVectorImpl<IndexExpr> &symbolList) {
  getList<SymbolIndexExpr>(symbolList);
}

void MemRefBoundIndexCapture::getLiteralList(
    SmallVectorImpl<IndexExpr> &literalList) {
  // Clear output.
  literalList.clear();
  // Scan tensor or memref.
  for (int i = 0; i < memRank; ++i)
    literalList.emplace_back(getLiteral(i));
}

template <class INDEX>
IndexExpr MemRefBoundIndexCapture::get(uint64_t i) {
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  assert(i < memRank && "index out of bound");
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
  Value dynVal =
      scope.getRewriter().create<DimOp>(scope.getLoc(), tensorOrMemref, i);
  return INDEX(dynVal);
}

template <class INDEX>
void MemRefBoundIndexCapture::getList(SmallVectorImpl<IndexExpr> &list) {
  // Clear output.
  list.clear();
  // Scan tensor or memref.
  for (int i = 0; i < memRank; ++i)
    list.emplace_back(get<INDEX>(i));
}

//===----------------------------------------------------------------------===//
// Generating Krnl Load / Store
//===----------------------------------------------------------------------===//

krnl_load::krnl_load(Value memref, SmallVectorImpl<IndexExpr> &indices) {
  IndexExprScope &currScope = IndexExprScope::getCurrentScope();
  SmallVector<Value, 4> loadIndices;
  for (IndexExpr ie : indices)
    loadIndices.emplace_back(ie.getValue());
  result = currScope.getRewriter().create<KrnlLoadOp>(
      currScope.getLoc(), memref, loadIndices);
}

krnl_store::krnl_store(
    Value val, Value memref, SmallVectorImpl<IndexExpr> &indices) {
  IndexExprScope &currScope = IndexExprScope::getCurrentScope();
  SmallVector<Value, 4> storeIndices;
  for (IndexExpr ie : indices)
    storeIndices.emplace_back(ie.getValue());
  currScope.getRewriter().create<KrnlStoreOp>(
      currScope.getLoc(), val, memref, storeIndices);
}
