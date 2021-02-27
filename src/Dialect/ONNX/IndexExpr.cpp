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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

IndexExprScope::IndexExprScope(
    ConversionPatternRewriter *rewriter, Location loc)
    : dims(), symbols(), rewriter(rewriter), loc(loc),
      parentScope(getCurrentScopePtr()), container() {
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope()
    : dims(), symbols(), rewriter(getCurrentScope().rewriter),
      loc(getCurrentScope().loc), parentScope(getCurrentScopePtr()),
      container() {
  getCurrentScopePtr() = this;
}

IndexExprScope::IndexExprScope(IndexExprScope &explicitParentScope)
    : IndexExprScope() {
  assert(&explicitParentScope == parentScope &&
         "provided parent scope was not the previously active scope");
}

IndexExprScope::~IndexExprScope() {
  // Free the memory of each IndexExprImpl in scope's container.
  for (IndexExprImpl *obj : container)
    delete obj;
  container.clear();
  getCurrentScopePtr() = parentScope;
}

/*static*/ IndexExprScope &IndexExprScope::getCurrentScope() {
  IndexExprScope *currScope = getCurrentScopePtr();
  assert(currScope != nullptr && "expected nonnull scope");
  return *currScope;
}

//===----------------------------------------------------------------------===//
// IndexExprScope support for creating krnl load and store ops.
//===----------------------------------------------------------------------===//

Value IndexExprScope::createKrnlLoadOp(
    Value memref, SmallVectorImpl<IndexExpr> &indices) {
  SmallVector<Value, 4> loadIndices;
  for (IndexExpr ie : indices)
    loadIndices.emplace_back(ie.getValue());
  return getRewriter().create<KrnlLoadOp>(getLoc(), memref, loadIndices);
}

void IndexExprScope::createKrnlStoreOp(
    Value val, Value memref, SmallVectorImpl<IndexExpr> &indices) {
  SmallVector<Value, 4> storeIndices;
  for (IndexExpr ie : indices)
    storeIndices.emplace_back(ie.getValue());
  getRewriter().create<KrnlStoreOp>(getLoc(), val, memref, storeIndices);
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

bool IndexExprScope::isParentOfCurrentScope() {
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

ConversionPatternRewriter &IndexExprScope::getRewriter() const {
  assert(rewriter);
  return *rewriter;
}

//===----------------------------------------------------------------------===//
// IndexExprImpl constructors, initializers
//===----------------------------------------------------------------------===//

IndexExprImpl::IndexExprImpl()
    : defined(false), literal(false), kind(IndexExprKind::NonAffine), intLit(0),
      affineExpr(nullptr), value(nullptr) {
  // Set scope from thread private global.
  scope = IndexExprScope::getCurrentScopePtr();
  assert(scope && "expected IndexExpr Scope to be defined");
  // Record the new index expr implementation.
  scope->addIndexExprImpl(this);
}

void IndexExprImpl::initAsUndefined() {
  init(/*isDefined*/ false, /*literal*/ false, IndexExprKind::NonAffine, 0,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsQuestionmark() {
  init(/*isDefined*/ true, /*literal*/ false, IndexExprKind::Questionmark, 0,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsLiteral(int64_t const val) {
  init(/*isDefined*/ true, /*literal*/ true, IndexExprKind::Affine, val,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsKind(Value const val, IndexExprKind const newKind) {
  if (newKind == IndexExprKind::Questionmark) {
    initAsQuestionmark();
    return;
  }
  // Val should exist, because we come here only when passing an actual val, but
  // we might consider checking.
  assert(val != nullptr && "expected a defined value");
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    // Ignore the predicate type as we treat all literal int as untyped.
    initAsLiteral(valIntLit);
    return;
  }
  // We have a value that is not a literal.
  if (scope->isShapeInferencePass()) {
    initAsQuestionmark();
    return;
  }
  // Check that the value is of the right type.
  auto type = val.getType();
  Value newVal = val;
  if (type.isa<IntegerType>()) {
    if (newKind != IndexExprKind::Predicate) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = scope->getRewriter().create<IndexCastOp>(
          scope->getLoc(), scope->getRewriter().getIndexType(), newVal);
    }
  } else if (type.isa<IndexType>()) {
    if (newKind == IndexExprKind::Predicate) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = scope->getRewriter().create<IndexCastOp>(
          scope->getLoc(), scope->getRewriter().getI1Type(), newVal);
    }
  } else {
    llvm_unreachable("unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // getAffineExpr.
  init(/*isDefined*/ true, /*literal*/ false, newKind, 0, AffineExpr(nullptr),
      newVal);
}

void IndexExprImpl::initAsAffineExpr(AffineExpr const val) {
  // Check if the affine expression is reduced to a constant expr.
  AffineExpr simpleVal =
      simplifyAffineExpr(val, scope->getNumDims(), scope->getNumSymbols());
  AffineConstantExpr constAffineExpr = simpleVal.dyn_cast<AffineConstantExpr>();
  if (constAffineExpr) {
    initAsLiteral(constAffineExpr.getValue());
  } else {
    init(/*isDefined*/ true, /*literal*/ false, IndexExprKind::Affine, 0,
        AffineExpr(val), Value(nullptr));
  }
}

void IndexExprImpl::init(bool newIsDefined, bool newIsIntLit,
    IndexExprKind newKind, int64_t const newIntLit,
    AffineExpr const newAffineExpr, Value const newValue) {
  defined = newIsDefined;
  literal = newIsIntLit;
  kind = newKind;
  intLit = newIntLit;
  affineExpr = newAffineExpr;
  value = newValue;
}

void IndexExprImpl::copy(IndexExprImpl const *other) {
  assert(scope && "all index expr must have a defined scope");
  // Preserve this scope, copy the remaining attributes from other.
  init(other->defined, other->literal, other->kind, other->intLit,
      other->affineExpr, other->value);
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
// IndexExpr list queries.
//===----------------------------------------------------------------------===//
bool IndexExpr::isDefined() const {
  assert(!getObj().defined || hasContext());
  return getObj().defined;
}

bool IndexExpr::isUndefined() const {
  // Undefined: its ok to have no impl object associated with it.
  return !indexExprObj || !getObj().defined;
}

bool IndexExpr::isLiteral() const {
  assert(isDefined());
  return getObj().literal;
}

bool IndexExpr::isQuestionmark() const {
  assert(isDefined());
  return getKind() == IndexExprKind::Questionmark;
}

bool IndexExpr::isAffine() const {
  assert(isDefined());
  // Note that we do bitvector and to check affine properties.
  return (int)getKind() & (int)IndexExprKind::Affine;
}

bool IndexExpr::isSymbol() const {
  assert(isDefined());
  return getKind() == IndexExprKind::Symbol;
}

bool IndexExpr::isDim() const {
  assert(isDefined());
  return getKind() == IndexExprKind::Dim;
}

bool IndexExpr::isPredType() const {
  assert(isDefined());
  return getKind() == IndexExprKind::Predicate;
}

bool IndexExpr::isShapeInferencePass() const {
  return getScope().isShapeInferencePass();
}

bool IndexExpr::hasContext() const { return getObj().scope != nullptr; }

bool IndexExpr::hasAffineExpr() const {
  assert(isDefined());
  return !(!getObj().affineExpr);
}

bool IndexExpr::hasValue() const {
  assert(isDefined());
  return !(!getObj().value);
}

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
// IndexExpr Getters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const {
  assert(isLiteral() && "expected a literal index expression");
  return getObj().intLit;
}

AffineExpr IndexExpr::getAffineExpr() const {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  assert(!isPredType() && "no affine support for predicate type");
  if (isLiteral()) {
    // Create a literal.
    getObj().affineExpr = getRewriter().getAffineConstantExpr(getObj().intLit);
  } else if (isSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(hasValue());
    int id = getScope().addSymbol(getObj().value);
    getObj().affineExpr = getRewriter().getAffineSymbolExpr(id);
  } else if (isDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(hasValue());
    int id = getScope().addDim(getObj().value);
    getObj().affineExpr = getRewriter().getAffineDimExpr(id);
  } else {
    assert(
        hasAffineExpr() && "requesting affine expr of incompatible IndexExpr");
  }
  return getObj().affineExpr;
}

Value IndexExpr::getValue() const {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");

  // If we already have a value, no need to recompute it as all values must be
  // in the same scope->
  if (hasValue())
    return getObj().value;

  if (isLiteral()) {
    // Create a literal constant. Literal pred type should be used directly to
    // eliminate the comparison, so we don't intend to support them here.
    assert(!isPredType() && "literal does not support affine expressions");
    getObj().value =
        getRewriter().create<ConstantIndexOp>(getLoc(), getObj().intLit);
  } else if (hasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    assert(!isPredType() && "no affine support for predicate type");
    int dimNum = getScope().getNumDims();
    int symNum = getScope().getNumSymbols();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {getObj().affineExpr}, getRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    getScope().getDimAndSymbolList(list);
    getObj().value = getRewriter().create<AffineApplyOp>(getLoc(), map, list);
  } else {
    llvm_unreachable("bad path");
  }
  return getObj().value;
}

IndexExprScope *IndexExpr::getScopePtr() const {
  assert(hasContext());
  return getObj().scope;
}

ConversionPatternRewriter &IndexExpr::getRewriter() const {
  return getScope().getRewriter();
}

void IndexExpr::debugPrint(const std::string &msg) const {
#if DEBUG
  printf("%s:", msg.c_str());
  if (isLiteral())
    printf(" literal(%lli)", getLiteral());
  if (hasAffineExpr())
    printf(" hasAffine");
  if (hasValue())
    printf(" hasValue");
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

IndexExprImpl &IndexExpr::getObj() const { return *getObjPtr(); }

IndexExprImpl *IndexExpr::getObjPtr() const {
  assert(indexExprObj);
  return indexExprObj;
}

IndexExprKind IndexExpr::getKind() const { return getObj().kind; }

//===----------------------------------------------------------------------===//
// Helpers for IndexExpressions
//===----------------------------------------------------------------------===//

/* static */ void IndexExpr::convertListOfIndexExprToIntegerDim(
    SmallVectorImpl<IndexExpr> &indexExprList,
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
  assert(getScopePtr() == b.getScopePtr() && "incompatible contexts");
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
        return LiteralIndexExpr(1);
      break;
    case CmpIPredicate::ne:
      if (aaa != bbb)
        return LiteralIndexExpr(1);
      break;
    case CmpIPredicate::slt:
      if (aaa < bbb)
        return LiteralIndexExpr(1);
      break;
    case CmpIPredicate::sle:
      if (aaa <= bbb)
        return LiteralIndexExpr(1);
      break;
    case CmpIPredicate::sgt:
      if (aaa > bbb)
        return LiteralIndexExpr(1);
      break;
    case CmpIPredicate::sge:
      if (aaa >= bbb)
        return LiteralIndexExpr(1);
      break;
    default:
      llvm_unreachable("unknown or illegal (unsigned) compare operator");
    }
    return LiteralIndexExpr(0);
  };
  F2 valueFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    Value compare = aa.getRewriter().create<CmpIOp>(
        aa.getLoc(), comparePred, aa.getValue(), bb.getValue());
    return PredicateIndexExpr(compare);
  };
  // Cannot have affine results, disable and pass null lambda function.
  return binaryOp(b, false, false, litFct, nullptr, valueFct);
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
    assert(vals[0].getScopePtr() == vals[i].getScopePtr() &&
           "incompatible contexts");
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

  assert(getScopePtr() == min.getScopePtr() &&
         getScopePtr() == max.getScopePtr() && "incompatible contexts");
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
  assert(compare.getScopePtr() == trueVal.getScopePtr() &&
         compare.getScopePtr() == falseVal.getScopePtr() &&
         "incompatible contexts");
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
  return compareOp(CmpIPredicate::slt, b);
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

LiteralIndexExpr::LiteralIndexExpr(int64_t const value) {
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implemtation");
  indexExprObj->initAsLiteral(value);
}

LiteralIndexExpr::LiteralIndexExpr(IndexExpr const otherIndexExpr) {
  assert(
      otherIndexExpr.isLiteral() && "cannot make a literal from non literal");
  indexExprObj = new IndexExprImpl();
  assert(indexExprObj && "failed to allocate IndexExpr implementation");
  indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
  return;
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
    indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    assert("cannot make a non-affine from a Questionmark");
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->copy(otherIndexExpr.getObjPtr());
    return;
  }
  case IndexExprKind::Predicate: {
    assert("cannot make a non-affine from a predicate");
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
    indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
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
    indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.getScope().isCurrentScope();
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    assert("cannot make an affine from a Questionmark");
  }
  case IndexExprKind::NonAffine: {
    assert("cannot make an affine from an non affine, affine are made of "
           "literals, dims, and symbols");
  }
  case IndexExprKind::Predicate: {
    assert("cannot make an affine from a predicate");
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
    indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.getScope().isCurrentScope();
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    assert("cannot make a dim from a Questionmark");
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Dim);
    return;
  }
  case IndexExprKind::Predicate: {
    assert("cannot make an dim from a predicate");
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
    indexExprObj->initAsLiteral(otherIndexExpr.getLiteral());
    return;
  }
  // Depending on what kind of index expr we got, take different actions.
  bool isSameScope = otherIndexExpr.getScope().isCurrentScope();
  switch (otherIndexExpr.getKind()) {
  case IndexExprKind::Questionmark: {
    assert("cannot make a symbol from a Questionmark");
  }
  case IndexExprKind::NonAffine: {
    indexExprObj->initAsKind(otherIndexExpr.getValue(), IndexExprKind::Symbol);
    return;
  }
  case IndexExprKind::Predicate: {
    assert("cannot make an symbol from a predicate");
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

bool ArrayValueIndexCapture::getSymbolList(
    int num, SmallVectorImpl<IndexExpr> &symbolList) {
  // Clear output.
  symbolList.clear();
  bool successful = true;
  for (int i = 0; i < num; ++i) {
    IndexExpr index = getSymbol(i);
    if (index.isUndefined())
      successful = false;
    symbolList.emplace_back(index);
  }
  return successful;
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions: MemRef Bounds
//===----------------------------------------------------------------------===//

MemRefBoundIndexCapture::MemRefBoundIndexCapture(Value tensorOrMemref)
    : tensorOrMemref(tensorOrMemref) {}

IndexExpr MemRefBoundIndexCapture::getDim(uint64_t i) {
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
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
  return DimIndexExpr(dynVal);
}

bool MemRefBoundIndexCapture::getDimList(SmallVectorImpl<IndexExpr> &dimList) {
  // Clear output.
  dimList.clear();
  // Scan type and shape, bail if incompatible.
  ShapedType type = tensorOrMemref.getType().cast<ShapedType>();
  int size = type.getShape().size();
  // Scan tensor or memref.
  bool successful = true;
  for (int i = 0; i < size; ++i) {
    IndexExpr index = getDim(i);
    if (index.isUndefined())
      successful = false;
    dimList.emplace_back(index);
  }
  return successful;
}
