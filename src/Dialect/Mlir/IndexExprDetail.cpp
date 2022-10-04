/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------IndexExprDetail.hpp - Index expression details---------===
////
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/IndexExprDetail.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include <mutex>

int64_t IndexExpr_gQuestionMarkCounter = -2;

using namespace mlir;

namespace onnx_mlir {

// A lock to protect access to IndexExpr_gQuestionMarkCounter.
std::mutex indexExprQuestionMarkMutex;

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
  const std::lock_guard<std::mutex> lock(indexExprQuestionMarkMutex);
  init(/*isDefined*/ true, /*literal*/ false, IndexExprKind::Questionmark,
      IndexExpr_gQuestionMarkCounter--, AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsLiteral(int64_t const val, const IndexExprKind kind) {
  assert((kind != IndexExprKind::Questionmark) &&
         "literals are either affine or predicate");
  init(/*isDefined*/ true, /*literal*/ true, kind, val, AffineExpr(nullptr),
      Value(nullptr));
}

static bool getIntegerLiteralFromValue(Value value, int64_t &intLit) {
  // From lib/Dialect/LinAlg/Transform/Promotion.cpp
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.getValue().cast<IntegerAttr>().getInt();
    return true;
  }
  // Since ConstantIndexOp is a subclass of ConstantOp, not sure if this one is
  // useful.
  if (auto constantOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.value();
    return true;
  }
  return false;
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
  int64_t valIntLit = 0;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    // Ignore the predicate type as we treat all literal int as untyped.
    initAsLiteral(valIntLit, newKind);
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
      newVal = scope->getRewriter().create<arith::IndexCastOp>(
          scope->getLoc(), scope->getRewriter().getIndexType(), newVal);
    }
  } else if (type.isa<IndexType>()) {
    if (newKind == IndexExprKind::Predicate) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = scope->getRewriter().create<arith::IndexCastOp>(
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
    initAsLiteral(constAffineExpr.getValue(), IndexExprKind::Affine);
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
  if (value == nullptr && !isShapeInferencePass()) {
    // Eagerly create values.
    getValue();
  }
}

void IndexExprImpl::copy(IndexExprImpl const *other) {
  assert(scope && "all index expr must have a defined scope");
  // Preserve this scope, copy the remaining attributes from other.
  init(other->defined, other->literal, other->kind, other->intLit,
      other->affineExpr, other->value);
}

//===----------------------------------------------------------------------===//
// IndexExprExpr queries.
//===----------------------------------------------------------------------===//

bool IndexExprImpl::isDefined() const { return defined; }

bool IndexExprImpl::isLiteral() const {
  assert(isDefined());
  return literal;
}

bool IndexExprImpl::isQuestionmark() const {
  assert(isDefined());
  return kind == IndexExprKind::Questionmark;
}

bool IndexExprImpl::isAffine() const {
  assert(isDefined());
  // To catch predicate that are literals as affine.
  if (isLiteral())
    return true;
  // Note that we do bitvector and to check affine properties.
  return (int)kind & (int)IndexExprKind::Affine;
}

bool IndexExprImpl::isSymbol() const {
  assert(isDefined());
  return kind == IndexExprKind::Symbol;
}

bool IndexExprImpl::isDim() const {
  assert(isDefined());
  return kind == IndexExprKind::Dim;
}

bool IndexExprImpl::isPredType() const {
  assert(isDefined());
  return kind == IndexExprKind::Predicate;
}

bool IndexExprImpl::isIndexType() const { return !isPredType(); }

bool IndexExprImpl::isShapeInferencePass() const {
  assert(hasScope());
  return scope->isShapeInferencePass();
}

bool IndexExprImpl::hasScope() const { return scope != nullptr; }

bool IndexExprImpl::isInCurrentScope() const {
  assert(hasScope());
  return scope->isCurrentScope();
}

bool IndexExprImpl::hasAffineExpr() const {
  assert(isDefined());
  return affineExpr != nullptr;
}

bool IndexExprImpl::hasValue() const {
  assert(isDefined());
  return value != nullptr;
}

//===----------------------------------------------------------------------===//
// IndexExprExpr getters.
//===----------------------------------------------------------------------===//

IndexExprScope &IndexExprImpl::getScope() const {
  assert(hasScope());
  return *scope;
}

IndexExprScope *IndexExprImpl::getScopePtr() const {
  assert(scope && "expected to have scope");
  return scope;
}

IndexExprKind IndexExprImpl::getKind() const { return kind; }

int64_t IndexExprImpl::getLiteral() const {
  assert(isLiteral() && "expected a literal index expression");
  return intLit;
}

int64_t IndexExprImpl::getQuestionmark() const {
  assert(isQuestionmark() && "expected a question mark index expression");
  return intLit;
}

//===----------------------------------------------------------------------===//
// IndexExprExpr transformational getters.
//===----------------------------------------------------------------------===//

AffineExpr IndexExprImpl::getAffineExpr() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  assert(!isPredType() && "no affine support for predicate type");
  if (hasAffineExpr()) {
    // Already computed it, use it.
    return affineExpr;
  }

  assert(isInCurrentScope() &&
         "create an affine expression only for index exprs in current scope");

  if (isLiteral()) {
    // Create a literal.
    affineExpr = getRewriter().getAffineConstantExpr(intLit);
  } else if (isSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(hasValue());
    int id = getScope().addSymbol(value);
    affineExpr = getRewriter().getAffineSymbolExpr(id);
  } else if (isDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(hasValue());
    int id = getScope().addDim(value);
    affineExpr = getRewriter().getAffineDimExpr(id);
  } else {
    llvm_unreachable("requesting affine expr of incompatible IndexExpr");
  }
  return affineExpr;
}

void IndexExprImpl::getAffineMapAndOperands(
    AffineMap &map, SmallVectorImpl<Value> &operands) {
  // Init.
  operands.clear();
  assert(isDefined() && !isQuestionmark() && !isPredType() &&
         "expected lit/affine/non-affine index expr");
  // Handle literal cases.
  if (isLiteral()) {
    map = getRewriter().getConstantAffineMap(intLit);
    return;
  }
  // Handle affine cases.
  if (isAffine()) {
    // Important to get the affine expressions before getting the dims/symbols.
    getAffineExpr();
    map = AffineMap::get(getScope().getNumDims(), getScope().getNumSymbols(),
        {affineExpr}, getRewriter().getContext());
    getScope().getDimAndSymbolList(operands);
    return;
  }
  // Non Affine, check if by any chance we have a min / max, in which case we
  // will extract the correct info.
  if (AffineMinOp affineMinOp = getValue().getDefiningOp<AffineMinOp>()) {
    map = affineMinOp.getAffineMap();
    for (Value val : affineMinOp.getMapOperands())
      operands.emplace_back(val);
    return;
  }
  if (AffineMaxOp affineMaxOp = getValue().getDefiningOp<AffineMaxOp>()) {
    map = affineMaxOp.getAffineMap();
    for (Value val : affineMaxOp.getMapOperands())
      operands.emplace_back(val);
    return;
  }
  // Non affine only known by its value, make a trivial map from it. Hope its ok
  // not to add the symbol in the global scope table, pretty sure it is.
  map = getRewriter().getSymbolIdentityMap();
  operands.emplace_back(getValue());
  return;
}

Value IndexExprImpl::getValue() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");

  // If we already have a value, no need to recompute it as all values must be
  // in the same scope->
  if (hasValue())
    return value;

  assert(isInCurrentScope() &&
         "create a value only for index exprs in current scope");

  if (isLiteral()) {
    // Create a literal constant. Literal pred type should be used directly to
    // eliminate the comparison, so we don't intend to support them here.
    if (isPredType()) {
      bool boolValue = (intLit != 0);
      value = getRewriter().create<arith::ConstantOp>(getLoc(),
          getRewriter().getI1Type(), getRewriter().getBoolAttr(boolValue));
    } else {
      value = getRewriter().create<arith::ConstantIndexOp>(getLoc(), intLit);
    }
  } else if (hasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    assert(!isPredType() && "no affine support for predicate type");
    int dimNum = getScope().getNumDims();
    int symNum = getScope().getNumSymbols();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, getRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    getScope().getDimAndSymbolList(list);
    value = getRewriter().create<AffineApplyOp>(getLoc(), map, list);
  } else {
    llvm_unreachable("bad path");
  }
  return value;
}

} // namespace onnx_mlir
