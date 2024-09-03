/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------IndexExprDetail.hpp - Index expression details---------===
////
//
// Copyright 2020-2024 The IBM Research Authors.
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <mutex>

#define DEBUG_TYPE "index-expr"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexExprImpl constructors, initializers
//===----------------------------------------------------------------------===//

// Default constructor initialize everything to a default value. Remains
// undefined (namely most methods will fails).
IndexExprImpl::IndexExprImpl()
    : defined(false), literal(false), isFloat(false),
      kind(IndexExprKind::NonAffine), intLit(0), affineExpr(nullptr),
      value(nullptr) {
  // Set scope from thread private global.
  scope = IndexExprScope::getCurrentScopePtr();
  assert(scope && "expected IndexExpr Scope to be defined");
  // Record the new index expr implementation.
  scope->addIndexExprImpl(this);
}

void IndexExprImpl::initAsUndefined() {
  init(/*isDefined*/ false, /*literal*/ false, /*isLitFloat*/ false,
      IndexExprKind::NonAffine, 0, AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsQuestionmark(bool isFloatFlag) {
  // Question mark has value of -1 by default.
  init(/*isDefined*/ true, /*literal*/ false, /*isLitFloat*/ isFloatFlag,
      IndexExprKind::Questionmark, ShapedType::kDynamic, AffineExpr(nullptr),
      Value(nullptr));
}

void IndexExprImpl::initAsQuestionmark(int64_t const val, bool isFloatFlag) {
  init(/*isDefined*/ true, /*literal*/ false, /*isLitFloat*/ isFloatFlag,
      IndexExprKind::Questionmark, val, AffineExpr(nullptr), Value(nullptr));
}

// Used for runtime dims; integer by default.
void IndexExprImpl::initAsQuestionmark(Value tensorOrMemref, int64_t index) {
  // Each question mark is assigned a unique integer that is obtained
  // by hashing the tensor/memref value and the target dimension index.
  // According to `llvm/ADT/hashing.h`, a hash value is per execution of the
  // program. Thus, it should not be trusted to be stable or predictable across
  // processes or executions.
  llvm::hash_code questionValue = llvm::hash_combine(
      mlir::hash_value(tensorOrMemref), llvm::hash_value(index));
  init(/*isDefined*/ true, /*literal*/ false,
      /*isLitFloat, as this is for shapes*/ false, IndexExprKind::Questionmark,
      questionValue, AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsLiteral(int64_t const val, const IndexExprKind kind) {
  assert((kind != IndexExprKind::Questionmark) &&
         "literals are either affine or predicate");
  init(/*isDefined*/ true, /*literal*/ true, /*isLitFloat*/ false, kind, val,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsLiteral(double const val, const IndexExprKind kind) {
  assert((kind != IndexExprKind::Questionmark) &&
         "literals are not question marks");
  // Use union to get the bit pattern of the double into an int without changes.
  union {
    int64_t ival;
    double fval;
  } intOrFloatVal;
  // Define using float value.
  intOrFloatVal.fval = val;
  // Use using int value, so that we don't have to create two init functions.
  init(/*isDefined*/ true, /*literal*/ true, /*isLitFloat*/ true, kind,
      intOrFloatVal.ival, AffineExpr(nullptr), Value(nullptr));
}

static bool getIntegerLiteralFromValue(Value value, int64_t &intLit) {
  // From lib/Dialect/LinAlg/Transform/Promotion.cpp
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (mlir::isa<IndexType>(constantOp.getType()))
      intLit = mlir::cast<IntegerAttr>(constantOp.getValue()).getInt();
    return true;
  }
  // Since ConstantIndexOp is a subclass of ConstantOp, not sure if this one is
  // needed.
  if (auto constantOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    if (mlir::isa<IndexType>(constantOp.getType()))
      intLit = constantOp.value();
    return true;
  }
  return false;
}

static bool getFloatLiteralFromValue(Value value, double &floatLit) {
  // From lib/Dialect/LinAlg/Transform/Promotion.cpp
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (mlir::isa<FloatType>(constantOp.getType()))
      floatLit =
          mlir::cast<FloatAttr>(constantOp.getValue()).getValueAsDouble();
    return true;
  }
  // Since ConstantFloatOp is a subclass of ConstantOp, not sure if this one is
  // needed.
  if (auto constantOp = value.getDefiningOp<arith::ConstantFloatOp>()) {
    if (mlir::isa<FloatType>(constantOp.getType()))
      floatLit = constantOp.value().convertToDouble();
    return true;
  }
  return false;
}

void IndexExprImpl::initAsKind(Value const val, IndexExprKind const newKind) {
  // Val should exist, because we come here only when passing an actual val, but
  // we might consider checking.
  assert(val != nullptr && "expected a defined value");
  // Check that the value is of the right type.
  auto type = val.getType();
  bool valIsFloat = (mlir::isa<FloatType>(type));
  // Questionmark
  if (newKind == IndexExprKind::Questionmark) {
    initAsQuestionmark(valIsFloat);
    return;
  }

  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit = 0;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    // Ignore the predicate type as we treat all literal int as untyped.
    initAsLiteral(valIntLit, newKind);
    return;
  }
  double valFloatLit = 0.0;
  if (getFloatLiteralFromValue(val, valFloatLit)) {
    // We have an float constant.
    initAsLiteral(valFloatLit, newKind);
    return;
  }
  // We have a value that is not a literal.
  if (scope->isShapeInferencePass()) {
    initAsQuestionmark(valIsFloat);
    return;
  }
  Value newVal = val;
  if (mlir::isa<IntegerType>(type)) {
    if (newKind != IndexExprKind::Predicate) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = scope->getRewriter().create<arith::IndexCastOp>(
          scope->getLoc(), scope->getRewriter().getIndexType(), newVal);
    }
  } else if (mlir::isa<IndexType>(type)) {
    if (newKind == IndexExprKind::Predicate) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = scope->getRewriter().create<arith::IndexCastOp>(
          scope->getLoc(), scope->getRewriter().getI1Type(), newVal);
    }
  } else if (mlir::isa<FloatType>(type)) {
    assert(newKind != IndexExprKind::Predicate && "float cannot be predicate");
    // Assume its a single precision float.
    unsigned width = mlir::cast<FloatType>(type).getWidth();
    assert(width == 32 && "Index expression only support f32 at this time");
  } else {
    llvm_unreachable("unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // getAffineExpr.
  // Note: init will determine if newVal's type is int or float and thus set the
  // isFloat flag to the right value.
  init(/*isDefined*/ true, /*literal*/ false, /*isLitFloat*/ false, newKind, 0,
      AffineExpr(nullptr), newVal);
}

void IndexExprImpl::initAsAffineExpr(AffineExpr const val) {
  // Check if the affine expression is reduced to a constant expr.
  AffineExpr simpleVal =
      simplifyAffineExpr(val, scope->getNumDims(), scope->getNumSymbols());
  AffineConstantExpr constAffineExpr =
      llvm::dyn_cast<AffineConstantExpr>(simpleVal);
  if (constAffineExpr) {
    initAsLiteral(constAffineExpr.getValue(), IndexExprKind::Affine);
  } else {
    init(/*isDefined*/ true, /*literal*/ false, /*isLitFloat*/ false,
        IndexExprKind::Affine, 0, AffineExpr(val), Value(nullptr));
  }
}

void IndexExprImpl::init(bool newIsDefined, bool newIsIntLit, bool isFloatLit,
    IndexExprKind newKind, int64_t const newIntOrFloatLit,
    AffineExpr const newAffineExpr, Value const newValue) {
  defined = newIsDefined;
  literal = newIsIntLit;
  kind = newKind;
  intLit = newIntOrFloatLit;
  isFloat = isFloatLit;
  affineExpr = newAffineExpr;
  value = newValue;
  if (value != nullptr) {
    // We have a value initialized index expr. Determine if we have an integer
    // or float expression.
    if (mlir::isa<FloatType>(value.getType())) {
      // Assume its a single precision float.
      unsigned width = mlir::cast<FloatType>(value.getType()).getWidth();
      assert(width == 32 && "Index expression only support f32 at this time");
      isFloat = true;
    }
  } else if (!isShapeInferencePass()) {
    // Eagerly create values.
    getValue();
  }
}

void IndexExprImpl::copy(IndexExprImpl const *other) {
  assert(scope && "all index expr must have a defined scope");
  // Preserve this scope, copy the remaining attributes from other.
  init(other->defined, other->literal, other->isFloat, other->kind,
      other->intLit, other->affineExpr, other->value);
}

//===----------------------------------------------------------------------===//
// IndexExprExpr queries.
//===----------------------------------------------------------------------===//

bool IndexExprImpl::isDefined() const { return defined; }

bool IndexExprImpl::isLiteral() const {
  assert(isDefined() && "index expression must be defined");
  return literal;
}

bool IndexExprImpl::isQuestionmark() const {
  assert(isDefined() && "index expression must be defined");
  return kind == IndexExprKind::Questionmark;
}

bool IndexExprImpl::isAffine() const {
  assert(isDefined() && "index expression must be defined");
  // To catch predicate that are literals as affine.
  if (isLiteral())
    return true;
  // Note that we do bitvector and to check affine properties.
  return (int)kind & (int)IndexExprKind::Affine;
}

bool IndexExprImpl::isSymbol() const {
  assert(isDefined() && "index expression must be defined");
  return kind == IndexExprKind::Symbol;
}

bool IndexExprImpl::isDim() const {
  assert(isDefined() && "index expression must be defined");
  return kind == IndexExprKind::Dim;
}

bool IndexExprImpl::isPredType() const {
  assert(isDefined() && "index expression must be defined");
  return kind == IndexExprKind::Predicate;
}

bool IndexExprImpl::isFloatType() const {
  assert(isDefined() && "index expression must be defined");
  return isFloat;
}

bool IndexExprImpl::isIndexType() const {
  return !isPredType() && !isFloatType();
}

bool IndexExprImpl::isShapeInferencePass() const {
  assert(hasScope());
  return scope->isShapeInferencePass();
}

bool IndexExprImpl::hasScope() const { return scope != nullptr; }

bool IndexExprImpl::isInCurrentScope() const {
  assert(hasScope());
  bool inScope = scope->isCurrentScope();
#if DETAILED_DEBUG_OF_SCOPE
  LLVM_DEBUG({
    if (!inScope)
      llvm::dbgs() << "IES: NOT IN SCOPE, IE " << ((long long)scope)
                   << " != curr "
                   << ((long long)IndexExprScope::getCurrentScopePtr()) << "\n";
    else
      llvm::dbgs() << "IES: in scope, IE " << ((long long)scope) << " == curr "
                   << ((long long)IndexExprScope::getCurrentScopePtr()) << "\n";
  });
#endif
  return inScope;
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
  assert(!isFloatType() && "expected integer literal");
  return intLit;
}

double IndexExprImpl::getFloatLiteral() const {
  assert(isLiteral() && "expected a literal index expression");
  assert(isFloatType() && "expected float literal");
  return floatLit;
}

int64_t IndexExprImpl::getQuestionmark() const {
  assert(isQuestionmark() && "expected a question mark index expression");
  // Question mark ID is valid for integer or float values, no need for asserts.
  return intLit;
}

//===----------------------------------------------------------------------===//
// IndexExprExpr transformational getters.
//===----------------------------------------------------------------------===//

AffineExpr IndexExprImpl::getAffineExpr() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  assert(!isPredType() && "no affine support for predicate type");
  assert(!isFloatType() && "affine expression not available for float type");
  if (hasAffineExpr()) {
    // Already computed it, use it.
    return affineExpr;
  }

  // Literal never have to be in scope, so bypass in scope test when that is the
  // case.
  assert((isLiteral() || isInCurrentScope()) &&
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
  assert(isDefined() && !isQuestionmark() && !isPredType() &&
         "expected lit/affine/non-affine index expr");
  assert(!isFloatType() && "affine expression not available for float");
  // Init.
  operands.clear();
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
  if (auto affineMinOp = getValue().getDefiningOp<affine::AffineMinOp>()) {
    map = affineMinOp.getAffineMap();
    // Wonder if specialized list is better than all dims and syms
    // (scope.getDimAndSymbolList(operands)).
    for (Value val : affineMinOp.getMapOperands())
      operands.emplace_back(val);
    return;
  }
  if (auto affineMaxOp = getValue().getDefiningOp<affine::AffineMaxOp>()) {
    map = affineMaxOp.getAffineMap();
    // Wonder if specialized list is better than all dims and syms
    // (scope.getDimAndSymbolList(operands)).
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
      assert(!isFloatType() && "predicate literal not available for float");
      bool boolValue = (intLit != 0);
      value = getRewriter().create<arith::ConstantOp>(getLoc(),
          getRewriter().getI1Type(), getRewriter().getBoolAttr(boolValue));
    } else if (isFloatType()) {
      // Treat float types as f32 as this is what we currently have in the ONNX
      // specs involving float and index calculations.
      float fval = floatLit;
      value = getRewriter().create<arith::ConstantFloatOp>(
          getLoc(), llvm::APFloat(fval), getRewriter().getF32Type());
    } else {
      value = getRewriter().create<arith::ConstantIndexOp>(getLoc(), intLit);
    }
  } else if (hasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    assert(!isPredType() && "no affine support for predicate type");
    assert(!isFloatType() && "no affine support for float type");
    int dimNum = getScope().getNumDims();
    int symNum = getScope().getNumSymbols();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, getRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    getScope().getDimAndSymbolList(list);
    value = getRewriter().create<affine::AffineApplyOp>(getLoc(), map, list);
  } else if (isQuestionmark()) {
    // There are cases where shape inference cannot determine the size even at
    // runtime before running some specialized computations. For example,
    // compress need to scan the vector of condition at runtime to determine the
    // actual number of output values. Thus we are ok with letting QuestionMarks
    // flow in such situations.
    // Note that this index expression / shape will have to be updated in some
    // ways before allocating an array based on this shape. This will be the
    // responsibility of the lowering pass.
    return nullptr;
  } else {
    llvm_unreachable("bad path");
  }
  return value;
}

//===----------------------------------------------------------------------===//
// IndexExprExpr setters.
//===----------------------------------------------------------------------===//

void IndexExprImpl::setLiteral(int64_t val) {
  assert(isLiteral() && "set literal allowed only for literal index expr ");
  assert(!isFloatType() && "cannot set int value for float index expr");
  intLit = val;
}

void IndexExprImpl::setLiteral(double val) {
  assert(isLiteral() && "set literal allowed only for literal index expr ");
  assert(isFloatType() && "cannot set float value for int index expr");
  floatLit = val;
}

void IndexExprImpl::setLiteral(const IndexExprImpl &obj) {
  assert(isLiteral() && "set literal allowed only for literal index expr ");
  if (isFloatType())
    setLiteral(obj.getFloatLiteral());
  else
    setLiteral(obj.getLiteral());
}

void IndexExprImpl::debugPrint(const std::string &msg) {
  LLVM_DEBUG({
    llvm::dbgs() << msg.c_str();
    if (!isDefined()) {
      llvm::dbgs() << " undefined\n";
      return;
    }
    if (isLiteral()) {
      if (isFloatType())
        llvm::dbgs() << " floatLiteral(" << getFloatLiteral() << ")";
      else
        llvm::dbgs() << " literal(" << (long long)getLiteral() << ")";
    }
    if (isFloatType())
      llvm::dbgs() << " isFloat";
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

} // namespace onnx_mlir
