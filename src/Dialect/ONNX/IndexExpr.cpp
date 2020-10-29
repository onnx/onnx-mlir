//===----------------IndexExpr.cpp - Index expression---------------------=== //
//
// copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calcualtions using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

// both debug variables will be removed once debugging is complete.
#define DEBUG 1
#define CEIL_FLOOR_IN_STD 1

#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// IndexExprContext constructors.
//===----------------------------------------------------------------------===//

IndexExprContext::IndexExprContext(
    ConversionPatternRewriter *rewriter, Location loc)
    : rewriter(rewriter), loc(loc), dims(), symbols(), parentContext(nullptr) {}

IndexExprContext::IndexExprContext(IndexExprContext &newParentContext)
    : rewriter(newParentContext.rewriter), loc(newParentContext.loc), dims(),
      symbols(), parentContext(nullptr) {
  // We resue the parent context, and in particuliar its affine
  // functions. Now because the affine functions of the parent context have
  // "ids" embedded in the AffineExpr, we must reuse the same mix of Dims and
  // Symbols here. I don't believe there is any sideeffects in considering a Dim
  // from the parent's context as a Dim in the child's context, even though the
  // parent's dim is supposed to be constant in the child's context.
  for (Value parentDim : newParentContext.dims)
    addDim(parentDim);
  for (Value parentSymbol : newParentContext.symbols)
    addSymbol(parentSymbol);
  // Save reference to parent context so that we may detect the reuse.
  parentContext = &newParentContext;
}

//===----------------------------------------------------------------------===//
// IndexExprContext builder for IndexExpr.
//===----------------------------------------------------------------------===//

IndexExpr IndexExprContext::createUndefinedIndex() {
  IndexExpr res;
  return res.initAsUndefined();
}

IndexExpr IndexExprContext::createQuestionmarkIndex() {
  IndexExpr res;
  return res.initAsQuestionmark(*this);
}

IndexExpr IndexExprContext::createLiteralIndex(int64_t val) {
  IndexExpr res;
  return res.initAsLiteral(*this, val);
}

IndexExpr IndexExprContext::createDimIndex(Value val) {
  IndexExpr res;
  return res.initAsDim(*this, val);
}

IndexExpr IndexExprContext::createDimIndexFromMemref(
    Value memref, ArrayRef<int64_t> memrefShape, int index) {
  IndexExpr res;
  return res.initAsDimFromMemref(*this, memref, memrefShape, index);
}

IndexExpr IndexExprContext::createSymbolIndexFromArrayAtIndex(
    Operation *op, Value arrayOperand, uint64_t index) {
  IndexExpr res;
  return res.initAsSymbolFromArrayAtIndex(*this, op, arrayOperand, index);
}

IndexExpr IndexExprContext::createSymbolIndexFromArrayAtIndex(
    Operation *op, Value arrayOperand, uint64_t index, int64_t defaultLiteral) {
  IndexExpr res;
  return res.initAsSymbolFromArrayAtIndex(
      *this, op, arrayOperand, index, defaultLiteral);
}

IndexExpr IndexExprContext::createSymbolIndex(Value val) {
  IndexExpr res;
  return res.initAsSymbol(*this, val);
}

// Additional builder for repurposing IndexExpr from parent context.
IndexExpr IndexExprContext::createSymbolIndexFromParentContext(
    IndexExpr &parentIndexExpr) {
  // Make sure that we are using the propper parent context
  assert(parentIndexExpr.getContextPtr() == parentContext &&
         "parent index is not from the parent's context");
  // When the parent expression is already affine in the outer context, it will
  // remain afine in the child's context as wee. So we keep it as such, to get
  // as exprssive affine expressions as possible. We could retrict reuse for
  // literal only.
  if (parentIndexExpr.isAffine()) {
    // Reuse affine expression.
    parentIndexExpr.debugPrint("Reuse parent");
    IndexExpr childIndexExpr(parentIndexExpr);
    childIndexExpr.setContext(*this);
    return childIndexExpr;
  }
  // Non affine, create a symbol.
  parentIndexExpr.debugPrint("Create symbol out of parent");
  return createSymbolIndex(parentIndexExpr.getValue());
}

//===----------------------------------------------------------------------===//
// IndexExprContext support for dim and symbol lists in affine exprs.
//===----------------------------------------------------------------------===//

int IndexExprContext::addDim(Value value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprContext::addSymbol(Value value) {
  symbols.emplace_back(value);
  return symbols.size() - 1;
}

//===----------------------------------------------------------------------===//
// IndexExprContext getters.
//===----------------------------------------------------------------------===//

void IndexExprContext::getDimAndSymbolList(SmallVectorImpl<Value> &list) const {
  list.clear();
  for (auto dim : dims)
    list.emplace_back(dim);
  for (auto sym : symbols)
    list.emplace_back(sym);
}

ConversionPatternRewriter &IndexExprContext::GetRewriter() const {
  assert(rewriter);
  return *rewriter;
}

//===----------------------------------------------------------------------===//
// IndexExprContext static helper functions.
//===----------------------------------------------------------------------===//

bool IndexExprContext::areAllLiteral(SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.isLiteral())
      return false;
  }
  return true;
}

bool IndexExprContext::areAllAffine(SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.isAffine())
      return false;
  }
  return true;
}

void IndexExprContext::getOutputDimsForType(
    SmallVectorImpl<IndexExpr> &outputIndices,
    SmallVectorImpl<int64_t> &outputDims) {
  outputDims.clear();
  for (IndexExpr &outputIndex : outputIndices) {
    if (outputIndex.isLiteral())
      outputDims.emplace_back(outputIndex.getLiteral());
    else
      outputDims.emplace_back(-1);
  }
}

//===----------------------------------------------------------------------===//
// IndexExpr constructors, initializers
//===----------------------------------------------------------------------===//

IndexExpr::IndexExpr()
    : defined(false), litteral(false), affine(false), symbol(false), dim(false),
      intLit(0), affineExpr(nullptr), value(nullptr), context(nullptr) {}

IndexExpr &IndexExpr::initAsUndefined() {
  return init(/*context*/ nullptr, /*isDefined*/ false, /*litteral*/ false,
      /*affine*/ false, /*symbol*/ false, /*dim*/ false, 0, AffineExpr(nullptr),
      Value(nullptr));
}

IndexExpr &IndexExpr::initAsQuestionmark(IndexExprContext &newContext) {
  return init(&newContext, /*isDefined*/ true, /*litteral*/ false,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, 0, AffineExpr(nullptr),
      Value(nullptr));
}

IndexExpr &IndexExpr::initAsLiteral(IndexExprContext &newContext, int64_t val) {
  return init(&newContext, /*isDefined*/ true, /*litteral*/ true,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, val,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExpr &IndexExpr::initAsDim(IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(
      newContext, val, /*affine*/ true, /*symbol*/ false, /*dim*/ true);
}

IndexExpr &IndexExpr::initAsSymbol(IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(
      newContext, val, /*affine*/ true, /*symbol*/ true, /*dim*/ false);
}

IndexExpr &IndexExpr::initAsValue(IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ false,
      /*symbol*/ false, /*dim*/ false);
}

IndexExpr &IndexExpr::initAsAffineExpr(
    IndexExprContext &newContext, AffineExpr val) {
  // Check if the affine expression is reduced to a constant expr.
  AffineExpr simpleVal = simplifyAffineExpr(
      val, newContext.getNumDims(), newContext.getNumSymbols());
  AffineConstantExpr constAffineExpr = simpleVal.dyn_cast<AffineConstantExpr>();
  if (constAffineExpr) {
    return initAsLiteral(newContext, constAffineExpr.getValue());
  }
  return init(&newContext, /*isDefined*/ true, /*litteral*/ false,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, 0, AffineExpr(val),
      Value(nullptr));
}

IndexExpr &IndexExpr::init(IndexExprContext *newContext, bool newIsDefined,
    bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
    int newIntLit, AffineExpr newAffineExpr, Value newValue) {
  context = newContext;
  defined = newIsDefined;
  litteral = newIsIntLit;
  affine = newIsAffine;
  symbol = newIsSymbol;
  dim = newIsDim;
  intLit = newIntLit;
  affineExpr = newAffineExpr;
  value = newValue;
  return *this;
}

IndexExpr &IndexExpr::initAsLitQuestionmarkOrValue(IndexExprContext &newContext,
    Value val, bool newIsAfine, bool newIsSymbol, bool newIsDim) {
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    return initAsLiteral(newContext, valIntLit);
  }
  // We have a value that is not a literal.
  if (newContext.isShapeInferencePass()) {
    return initAsQuestionmark(newContext);
  }
  // Check that the value is of the right type.
  auto type = val.getType();
  if (type.isa<IntegerType>()) {
    // We need to convert the int into an index, since we are dealing with index
    // expressions.
    val = newContext.GetRewriter().create<IndexCastOp>(
        newContext.GetLocation(), newContext.GetRewriter().getIndexType(), val);
  } else {
    assert(type.isa<IndexType>() && "unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // getAffineExpr.
  assert(!(newIsDim && newIsSymbol) &&
         "cannot have dim and symbol at the same time");
  return init(&newContext, /*isDefined*/ true, /*litteral*/ false,
      /*affine*/ newIsAfine, /*symbol*/ newIsSymbol, /*dim*/ newIsDim, 0,
      AffineExpr(nullptr), val);
}

//===----------------------------------------------------------------------===//
// IndexExpr initializers that extract info
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::initAsDimFromMemref(IndexExprContext &newContext,
    Value memref, ArrayRef<int64_t> memrefShape, int index) {
  if (memrefShape[index] >= 0) {
    // We have a constant dimension.
    int64_t intVal = memrefShape[index];
    return initAsLiteral(newContext, intVal);
  }
  // We have a dynamic dimension.
  if (newContext.isShapeInferencePass()) {
    return initAsQuestionmark(newContext);
  }
  Value dynVal = newContext.GetRewriter().create<DimOp>(
      newContext.GetLocation(), memref, index);
  return initAsDim(newContext, dynVal);
}

IndexExpr &IndexExpr::initAsSymbolFromArrayAtIndex(IndexExprContext &newContext,
    Operation *op, Value arrayOperand, uint64_t i) {
  if (auto attrArray = getDenseElementAttributeFromValue(arrayOperand)) {
    // We extracted an dense attribute from definition of operand.
    if (i >= attrArray.getType().getDimSize(0)) {
      printf("error 1\n");
      op->emitError("operand literal has wrong shape");
      return initAsUndefined();
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return initAsLiteral(newContext, attrInt);
  }
  // We must read value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return initAsQuestionmark(newContext);
  }
  // Emit code to read array.
  Value indexVal = emitConstantOp(newContext.GetRewriter(),
      newContext.GetLocation(), newContext.GetRewriter().getIndexType(), i);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.GetRewriter().create<AffineLoadOp>(
      newContext.GetLocation(), arrayOperand, memrefVal);
  return initAsSymbol(newContext, loadVal);
}

IndexExpr &IndexExpr::initAsSymbolFromArrayAtIndex(IndexExprContext &newContext,
    Operation *op, Value arrayOperand, uint64_t i, int64_t defaultLiteral) {
  // Check if we have an operand.
  if (arrayOperand.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value.
    return initAsLiteral(newContext, defaultLiteral);
  }
  if (auto attrArray = getDenseElementAttributeFromValue(arrayOperand)) {
    // We extracted an dense attribute from definition of operand.
    if (i > attrArray.getType().getDimSize(0)) {
      // Not enought attributes for this index, return the default value.
      return initAsLiteral(newContext, defaultLiteral);
    }
    // We have enought attributes for this index, get the value.
    Attribute attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return initAsLiteral(newContext, attrInt);
  }
  // Read the value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return initAsQuestionmark(newContext);
  }
  // Emit the code to read array.
  Value indexVal = emitConstantOp(newContext.GetRewriter(),
      newContext.GetLocation(), newContext.GetRewriter().getIndexType(), i);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.GetRewriter().create<AffineLoadOp>(
      newContext.GetLocation(), arrayOperand, memrefVal);
  return initAsSymbol(newContext, loadVal);
}

//===----------------------------------------------------------------------===//
// IndexExpr copy and setters.
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::copy(IndexExpr &a) {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. copy info in the meanwhile.
  *this = a;
  return *this;
}

void IndexExpr::setContext(IndexExprContext &newContext) {
  context = &newContext;
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

bool IndexExpr::isDefined() const {
  assert(!defined || hasContext());
  return defined;
}

bool IndexExpr::isLiteral() const {
  assert(isDefined());
  return litteral;
}

bool IndexExpr::isQuestionmark() const {
  assert(isDefined());
  return !isLiteral();
}

bool IndexExpr::isAffine() const {
  assert(isDefined());
  return affine;
}

bool IndexExpr::isSymbol() const {
  assert(isDefined());
  return symbol;
}

bool IndexExpr::isDim() const {
  assert(isDefined());
  return dim;
}

bool IndexExpr::isShapeInferencePass() const {
  assert(hasContext());
  return context->isShapeInferencePass();
}

bool IndexExpr::hasContext() const { return context != nullptr; }

bool IndexExpr::hasAffineExpr() const {
  assert(isDefined());
  return !(!affineExpr);
}

bool IndexExpr::hasValue() const {
  assert(isDefined());
  return !(!value);
}

//===----------------------------------------------------------------------===//
// IndexExpr Getters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const {
  assert(isLiteral());
  return intLit;
}

AffineExpr IndexExpr::getAffineExpr() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  if (isLiteral()) {
    // Create a literal.
    affineExpr = context->GetRewriter().getAffineConstantExpr(intLit);
  } else if (isSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(hasValue());
    int id = context->addSymbol(value);
    affineExpr = context->GetRewriter().getAffineSymbolExpr(id);
  } else if (isDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(hasValue());
    int id = context->addDim(value);
    affineExpr = context->GetRewriter().getAffineDimExpr(id);
  } else {
    assert(
        hasAffineExpr() && "requesting affine expr of incompatible IndexExpr");
  }
  return affineExpr;
}

Value IndexExpr::getValue() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  if (isLiteral()) {
    // Create a litteral constant.
    value = context->GetRewriter().create<ConstantIndexOp>(
        context->GetLocation(), intLit);
  } else if (hasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = context->getNumDims();
    int symNum = context->getNumSymbols();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {affineExpr}, context->GetRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    context->getDimAndSymbolList(list);
    value = context->GetRewriter().create<AffineApplyOp>(
        context->GetLocation(), map, list);
  } else {
    assert(hasValue());
  }
  return value;
}

IndexExprContext &IndexExpr::getContext() const {
  assert(hasContext());
  return *context;
}

IndexExprContext *IndexExpr::getContextPtr() const {
  assert(hasContext());
  return context;
}

Location IndexExpr::getLoc() const { return getContext().GetLocation(); }

void IndexExpr::debugPrint(const std::string &msg) {
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
  printf(" context(0x%llx)\n", (long long unsigned)context);
#endif
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Used for add/sub/mult/ceilDiv/floorDiv
IndexExpr &IndexExpr::BinaryOp(IndexExpr &a, IndexExpr &b, bool affineWithLitB,
    bool canBeAffine, F2 litFct, F2 affineExprFct, F2 valueFct) {
  assert(a.getContextPtr() == b.getContextPtr() && "incompatible contexts");
  // Literal integer if a and b are literals. Affine if canBeAffine is true,
  // both a and b are affine, and possibly a and/or b are also constant.
  bool resIsLit = a.isLiteral() && b.isLiteral();
  bool resIsAffine = resIsLit || (canBeAffine && a.isAffine() && b.isAffine() &&
                                     (!affineWithLitB || b.isLiteral()));

  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit) {
    // Constant, use constant computations.
    litFct(*this, a, b);
  } else if (a.isShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    initAsQuestionmark(a.getContext());
  } else if (resIsAffine) {
    // Use affine values.
    affineExprFct(*this, a, b);
  } else {
    // Use values.
    valueFct(*this, a, b);
  }
  return *this;
}

// Used for clamp.
IndexExpr &IndexExpr::TernaryOp(
    IndexExpr &a, IndexExpr &b, IndexExpr &c, F3 litFct, F3 valueFct) {
  assert(a.getContextPtr() == b.getContextPtr() &&
         a.getContextPtr() == c.getContextPtr() && "incompatible contexts");
  // Literal integer if a, b, and c are literals. Output is not affine (unless
  // all 3 are literals).
  bool resIsLit = a.isLiteral() && b.isLiteral() && c.isLiteral();
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit) {
    // Constant, use constant computations.
    litFct(*this, a, b, c);
  } else if (a.isShapeInferencePass()) {
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    initAsQuestionmark(a.getContext());
  } else {
    // Use values.
    valueFct(*this, a, b, c);
  }
  return *this;
}

IndexExpr &IndexExpr::QuaternarySelectOp(IndexExpr &compA, IndexExpr &compB,
    IndexExpr &trueVal, IndexExpr &falseVal, F4 litFct, F4 valueFct) {
  assert(compA.getContextPtr() == compB.getContextPtr() &&
         compA.getContextPtr() == trueVal.getContextPtr() &&
         compA.getContextPtr() == falseVal.getContextPtr() &&
         "incompatible contexts");
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (compA.isLiteral() && compB.isLiteral()) {
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    litFct(*this, compA, compB, trueVal, falseVal);
  } else if (compA.isShapeInferencePass()) {
    // Just set as undefined
    initAsQuestionmark(compA.getContext());
  } else {
    // We cannot represent this as an affine expression, so go directly
    // to values.
    valueFct(*this, compA, compB, trueVal, falseVal);
  }
  return *this;
}

// The affine reduction labda function processes the whole list and must init
// the result.
IndexExpr &IndexExpr::reductionOp(
    SmallVectorImpl<IndexExpr> &vals, F2 litRed, Flist affineRed, F2 valueRed) {
  // If no values, result is undefined.
  int size = vals.size();
  if (size == 0) {
    initAsUndefined();
    return *this;
  }
  // Set the output to the first value.
  copy(vals[0]);
  // If list has one element, we are done. Literal/Affine... will be the same as
  // this single element.
  if (vals.size() == 1)
    return *this;
  // Have multiple values, need to do some checks.
  bool resIsLit = true;
  bool resIsAffine = true;
  for (int i = 0; i < size; ++i) {
    if (!vals[i].isLiteral())
      resIsLit = false;
    if (!vals[i].isAffine())
      resIsAffine = false;
    assert(vals[0].getContextPtr() == vals[i].getContextPtr() &&
           "incompatible contexts");
  }
  if (resIsLit) {
    // Process int literals, if we only have literal values.
    // Result was set to first element, which by default is literal/affine. This
    // will be the correct result for the output.
    for (int i = 1; i < size; ++i) {
      litRed(*this, vals[i], *this);
    }
  } else if (vals[0].isShapeInferencePass()) {
    // Just set as undefined
    initAsQuestionmark(vals[0].getContext());
  } else if (resIsAffine) {
    affineRed(*this, vals);
  } else {
    for (int i = 1; i < size; ++i) {
      valueRed(*this, vals[i], *this);
    }
  }
  return *this;
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops.
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::add(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsLiteral(aa.getContext(), aa.getLiteral() + bb.getLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsAffineExpr(
        aa.getContext(), aa.getAffineExpr() + bb.getAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsValue(
        aa.getContext(), aa.getContext().GetRewriter().create<AddIOp>(
                             aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return BinaryOp(a, b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::sub(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsLiteral(aa.getContext(), aa.getLiteral() - bb.getLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsAffineExpr(
        aa.getContext(), aa.getAffineExpr() - bb.getAffineExpr());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsValue(
        aa.getContext(), aa.getContext().GetRewriter().create<SubIOp>(
                             aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return BinaryOp(a, b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::mult(IndexExpr &a, IndexExpr &b) {
  // In the lambda function below, if one is literal, it is assumed that it is
  // in the second position (b).
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsLiteral(aa.getContext(), aa.getLiteral() * bb.getLiteral());
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand aa must be a literal.
    res.initAsAffineExpr(aa.getContext(), aa.getAffineExpr() * bb.getLiteral());
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      res.copy(aa);
    } else {
      res.initAsValue(
          aa.getContext(), aa.getContext().GetRewriter().create<MulIOp>(
                               aa.getLoc(), aa.getValue(), bb.getValue()));
    }
  };
  // Literal should be place in second argument; do so if a is a lit.
  if (a.isLiteral())
    return BinaryOp(b, a, true, true, litFct, affineExprFct, valueFct);
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::floorDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = floor((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    res.initAsLiteral(aa.getContext(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1) {
      res.copy(aa);
    } else if (bval > 1) {
      res.initAsAffineExpr(aa.getContext(), aa.getAffineExpr().floorDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.initAsValue(aa.getContext(),
          aa.getContext().GetRewriter().create<SignedFloorDivIOp>(
              aa.getLoc(), aa.getValue(), bb.getValue()));
#else
      llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                       "support in std");
#endif
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      res.copy(aa);
    } else {
#if CEIL_FLOOR_IN_STD
      res.initAsValue(aa.getContext(),
          aa.getContext().GetRewriter().create<SignedFloorDivIOp>(
              aa.getLoc(), aa.getValue(), bb.getValue()));
#else
      llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                       "support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::ceilDiv(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    int64_t rval = ceil((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    res.initAsLiteral(aa.getContext(), rval);
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1) {
      res.copy(aa);
    } else if (bval > 1) {
      res.initAsAffineExpr(aa.getContext(), aa.getAffineExpr().ceilDiv(bval));
    } else {
#if CEIL_FLOOR_IN_STD
      res.initAsValue(aa.getContext(),
          aa.getContext().GetRewriter().create<SignedCeilDivIOp>(
              aa.getLoc(), aa.getValue(), bb.getValue()));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      res.copy(aa);
    } else {
#if CEIL_FLOOR_IN_STD
      res.initAsValue(aa.getContext(),
          aa.getContext().GetRewriter().create<SignedCeilDivIOp>(
              aa.getLoc(), aa.getValue(), bb.getValue()));
#else
      llvm_unreachable(
          "not implemented yet, wait for the new LLVM/MLIR support in std");
#endif
    }
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::mod(IndexExpr &a, IndexExpr &b) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsLiteral(aa.getContext(), mlir::mod(aa.getLiteral(), bb.getLiteral()));
  };
  F2 affineExprFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval >= 0) {
      res.initAsAffineExpr(aa.getContext(), aa.getAffineExpr() % bval);
    } else {
      res.initAsValue(
          aa.getContext(), aa.getContext().GetRewriter().create<SignedRemIOp>(
                               aa.getLoc(), aa.getValue(), bb.getValue()));
    }
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    res.initAsValue(
        aa.getContext(), aa.getContext().GetRewriter().create<SignedRemIOp>(
                             aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return BinaryOp(a, b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::clamp(IndexExpr &val, IndexExpr &min, IndexExpr &max) {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 litFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                  IndexExpr &max) {
    // assume signed compares
    int64_t smin = min.getLiteral();
    int64_t smax = max.getLiteral();
    int64_t sval = val.getLiteral();
    if (sval < smin)
      sval = smin;
    if (sval > smax)
      sval = smax;
    res.initAsLiteral(val.getContext(), sval);
  };
  F3 valueFct = [&](IndexExpr &res, IndexExpr &val, IndexExpr &min,
                    IndexExpr &max) {
    // copy min, max, and val as we don't want to change the original values.
    IndexExpr minBound(min), newVal(val), maxBound(max);
    newVal.select(val, CmpIPredicate::slt, minBound, minBound, val);
    res.select(newVal, CmpIPredicate::sgt, maxBound, maxBound, newVal);
  };
  return TernaryOp(val, min, max, litFct, valueFct);
}

IndexExpr &IndexExpr::select(IndexExpr &condA, CmpIPredicate comparePred,
    IndexExpr &condB, IndexExpr &trueVal, IndexExpr &falseVal) {
  F4 litFct = [&](IndexExpr &res, IndexExpr &ca, IndexExpr &cb, IndexExpr &tv,
                  IndexExpr &fv) {
    int64_t sca = ca.getLiteral();
    int64_t scb = cb.getLiteral();
    uint64_t uca = (uint64_t)sca;
    uint64_t ucb = (uint64_t)scb;
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (sca == scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::ne:
      if (sca != scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::slt:
      if (sca < scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::sle:
      if (sca <= scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::sgt:
      if (sca > scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::sge:
      if (sca >= scb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::ult:
      if (uca < ucb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::ule:
      if (uca <= ucb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::ugt:
      if (uca > ucb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    case CmpIPredicate::uge:
      if (uca >= ucb)
        res.copy(tv);
      else
        res.copy(fv);
      break;
    default:
      llvm_unreachable("unknown compare opeartor");
    }
  };
  F4 valueFct = [&](IndexExpr &res, IndexExpr &ca, IndexExpr &cb, IndexExpr &tv,
                    IndexExpr &fv) {
    Value compare = ca.getContext().GetRewriter().create<CmpIOp>(
        ca.getLoc(), comparePred, ca.getValue(), cb.getValue());
    Value results = ca.getContext().GetRewriter().create<SelectOp>(
        ca.getLoc(), compare, tv.getValue(), fv.getValue());
    res.initAsValue(ca.getContext(), results);
  };
  return QuaternarySelectOp(condA, condB, trueVal, falseVal, litFct, valueFct);
}

IndexExpr &IndexExpr::min(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.getLiteral();
    auto bbb = bb.getLiteral();
    res.initAsLiteral(aa.getContext(), (aaa < bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.getAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContext &currContext = vvals[0].getContext();
    int dimNum = currContext.getNumDims();
    int symNum = currContext.getNumSymbols();
    auto mapContext = currContext.GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContext.getDimAndSymbolList(dimAndSymList);
    Value minVal = currContext.GetRewriter().create<AffineMinOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.initAsValue(currContext, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.getContext().GetRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::slt, aa.getValue(), bb.getValue());
    Value resVal = aa.getContext().GetRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), bb.getValue());
    res.initAsValue(aa.getContext(), resVal);
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

IndexExpr &IndexExpr::max(SmallVectorImpl<IndexExpr> &vals) {
  F2 litFct = [](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    auto aaa = aa.getLiteral();
    auto bbb = bb.getLiteral();
    res.initAsLiteral(aa.getContext(), (aaa > bbb) ? aaa : bbb);
  };
  Flist affineExprFct = [&](IndexExpr &res, SmallVectorImpl<IndexExpr> &vvals) {
    // Create a list of affine expression
    assert(vvals.size() > 1 && "come here only with 2 or more values");
    SmallVector<AffineExpr, 4> affineExprs;
    for (IndexExpr &vv : vvals) {
      affineExprs.emplace_back(vv.getAffineExpr());
    }
    // Compute a map including the list of affine expressions.
    IndexExprContext &currContext = vvals[0].getContext();
    int dimNum = currContext.getNumDims();
    int symNum = currContext.getNumSymbols();
    auto mapContext = currContext.GetRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    currContext.getDimAndSymbolList(dimAndSymList);
    Value minVal = currContext.GetRewriter().create<AffineMaxOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.initAsValue(currContext, minVal);
  };
  F2 valueFct = [&](IndexExpr &res, IndexExpr &aa, IndexExpr &bb) {
    Value compareVal = aa.getContext().GetRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::sgt, aa.getValue(), bb.getValue());
    Value resVal = aa.getContext().GetRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), bb.getValue());
    res.initAsValue(aa.getContext(), resVal);
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

IndexExpr &IndexExpr::add(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.getContext().createLiteralIndex(b);
  return add(a, bIndex);
}

IndexExpr &IndexExpr::incBy(IndexExpr &b) { return add(*this, b); }

IndexExpr &IndexExpr::incBy(int64_t b) { return add(*this, b); }

IndexExpr &IndexExpr::sub(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.getContext().createLiteralIndex(b);
  return sub(a, bIndex);
}

IndexExpr &IndexExpr::sub(int64_t a, IndexExpr &b) {
  IndexExpr aIndex = b.getContext().createLiteralIndex(a);
  return sub(aIndex, b);
}

IndexExpr &IndexExpr::decBy(IndexExpr &b) { return sub(*this, b); }

IndexExpr &IndexExpr::decBy(int64_t b) { return sub(*this, b); }

IndexExpr &IndexExpr::mult(IndexExpr &a, int64_t b) {
  IndexExpr bIndex = a.getContext().createLiteralIndex(b);
  return mult(a, bIndex);
}

IndexExpr &IndexExpr::multBy(IndexExpr &b) { return mult(*this, b); }

IndexExpr &IndexExpr::multBy(int64_t b) { return mult(*this, b); }

IndexExpr &IndexExpr::clamp(IndexExpr &val, int64_t min, IndexExpr &max) {
  IndexExpr minIndex = val.getContext().createLiteralIndex(min);
  return clamp(val, minIndex, max);
}

IndexExpr &IndexExpr::floorDivBy(IndexExpr &b) { return floorDiv(*this, b); }
IndexExpr &IndexExpr::ceilDivBy(IndexExpr &b) { return ceilDiv(*this, b); }
IndexExpr &IndexExpr::modBy(IndexExpr &b) { return mod(*this, b); }

IndexExpr &IndexExpr::select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr &trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.getContext().createLiteralIndex(condB);
  return select(condA, comparePred, condBIndex, trueVal, falseVal);
}

IndexExpr &IndexExpr::select(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal, IndexExpr &falseVal) {
  IndexExpr condBIndex = condA.getContext().createLiteralIndex(condB);
  IndexExpr trueValIndex = condA.getContext().createLiteralIndex(trueVal);
  return select(condA, comparePred, condBIndex, trueValIndex, falseVal);
}

IndexExpr &IndexExpr::assignIf(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr &trueVal) {
  return select(condA, comparePred, condB, trueVal, *this);
}

IndexExpr &IndexExpr::assignIf(IndexExpr &condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal) {
  return select(condA, comparePred, condB, trueVal, *this);
}
