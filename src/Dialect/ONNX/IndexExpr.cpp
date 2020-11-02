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

IndexExprContext::~IndexExprContext() {
  // Free the memory of each IndexExprImpl in context's container.
  for (IndexExprImpl *obj : container)
    free(obj);
  container.clear();
}

//===----------------------------------------------------------------------===//
// IndexExprContext builder for IndexExpr.
//===----------------------------------------------------------------------===//

IndexExprImpl *IndexExprContext::createIndexExprImpl() {
  // Create implementation object.
  IndexExprImpl *obj = new IndexExprImpl(this);
  assert(obj && "failed to allocate object");
  // Record implementation object in container, so that the context may free
  // them upon context destruction.
  container.emplace_back(obj);
  return obj;
}

IndexExpr IndexExprContext::createIndex(IndexExpr other) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->copy(other.getObjPtr());
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createUndefinedIndex() {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsUndefined();
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createQuestionmarkIndex() {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsQuestionmark(*this);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createLiteralIndex(int64_t val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsLiteral(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createDimIndex(Value val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsDim(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createAffineIndex(AffineExpr val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsAffineExpr(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createValueIndex(Value val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsValue(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createDimIndexFromMemref(
    Value memref, ArrayRef<int64_t> memrefShape, int index) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsDimFromMemref(*this, memref, memrefShape, index);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createSymbolIndexFromArrayAtIndex(
    Operation *op, Value array, uint64_t indexInArray) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsSymbolFromArrayAtIndex(*this, op, array, indexInArray);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createSymbolIndexFromArrayAtIndex(
    Operation *op, Value array, uint64_t indexInArray, int64_t defaultLiteral) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsSymbolFromArrayAtIndex(
      *this, op, array, indexInArray, defaultLiteral);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createSymbolIndex(Value val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsSymbol(*this, val);
  return IndexExpr(obj);
}

// Additional builder for repurposing IndexExpr from parent context.
IndexExpr IndexExprContext::createSymbolIndexFromParentContext(
    IndexExpr parentIndexExpr) {
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
    IndexExprImpl *obj = createIndexExprImpl();
    obj->copy(parentIndexExpr.getObjPtr());
    return IndexExpr(obj);
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

ConversionPatternRewriter &IndexExprContext::getRewriter() const {
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
// IndexExprImpl constructors, initializers
//===----------------------------------------------------------------------===//

IndexExprImpl::IndexExprImpl(IndexExprContext *indexExprContext)
    : defined(false), litteral(false), affine(false), symbol(false), dim(false),
      intLit(0), affineExpr(nullptr), value(nullptr),
      context(indexExprContext) {}

IndexExprImpl &IndexExprImpl::initAsUndefined() {
  return init(/*context*/ nullptr, /*isDefined*/ false, /*litteral*/ false,
      /*affine*/ false, /*symbol*/ false, /*dim*/ false, 0, AffineExpr(nullptr),
      Value(nullptr));
}

IndexExprImpl &IndexExprImpl::initAsQuestionmark(IndexExprContext &newContext) {
  return init(&newContext, /*isDefined*/ true, /*litteral*/ false,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, 0, AffineExpr(nullptr),
      Value(nullptr));
}

IndexExprImpl &IndexExprImpl::initAsLiteral(
    IndexExprContext &newContext, int64_t val) {
  return init(&newContext, /*isDefined*/ true, /*litteral*/ true,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, val,
      AffineExpr(nullptr), Value(nullptr));
}

IndexExprImpl &IndexExprImpl::initAsDim(
    IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(
      newContext, val, /*affine*/ true, /*symbol*/ false, /*dim*/ true);
}

IndexExprImpl &IndexExprImpl::initAsSymbol(
    IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(
      newContext, val, /*affine*/ true, /*symbol*/ true, /*dim*/ false);
}

IndexExprImpl &IndexExprImpl::initAsValue(
    IndexExprContext &newContext, Value val) {
  return initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ false,
      /*symbol*/ false, /*dim*/ false);
}

IndexExprImpl &IndexExprImpl::initAsAffineExpr(
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

IndexExprImpl &IndexExprImpl::init(IndexExprContext *newContext,
    bool newIsDefined, bool newIsIntLit, bool newIsAffine, bool newIsSymbol,
    bool newIsDim, int newIntLit, AffineExpr newAffineExpr, Value newValue) {
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

IndexExprImpl &IndexExprImpl::initAsLitQuestionmarkOrValue(
    IndexExprContext &newContext, Value val, bool newIsAfine, bool newIsSymbol,
    bool newIsDim) {
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
    val = newContext.getRewriter().create<IndexCastOp>(
        newContext.getLoc(), newContext.getRewriter().getIndexType(), val);
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
// IndexExprImpl initializers that extract info
//===----------------------------------------------------------------------===//

IndexExprImpl &IndexExprImpl::initAsDimFromMemref(IndexExprContext &newContext,
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
  Value dynVal = newContext.getRewriter().create<DimOp>(
      newContext.getLoc(), memref, index);
  return initAsDim(newContext, dynVal);
}

IndexExprImpl &IndexExprImpl::initAsSymbolFromArrayAtIndex(
    IndexExprContext &newContext, Operation *op, Value array,
    uint64_t indexInArray) {
  if (auto attrArray = getDenseElementAttributeFromValue(array)) {
    // We extracted an dense attribute from definition of operand.
    if (indexInArray >= attrArray.getType().getDimSize(0)) {
      printf("error 1\n");
      op->emitError("operand literal has wrong shape");
      return initAsUndefined();
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({indexInArray}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return initAsLiteral(newContext, attrInt);
  }
  // We must read value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return initAsQuestionmark(newContext);
  }
  // Emit code to read array.
  Value indexVal = emitConstantOp(newContext.getRewriter(), newContext.getLoc(),
      newContext.getRewriter().getIndexType(), indexInArray);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.getRewriter().create<AffineLoadOp>(
      newContext.getLoc(), array, memrefVal);
  return initAsSymbol(newContext, loadVal);
}

IndexExprImpl &IndexExprImpl::initAsSymbolFromArrayAtIndex(
    IndexExprContext &newContext, Operation *op, Value array,
    uint64_t indexInArray, int64_t defaultLiteral) {
  // Check if we have an operand.
  if (array.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value.
    return initAsLiteral(newContext, defaultLiteral);
  }
  if (auto attrArray = getDenseElementAttributeFromValue(array)) {
    // We extracted an dense attribute from definition of operand.
    if (indexInArray > attrArray.getType().getDimSize(0)) {
      // Not enought attributes for this index, return the default value.
      return initAsLiteral(newContext, defaultLiteral);
    }
    // We have enought attributes for this index, get the value.
    Attribute attrVal = attrArray.getValue(ArrayRef<uint64_t>({indexInArray}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return initAsLiteral(newContext, attrInt);
  }
  // Read the value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    return initAsQuestionmark(newContext);
  }
  // Emit the code to read array.
  Value indexVal = emitConstantOp(newContext.getRewriter(), newContext.getLoc(),
      newContext.getRewriter().getIndexType(), indexInArray);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.getRewriter().create<AffineLoadOp>(
      newContext.getLoc(), array, memrefVal);
  return initAsSymbol(newContext, loadVal);
}

void IndexExprImpl::copy(IndexExprImpl *other) {
  assert(context && "all index expr must have a defined context");
  // Preserve this's context, copy the remaining attributes from other.
  init(context, other->defined, other->litteral, other->affine, other->symbol,
      other->dim, other->intLit, other->affineExpr, other->value);
}

//===----------------------------------------------------------------------===//
// IndexExpr copy and setters.
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::deepCopy() const {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. copy info in the meanwhile.
  return getContext().createIndex(*this);
}

/*
void IndexExpr::setContext(IndexExprContext &newContext) {
  getObj().context = &newContext;
}
*/

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
//===----------------------------------------------------------------------===//

bool IndexExpr::isDefined() const {
  assert(!getObj().defined || hasContext());
  return getObj().defined;
}

bool IndexExpr::isLiteral() const {
  assert(isDefined());
  return getObj().litteral;
}

bool IndexExpr::isQuestionmark() const {
  assert(isDefined());
  return !isLiteral();
}

bool IndexExpr::isAffine() const {
  assert(isDefined());
  return getObj().affine;
}

bool IndexExpr::isSymbol() const {
  assert(isDefined());
  return getObj().symbol;
}

bool IndexExpr::isDim() const {
  assert(isDefined());
  return getObj().dim;
}

bool IndexExpr::isShapeInferencePass() const {
  return getContext().isShapeInferencePass();
}

bool IndexExpr::hasContext() const { return getObj().context != nullptr; }

bool IndexExpr::hasAffineExpr() const {
  assert(isDefined());
  return !(!getObj().affineExpr);
}

bool IndexExpr::hasValue() const {
  assert(isDefined());
  return !(!getObj().value);
}

//===----------------------------------------------------------------------===//
// IndexExpr Getters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const {
  assert(isLiteral());
  return getObj().intLit;
}

AffineExpr IndexExpr::getAffineExpr() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  if (isLiteral()) {
    // Create a literal.
    getObj().affineExpr = getRewriter().getAffineConstantExpr(getObj().intLit);
  } else if (isSymbol()) {
    // Create a symbol value expr and register its value in the
    // array of symbols. Has value because symbols are gen on demand from
    // values.
    assert(hasValue());
    int id = getContext().addSymbol(getObj().value);
    getObj().affineExpr = getContext().getRewriter().getAffineSymbolExpr(id);
  } else if (isDim()) {
    // Create a dim/index value expr and register its value in the
    // array of dims/indices. Has value because dims are gen on demand from
    // values.
    assert(hasValue());
    int id = getContext().addDim(getObj().value);
    getObj().affineExpr = getContext().getRewriter().getAffineDimExpr(id);
  } else {
    assert(
        hasAffineExpr() && "requesting affine expr of incompatible IndexExpr");
  }
  return getObj().affineExpr;
}

Value IndexExpr::getValue() {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
  if (isLiteral()) {
    // Create a litteral constant.
    getObj().value =
        getRewriter().create<ConstantIndexOp>(getLoc(), getObj().intLit);
  } else if (hasAffineExpr()) {
    // Has an affine expression: need to build a map, and then perform an
    // affine.apply.
    int dimNum = getContext().getNumDims();
    int symNum = getContext().getNumSymbols();
    AffineMap map = AffineMap::get(
        dimNum, symNum, {getObj().affineExpr}, getRewriter().getContext());
    // We need to concatenate the dims and symbol into a single
    // list, and then use the apply.
    SmallVector<Value, 4> list;
    getContext().getDimAndSymbolList(list);
    getObj().value = getRewriter().create<AffineApplyOp>(getLoc(), map, list);
  } else {
    assert(hasValue());
  }
  return getObj().value;
}

IndexExprContext &IndexExpr::getContext() const { return *getContextPtr(); }

IndexExprContext *IndexExpr::getContextPtr() const {
  assert(hasContext());
  return getObj().context;
}

ConversionPatternRewriter &IndexExpr::getRewriter() const {
  return getContext().getRewriter();
}

Location IndexExpr::getLoc() const { return getContext().getLoc(); }

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
  printf(" context(0x%llx)\n", (long long unsigned)getContextPtr());
#endif
}

IndexExprImpl &IndexExpr::getObj() const { return *getObjPtr(); }

IndexExprImpl *IndexExpr::getObjPtr() const {
  assert(indexExprObj);
  return indexExprObj;
}

//===----------------------------------------------------------------------===//
// IndexExpr Op Support.
//===----------------------------------------------------------------------===//

// Used for add/sub/mult/ceilDiv/floorDiv
IndexExpr IndexExpr::BinaryOp(IndexExpr b, bool affineWithLitB,
    bool canBeAffine, F2 litFct, F2 affineExprFct, F2 valueFct) {
  assert(getContextPtr() == b.getContextPtr() && "incompatible contexts");
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
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    return getContext().createQuestionmarkIndex();
  if (resIsAffine)
    // Use affine values.
    return affineExprFct(*this, b);
  // Use values.
  return valueFct(*this, b);
}

// Used for clamp.
IndexExpr IndexExpr::TernaryOp(
    IndexExpr b, IndexExpr c, F3 litFct, F3 valueFct) {
  assert(getContextPtr() == b.getContextPtr() &&
         getContextPtr() == c.getContextPtr() && "incompatible contexts");
  // Literal integer if a, b, and c are literals. Output is not affine (unless
  // all 3 are literals).
  bool resIsLit = isLiteral() && b.isLiteral() && c.isLiteral();
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this, b, c);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    return getContext().createQuestionmarkIndex();
  // Use values.
  return valueFct(*this, b, c);
}

/*static*/ IndexExpr IndexExpr::QuaternarySelectOp(IndexExpr compA,
    IndexExpr compB, IndexExpr trueVal, IndexExpr falseVal, F4 litFct,
    F4 valueFct) {
  assert(compA.getContextPtr() == compB.getContextPtr() &&
         compA.getContextPtr() == trueVal.getContextPtr() &&
         compA.getContextPtr() == falseVal.getContextPtr() &&
         "incompatible contexts");
  // Check first if the test (ca & cb) can be evaluated at compile time.
  if (compA.isLiteral() && compB.isLiteral())
    // Comparison will set the right const/affine depending on the input
    // selected, as the compare can be evaluated at compile time.
    return litFct(compA, compB, trueVal, falseVal);
  if (compA.isShapeInferencePass())
    // Just set as undefined
    return compA.getContext().createQuestionmarkIndex();
  // We cannot represent this as an affine expression, so go directly
  // to values.
  return valueFct(compA, compB, trueVal, falseVal);
}

// The affine reduction labda function processes the whole list and must init
// the result. Literal and Values treat one operation at a time
/* static*/ IndexExpr IndexExpr::reductionOp(
    SmallVectorImpl<IndexExpr> &vals, F2 litRed, Flist affineRed, F2 valueRed) {
  // If no values, result is undefined.
  int size = vals.size();
  if (size == 0) {
    return vals[0].getContext().createUndefinedIndex();
  }
  // Set the output to the first value.
  IndexExpr res = vals[0].deepCopy();
  // If list has one element, we are done. Literal/Affine... will be the same as
  // this single element.
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
    assert(vals[0].getContextPtr() == vals[i].getContextPtr() &&
           "incompatible contexts");
  }
  if (resIsLit) {
    // Process int literals, if we only have literal values.
    // Result was set to first element, which by default is literal/affine. This
    // will be the correct result for the output.
    for (int i = 1; i < size; ++i) {
      litRed(res, vals[i]);
    }
    return res;
  }
  if (vals[0].isShapeInferencePass()) {
    // Just set as undefined
    res.getObj().initAsQuestionmark(res.getContext());
    return res;
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

IndexExpr IndexExpr::operator+(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() + bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() + bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createValueIndex(aa.getRewriter().create<AddIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return BinaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator-(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() - bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() - bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createValueIndex(aa.getRewriter().create<SubIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return BinaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator*(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() * bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() * bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1)
      return aa.deepCopy();
    return aa.getContext().createValueIndex(aa.getRewriter().create<MulIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Literal should be place in second argument; do so if a is a lit.
  if (isLiteral())
    return b.BinaryOp(*this, true, true, litFct, affineExprFct, valueFct);
  return BinaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::floorDiv(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    int64_t rval = floor((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return aa.getContext().createAffineIndex(
          aa.getAffineExpr().floorDiv(bval));
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedFloorDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedFloorDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  // Index b must be a literal.
  return BinaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::ceilDiv(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    int64_t rval = ceil((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return aa.getContext().createAffineIndex(
          aa.getAffineExpr().ceilDiv(bval));
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedCeilDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedCeilDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  // Index b must be a literal.
  return BinaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator%(IndexExpr b) {
  F2 litFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    int64_t rval = mlir::mod(aa.getLiteral(), bb.getLiteral());
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval >= 0)
      return aa.getContext().createAffineIndex(aa.getAffineExpr() % bval);
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedRemIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  F2 valueFct = [](IndexExpr aa, IndexExpr bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
#if CEIL_FLOOR_IN_STD
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedRemIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
#else
    llvm_unreachable("not implemented yet, wait for the new LLVM/MLIR "
                     "support in std");
#endif
  };
  // Index b must be a literal.
  return BinaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::clamp(IndexExpr min, IndexExpr max) {
  // Functions below uncoditionally override rr with the clipped value of val.
  F3 litFct = [](IndexExpr val, IndexExpr min, IndexExpr max) -> IndexExpr {
    // assume signed compares
    int64_t smin = min.getLiteral();
    int64_t smax = max.getLiteral();
    int64_t res = val.getLiteral();
    if (res < smin)
      res = smin;
    if (res > smax)
      res = smax;
    return val.getContext().createLiteralIndex(res);
  };
  F3 valueFct = [](IndexExpr val, IndexExpr min, IndexExpr max) {
    IndexExpr res1 = select(val, CmpIPredicate::slt, min, min, val);
    IndexExpr res2 = select(res1, CmpIPredicate::sgt, max, max, res1);
    return res2;
  };
  return TernaryOp(min, max, litFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::select(IndexExpr condA,
    CmpIPredicate comparePred, IndexExpr condB, IndexExpr trueVal,
    IndexExpr falseVal) {
  F4 litFct = [&](IndexExpr ca, IndexExpr cb, IndexExpr tv,
                  IndexExpr fv) -> IndexExpr {
    int64_t sca = ca.getLiteral();
    int64_t scb = cb.getLiteral();
    uint64_t uca = (uint64_t)sca;
    uint64_t ucb = (uint64_t)scb;
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (sca == scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::ne:
      if (sca != scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::slt:
      if (sca < scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::sle:
      if (sca <= scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::sgt:
      if (sca > scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::sge:
      if (sca >= scb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::ult:
      if (uca < ucb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::ule:
      if (uca <= ucb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::ugt:
      if (uca > ucb)
        return tv.deepCopy();
      return fv.deepCopy();
    case CmpIPredicate::uge:
      if (uca >= ucb)
        return tv.deepCopy();
      return fv.deepCopy();
    default:
      llvm_unreachable("unknown compare opeartor");
    }
  };
  F4 valueFct = [&](IndexExpr ca, IndexExpr cb, IndexExpr tv,
                    IndexExpr fv) -> IndexExpr {
    Value compare = ca.getRewriter().create<CmpIOp>(
        ca.getLoc(), comparePred, ca.getValue(), cb.getValue());
    Value results = ca.getRewriter().create<SelectOp>(
        ca.getLoc(), compare, tv.getValue(), fv.getValue());
    return ca.getContext().createValueIndex(results);
  };
  return QuaternarySelectOp(condA, condB, trueVal, falseVal, litFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::min(SmallVectorImpl<IndexExpr> &vals) {
  // Res is already an literal int, we are reducing into it.
  F2 litFct = [](IndexExpr res, IndexExpr aa) -> IndexExpr {
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
    IndexExprContext &context = vvals[0].getContext();
    int dimNum = context.getNumDims();
    int symNum = context.getNumSymbols();
    auto mapContext = context.getRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    context.getDimAndSymbolList(dimAndSymList);
    Value minVal = context.getRewriter().create<AffineMinOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsValue(context, minVal);
    return res;
  };
  // Res is already defined, we are reducing into it.
  F2 valueFct = [](IndexExpr res, IndexExpr aa) {
    Value compareVal = res.getRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::slt, aa.getValue(), res.getValue());
    Value resVal = aa.getContext().getRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsValue(res.getContext(), res.getValue());
    return res;
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

/*static*/ IndexExpr IndexExpr::max(SmallVectorImpl<IndexExpr> &vals) {
  // Res is already an literal int, we are reducing into it.
  F2 litFct = [](IndexExpr res, IndexExpr aa) -> IndexExpr {
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
    IndexExprContext &context = vvals[0].getContext();
    int dimNum = context.getNumDims();
    int symNum = context.getNumSymbols();
    auto mapContext = context.getRewriter().getContext();
    AffineMap map = AffineMap::get(dimNum, symNum, affineExprs, mapContext);
    // Compute the min value out of this map.
    SmallVector<Value, 4> dimAndSymList;
    context.getDimAndSymbolList(dimAndSymList);
    Value minVal = context.getRewriter().create<AffineMaxOp>(
        vvals[0].getLoc(), map, dimAndSymList);
    res.getObj().initAsValue(context, minVal);
    return res;
   };
  // Res is already defined, we are reducing into it.
  F2 valueFct = [](IndexExpr res, IndexExpr aa) {
    Value compareVal = res.getRewriter().create<CmpIOp>(
        aa.getLoc(), CmpIPredicate::sgt, aa.getValue(), res.getValue());
    Value resVal = aa.getContext().getRewriter().create<SelectOp>(
        aa.getLoc(), compareVal, aa.getValue(), res.getValue());
    res.getObj().initAsValue(res.getContext(), res.getValue());
    return res;
  };
  return reductionOp(vals, litFct, affineExprFct, valueFct);
}

//===----------------------------------------------------------------------===//
// IndexExpr Ops Derivatives
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::operator+(int64_t b) {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this + bIndex;
}

IndexExpr IndexExpr::operator-(int64_t b) {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this - bIndex;
}

IndexExpr IndexExpr::operator*(int64_t b) {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this * bIndex;
}

IndexExpr IndexExpr::clamp(int64_t min, IndexExpr max) {
  IndexExpr minIndex = getContext().createLiteralIndex(min);
  return clamp(minIndex, max);
}

IndexExpr IndexExpr::select(IndexExpr condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr trueVal, IndexExpr falseVal) {
  IndexExpr condBIndex = condA.getContext().createLiteralIndex(condB);
  return select(condA, comparePred, condBIndex, trueVal, falseVal);
}

IndexExpr IndexExpr::select(IndexExpr condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal, IndexExpr falseVal) {
  IndexExpr condBIndex = condA.getContext().createLiteralIndex(condB);
  IndexExpr trueValIndex = condA.getContext().createLiteralIndex(trueVal);
  return select(condA, comparePred, condBIndex, trueValIndex, falseVal);
}

IndexExpr IndexExpr::setIf(IndexExpr condA, CmpIPredicate comparePred,
    int64_t condB, IndexExpr trueVal) {
  return select(condA, comparePred, condB, trueVal, *this);
}

IndexExpr IndexExpr::setIf(IndexExpr condA, CmpIPredicate comparePred,
    int64_t condB, int64_t trueVal) {
  return select(condA, comparePred, condB, trueVal, *this);
}
