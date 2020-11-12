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
#define DEBUG 0

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
    : rewriter(rewriter), loc(loc), dims(), symbols(), parentContext(nullptr),
      zero(nullptr), one(nullptr), minusOne(nullptr) {}

IndexExprContext::IndexExprContext(IndexExprContext &newParentContext)
    : rewriter(newParentContext.rewriter), loc(newParentContext.loc), dims(),
      symbols(), parentContext(nullptr), zero(nullptr), one(nullptr),
      minusOne(nullptr) {
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

IndexExpr IndexExprContext::createIndex(IndexExpr const other) {
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

IndexExpr IndexExprContext::createLiteralIndex(int64_t const val) {
  IndexExprImpl *obj;
  // Provide reuse for 0/1/-1.
  if (val == 0) {
    if (zero)
      return IndexExpr(zero);
    zero = obj = createIndexExprImpl();
  } else if (val == 1) {
    if (one)
      return IndexExpr(one);
    one = obj = createIndexExprImpl();
  } else if (val == -1) {
    if (minusOne)
      return IndexExpr(minusOne);
    minusOne = obj = createIndexExprImpl();
  } else {
    obj = createIndexExprImpl();
  }
  obj->initAsLiteral(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createDimIndex(Value const val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsDim(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createLoopIterIndex(Value const val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsDim(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createAffineIndex(AffineExpr const val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsAffineExpr(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createValueIndex(Value const val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsValue(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createPredicateValueIndex(Value const val) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsPredicateValue(*this, val);
  return IndexExpr(obj);
}

IndexExpr IndexExprContext::createDimIndexFromShapedType(
    Value tensorOrMemref, int index) {
  IndexExprImpl *obj = createIndexExprImpl();
  obj->initAsDimFromShapedType(*this, tensorOrMemref, index);
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
    IndexExpr const parentIndexExpr) {
  // Make sure that we are using the propper parent context
  assert(parentIndexExpr.getContextPtr() == parentContext &&
         "parent index is not from the parent's context");
  // When the parent expression is already affine in the outer context, it will
  // remain afine in the child's context as wee. So we keep it as such, to get
  // as exprssive affine expressions as possible. We could retrict reuse for
  // literal only.
  if (parentIndexExpr.isAffine()) {
    // Reuse affine expression.
    IndexExprImpl *obj = createIndexExprImpl();
    obj->copy(parentIndexExpr.getObjPtr());
    return IndexExpr(obj);
  }
  // Non affine, create a symbol.
  return createSymbolIndex(parentIndexExpr.getValue());
}

//===----------------------------------------------------------------------===//
// IndexExprContext builder for lists of IndexExpr.
//===----------------------------------------------------------------------===//

bool IndexExprContext::createDimIndicesFromShapedType(
    Value tensorOrMemref, SmallVectorImpl<IndexExpr> &dimIndices) {
  // Clear output.
  dimIndices.clear();
  // Scan type and shape, bail if incompatible.
  ShapedType type = tensorOrMemref.getType().cast<ShapedType>();
  int size = type.getShape().size();
  // Scan tensor or memref.
  bool successful = true;
  for (int i = 0; i < size; ++i) {
    IndexExpr index = createDimIndexFromShapedType(tensorOrMemref, i);
    if (index.isUndefined())
      successful = false;
    dimIndices.emplace_back(index);
  }
  return successful;
}

bool IndexExprContext::createSymbolIndicesFromArray(Operation *op, Value array,
    int arraySize, SmallVectorImpl<IndexExpr> &symbolIndices) {
  // Clear output.
  symbolIndices.clear();
  // createSymbolIndexFromArrayAtIndex();
  bool successful = true;
  for (int i = 0; i < arraySize; ++i) {
    IndexExpr index = createSymbolIndexFromArrayAtIndex(op, array, i);
    if (index.isUndefined())
      successful = false;
    symbolIndices.emplace_back(index);
  }
  return successful;
}

bool IndexExprContext::createSymbolIndicesFromArray(Operation *op, Value array,
    int arraySize, int64_t defaultLiteral,
    SmallVectorImpl<IndexExpr> &symbolIndices) {
  // Clear output.
  symbolIndices.clear();
  // createSymbolIndexFromArrayAtIndex();
  bool successful = true;
  for (int i = 0; i < arraySize; ++i) {
    IndexExpr index =
        createSymbolIndexFromArrayAtIndex(op, array, i, defaultLiteral);
    if (index.isUndefined())
      successful = false;
    symbolIndices.emplace_back(index);
  }
  return successful;
}

//===----------------------------------------------------------------------===//
// IndexExprContext support for creating possibly affine load and store ops.
//===----------------------------------------------------------------------===//

Value IndexExprContext::createLoadOp(
    Value memref, SmallVectorImpl<IndexExpr> &indices) {
  bool affineIndices = true;
  SmallVector<Value, 4> loadIndices;
  for (IndexExpr ie : indices) {
    if (!ie.isAffine())
      affineIndices = false;
    loadIndices.emplace_back(ie.getValue());
  }
  if (affineIndices)
    return getRewriter().create<AffineLoadOp>(getLoc(), memref, loadIndices);
  // Not affine, use regular load.
  return getRewriter().create<LoadOp>(getLoc(), memref, loadIndices);
}

void IndexExprContext::createStoreOp(
    Value val, Value memref, SmallVectorImpl<IndexExpr> &indices) {
  bool affineIndices = true;
  SmallVector<Value, 4> storeIndices;
  for (IndexExpr ie : indices) {
    if (!ie.isAffine())
      affineIndices = false;
    storeIndices.emplace_back(ie.getValue());
  }
  if (affineIndices) {
    getRewriter().create<AffineStoreOp>(getLoc(), val, memref, storeIndices);
  } else { // Not affine, use regular load.
    getRewriter().create<StoreOp>(getLoc(), val, memref, storeIndices);
  }
}

//===----------------------------------------------------------------------===//
// IndexExprContext support for dim and symbol lists in affine exprs.
//===----------------------------------------------------------------------===//

int IndexExprContext::addDim(Value const value) {
  dims.emplace_back(value);
  return dims.size() - 1;
  ;
}
int IndexExprContext::addSymbol(Value const value) {
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

/*static*/ bool IndexExprContext::areAllLiteral(
    SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.isLiteral())
      return false;
  }
  return true;
}

/*static*/ bool IndexExprContext::areAllAffine(
    SmallVectorImpl<IndexExpr> &list) {
  for (auto index : list) {
    if (!index.isAffine())
      return false;
  }
  return true;
}

/*static*/ void IndexExprContext::getOutputDimsForType(
    SmallVectorImpl<IndexExpr> &outputIndices,
    SmallVectorImpl<int64_t> &outputDims) {
  outputDims.clear();
  for (IndexExpr &outputIndex : outputIndices) {
    if (outputIndex.isLiteral()) {
      int64_t val = outputIndex.getLiteral();
      assert(val >= 0 && "expected positive values only");
      outputDims.emplace_back(val);
    } else
      outputDims.emplace_back(-1);
  }
}

//===----------------------------------------------------------------------===//
// IndexExprImpl constructors, initializers
//===----------------------------------------------------------------------===//

IndexExprImpl::IndexExprImpl(IndexExprContext *indexExprContext)
    : defined(false), literal(false), affine(false), symbol(false), dim(false),
      intLit(0), affineExpr(nullptr), value(nullptr),
      context(indexExprContext) {}

void IndexExprImpl::initAsUndefined() {
  init(/*context*/ nullptr, /*isDefined*/ false, /*literal*/ false,
      /*affine*/ false, /*symbol*/ false, /*dim*/ false, /*predType*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsQuestionmark(IndexExprContext &newContext) {
  init(&newContext, /*isDefined*/ true, /*literal*/ false,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, /*predType*/ false, 0,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsLiteral(
    IndexExprContext &newContext, int64_t const val) {
  init(&newContext, /*isDefined*/ true, /*literal*/ true,
      /*affine*/ true, /*symbol*/ false, /*dim*/ false, /*predType*/ false, val,
      AffineExpr(nullptr), Value(nullptr));
}

void IndexExprImpl::initAsDim(IndexExprContext &newContext, Value const val) {
  initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ true,
      /*symbol*/ false, /*dim*/ true, /*predType*/ false);
}

void IndexExprImpl::initAsSymbol(
    IndexExprContext &newContext, Value const val) {
  initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ true,
      /*symbol*/ true, /*dim*/ false, /*predType*/ false);
}

void IndexExprImpl::initAsValue(IndexExprContext &newContext, Value const val) {
  initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ false,
      /*symbol*/ false, /*dim*/ false, /*predType*/ false);
}

void IndexExprImpl::initAsPredicateValue(
    IndexExprContext &newContext, Value const val) {
  initAsLitQuestionmarkOrValue(newContext, val, /*affine*/ false,
      /*symbol*/ false, /*dim*/ false, /*predType*/ true);
}

void IndexExprImpl::initAsAffineExpr(
    IndexExprContext &newContext, AffineExpr const val) {
  // Check if the affine expression is reduced to a constant expr.
  AffineExpr simpleVal = simplifyAffineExpr(
      val, newContext.getNumDims(), newContext.getNumSymbols());
  AffineConstantExpr constAffineExpr = simpleVal.dyn_cast<AffineConstantExpr>();
  if (constAffineExpr) {
    initAsLiteral(newContext, constAffineExpr.getValue());
  } else {
    init(&newContext, /*isDefined*/ true, /*literal*/ false,
        /*affine*/ true, /*symbol*/ false, /*dim*/ false, /*predType*/ false, 0,
        AffineExpr(val), Value(nullptr));
  }
}

void IndexExprImpl::init(IndexExprContext *newContext, bool newIsDefined,
    bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
    bool newIsPredType, int64_t const newIntLit, AffineExpr const newAffineExpr,
    Value const newValue) {
  context = newContext;
  defined = newIsDefined;
  literal = newIsIntLit;
  affine = newIsAffine;
  symbol = newIsSymbol;
  dim = newIsDim;
  predType = newIsPredType;
  intLit = newIntLit;
  affineExpr = newAffineExpr;
  value = newValue;
}

void IndexExprImpl::initAsLitQuestionmarkOrValue(IndexExprContext &newContext,
    Value const val, bool newIsAfine, bool newIsSymbol, bool newIsDim,
    bool newIsPredType) {
  // Do we have a literal integer, if we do, handle it now.
  int64_t valIntLit;
  if (getIntegerLiteralFromValue(val, valIntLit)) {
    // We have an integer. No need for symbol or dim. It is by default affine.
    // Ignore the predicate type as we treat all literal int as untyped.
    initAsLiteral(newContext, valIntLit);
    return;
  }
  // We have a value that is not a literal.
  if (newContext.isShapeInferencePass()) {
    initAsQuestionmark(newContext);
    return;
  }
  // Check that the value is of the right type.
  auto type = val.getType();
  Value newVal = val;
  if (type.isa<IntegerType>()) {
    if (!newIsPredType) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = newContext.getRewriter().create<IndexCastOp>(
          newContext.getLoc(), newContext.getRewriter().getIndexType(), newVal);
    } else {
      // We have an integer for predicate types, all good.
    }
  } else if (type.isa<IndexType>()) {
    if (newIsPredType) {
      // We need to convert the int into an index, since we are dealing with
      // index expressions.
      newVal = newContext.getRewriter().create<IndexCastOp>(
          newContext.getLoc(), newContext.getRewriter().getI1Type(), newVal);
    } else {
      // have an index type for a non-predicate type, all good.
    }
  } else {
    llvm_unreachable("unsupported element type");
  }
  // Now record the value. Affine Expr will be created on demand by
  // getAffineExpr.
  assert(!(newIsDim && newIsSymbol) &&
         "cannot have dim and symbol at the same time");
  init(&newContext, /*isDefined*/ true, /*literal*/ false, newIsAfine,
      newIsSymbol, newIsDim, newIsPredType, 0, AffineExpr(nullptr), newVal);
}

//===----------------------------------------------------------------------===//
// IndexExprImpl initializers that extract info
//===----------------------------------------------------------------------===//

void IndexExprImpl::initAsDimFromShapedType(
    IndexExprContext &newContext, Value tensorOrMemref, int index) {
  // Get shape from tensor or memref value.
  ArrayRef<int64_t> shape =
      tensorOrMemref.getType().cast<ShapedType>().getShape();
  if (shape[index] >= 0) {
    // We have a constant dimension.
    int64_t intVal = shape[index];
    initAsLiteral(newContext, intVal);
    return;
  }
  // We have a dynamic dimension.
  if (newContext.isShapeInferencePass()) {
    initAsQuestionmark(newContext);
  } else {
    Value dynVal = newContext.getRewriter().create<DimOp>(
        newContext.getLoc(), tensorOrMemref, index);
    initAsDim(newContext, dynVal);
  }
}

void IndexExprImpl::initAsSymbolFromArrayAtIndex(IndexExprContext &newContext,
    Operation *op, Value array, uint64_t indexInArray) {
  if (auto attrArray = getDenseElementAttributeFromValue(array)) {
    // We extracted an dense attribute from definition of operand.
    if (indexInArray >= attrArray.getType().getDimSize(0)) {
      printf("error 1\n");
      op->emitError("operand literal has wrong shape");
      initAsUndefined();
      return;
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({indexInArray}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    initAsLiteral(newContext, attrInt);
    return;
  }
  // We must read value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    initAsQuestionmark(newContext);
    return;
  }
  // Emit code to read array.
  Value indexVal = emitConstantOp(newContext.getRewriter(), newContext.getLoc(),
      newContext.getRewriter().getIndexType(), indexInArray);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.getRewriter().create<AffineLoadOp>(
      newContext.getLoc(), array, memrefVal);
  initAsSymbol(newContext, loadVal);
}

void IndexExprImpl::initAsSymbolFromArrayAtIndex(IndexExprContext &newContext,
    Operation *op, Value array, uint64_t indexInArray, int64_t defaultLiteral) {
  // Check if we have an operand.
  if (array.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value.
    initAsLiteral(newContext, defaultLiteral);
    return;
  }
  if (auto attrArray = getDenseElementAttributeFromValue(array)) {
    // We extracted an dense attribute from definition of operand.
    if (indexInArray > attrArray.getType().getDimSize(0)) {
      // Not enought attributes for this index, return the default value.
      initAsLiteral(newContext, defaultLiteral);
      return;
    }
    // We have enought attributes for this index, get the value.
    Attribute attrVal = attrArray.getValue(ArrayRef<uint64_t>({indexInArray}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    initAsLiteral(newContext, attrInt);
    return;
  }
  // Read the value from an array.
  if (newContext.isShapeInferencePass()) {
    // Not a constant; don't add code.
    initAsQuestionmark(newContext);
    return;
  }
  // Emit the code to read array.
  Value indexVal = emitConstantOp(newContext.getRewriter(), newContext.getLoc(),
      newContext.getRewriter().getIndexType(), indexInArray);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = newContext.getRewriter().create<AffineLoadOp>(
      newContext.getLoc(), array, memrefVal);
  initAsSymbol(newContext, loadVal);
}

void IndexExprImpl::copy(IndexExprImpl const *other) {
  assert(context && "all index expr must have a defined context");
  // Preserve this's context, copy the remaining attributes from other.
  init(context, other->defined, other->literal, other->affine, other->symbol,
      other->dim, other->predType, other->intLit, other->affineExpr,
      other->value);
}

//===----------------------------------------------------------------------===//
// IndexExpr copy and setters.
//===----------------------------------------------------------------------===//

IndexExpr IndexExpr::deepCopy() const {
  // If we go to a model like Values & AffineExpr with a pointer to the actual
  // data, we should just make the indirection here. copy info in the meanwhile.
  return getContext().createIndex(*this);
}

//===----------------------------------------------------------------------===//
// IndexExpr list querries.
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

bool IndexExpr::isPredType() const {
  assert(isDefined());
  return getObj().predType;
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

bool IndexExpr::isLiteralAndIdenticalTo(int64_t b) const {
  // When dealing with non-literal, don't test and return true.
  if (!isLiteral())
    return false;
  // We have a literal, now make sure they are the same
  return getLiteral() == b;
}

bool IndexExpr::isLiteralAndDifferentThan(int64_t b) const {
  // When dealing with non-literal, don't test and return true.
  if (!isLiteral())
    return false;
  // We have a literal, now make sure they are different
  return getLiteral() != b;
}

bool IndexExpr::isLiteralAndIdenticalTo(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return true.
  if (!isLiteral() || !b.isLiteral())
    return false;
  // We have literals, now make sure they are the same
  return getLiteral() == b.getLiteral();
}

bool IndexExpr::isLiteralAndDifferentThan(IndexExpr const b) const {
  // When dealing with non-literal, don't test and return true.
  if (!isLiteral() || !b.isLiteral())
    return false;
  // We have literals, now make sure they are different
  return getLiteral() != b.getLiteral();
}

//===----------------------------------------------------------------------===//
// IndexExpr Getters.
//===----------------------------------------------------------------------===//

int64_t IndexExpr::getLiteral() const {
  assert(isLiteral());
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

Value IndexExpr::getValue() const {
  assert(!isShapeInferencePass() && "cannot get affine during shape inference");
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

IndexExprContext *IndexExpr::getContextPtr() const {
  assert(hasContext());
  return getObj().context;
}

ConversionPatternRewriter &IndexExpr::getRewriter() const {
  return getContext().getRewriter();
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
IndexExpr IndexExpr::binaryOp(IndexExpr const b, bool affineWithLitB,
    bool canBeAffine, F2 litFct, F2 affineExprFct, F2 valueFct) const {
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

IndexExpr IndexExpr::compareOp(
    CmpIPredicate comparePred, IndexExpr const b) const {
  F2 litFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t aaa = aa.getLiteral();
    int64_t bbb = bb.getLiteral();
    switch (comparePred) {
    case CmpIPredicate::eq:
      if (aaa == bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    case CmpIPredicate::ne:
      if (aaa != bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    case CmpIPredicate::slt:
      if (aaa < bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    case CmpIPredicate::sle:
      if (aaa <= bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    case CmpIPredicate::sgt:
      if (aaa > bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    case CmpIPredicate::sge:
      if (aaa >= bbb)
        return aa.getContext().createLiteralIndex(1);
      break;
    default:
      llvm_unreachable("unknown or illegal (unsigned) compare opeartor");
    }
    return aa.getContext().createLiteralIndex(0);
  };
  F2 valueFct = [&](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    Value compare = aa.getRewriter().create<CmpIOp>(
        aa.getLoc(), comparePred, aa.getValue(), bb.getValue());
    return aa.getContext().createPredicateValueIndex(compare);
  };
  // Cannot have affine results, disable and pass null lambda function.
  return binaryOp(b, false, false, litFct, nullptr, valueFct);
}

// The affine reduction labda function processes the whole list and must init
// the result. Literal and Values treat one operation at a time
/* static*/ IndexExpr IndexExpr::reductionOp(SmallVectorImpl<IndexExpr> &vals,
    F2Self litRed, Flist affineRed, F2Self valueRed) {
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

IndexExpr IndexExpr::operator+(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() + bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() + bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createValueIndex(aa.getRewriter().create<AddIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return binaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator-(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() - bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() - bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createValueIndex(aa.getRewriter().create<SubIOp>(
        aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  return binaryOp(b, false, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator*(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createLiteralIndex(
        aa.getLiteral() * bb.getLiteral());
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    return aa.getContext().createAffineIndex(
        aa.getAffineExpr() * bb.getAffineExpr());
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1)
      return aa.deepCopy();
    return aa.getContext().createValueIndex(aa.getRewriter().create<MulIOp>(
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
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return aa.getContext().createAffineIndex(
          aa.getAffineExpr().floorDiv(bval));
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedFloorDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedFloorDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::ceilDiv(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = ceil((1.0 * aa.getLiteral()) / (1.0 * bb.getLiteral()));
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval == 1)
      return aa.deepCopy();
    if (bval > 1)
      return aa.getContext().createAffineIndex(
          aa.getAffineExpr().ceilDiv(bval));
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedCeilDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedCeilDivIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  // Index b must be a literal.
  return binaryOp(b, true, true, litFct, affineExprFct, valueFct);
}

IndexExpr IndexExpr::operator%(IndexExpr const b) const {
  F2 litFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    int64_t rval = mlir::mod(aa.getLiteral(), bb.getLiteral());
    return aa.getContext().createLiteralIndex(rval);
  };
  F2 affineExprFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    // Operand bb must be a literal.
    int64_t bval = bb.getLiteral();
    if (bval >= 0)
      return aa.getContext().createAffineIndex(aa.getAffineExpr() % bval);
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedRemIOp>(
            aa.getLoc(), aa.getValue(), bb.getValue()));
  };
  F2 valueFct = [](IndexExpr const aa, IndexExpr const bb) -> IndexExpr {
    if (bb.isLiteral() && bb.getLiteral() == 1) {
      return aa.deepCopy();
    }
    return aa.getContext().createValueIndex(
        aa.getRewriter().create<SignedRemIOp>(
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
    return val.getContext().createLiteralIndex(res);
  };
  F3 valueFct = [](IndexExpr const val, IndexExpr const min,
                    IndexExpr const max) {
    IndexExpr res1 = select(val < min, min, val);
    IndexExpr res2 = select(res1 > max, max, res1);
    return res2;
  };

  assert(getContextPtr() == min.getContextPtr() &&
         getContextPtr() == max.getContextPtr() && "incompatible contexts");
  // Literal integer if a, b, and c are literals. Output is not affine (unless
  // all 3 are literals).
  bool resIsLit = isLiteral() && min.isLiteral() && max.isLiteral();
  // We use now use the result of the above determination on whether the new
  // index is literal and/or affine.
  if (resIsLit)
    // Constant, use constant computations.
    return litFct(*this, min, max);
  if (isShapeInferencePass())
    // In shape analysis, if not constant: do noting, aka leave Values & Affine
    // expr undefined.
    return getContext().createQuestionmarkIndex();
  // Use values.
  return valueFct(*this, min, max);
}

/*static*/ IndexExpr IndexExpr::select(IndexExpr const compare,
    IndexExpr const trueVal, IndexExpr const falseVal) {
  assert(compare.getContextPtr() == trueVal.getContextPtr() &&
         compare.getContextPtr() == falseVal.getContextPtr() &&
         "incompatible contexts");
  // When compare result is literal, just feed forward the right value.
  if (compare.isLiteral()) {
    if (compare.getLiteral())
      return trueVal.deepCopy();
    return falseVal.deepCopy();
  }
  // Dynamic value, just set as undefined during shape inference pass.
  if (compare.isShapeInferencePass())
    return compare.getContext().createQuestionmarkIndex();
  // Generate code for the select.
  Value results = compare.getRewriter().create<SelectOp>(compare.getLoc(),
      compare.getValue(), trueVal.getValue(), falseVal.getValue());
  return compare.getContext().createValueIndex(results);
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
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
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
  F2Self valueFct = [](IndexExpr res, IndexExpr const aa) {
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

IndexExpr IndexExpr::operator+(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this + bIndex;
}

IndexExpr IndexExpr::operator-(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this - bIndex;
}

IndexExpr IndexExpr::operator*(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this * bIndex;
}

IndexExpr IndexExpr::operator==(IndexExpr const b) const {
  return compareOp(CmpIPredicate::eq, b);
}

IndexExpr IndexExpr::operator==(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this == bIndex;
}

IndexExpr IndexExpr::operator!=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::ne, b);
}

IndexExpr IndexExpr::operator!=(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this != bIndex;
}

IndexExpr IndexExpr::operator<=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::slt, b);
}

IndexExpr IndexExpr::operator<=(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this <= bIndex;
}

IndexExpr IndexExpr::operator<(IndexExpr const b) const {
  return compareOp(CmpIPredicate::slt, b);
}

IndexExpr IndexExpr::operator<(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this < bIndex;
}

IndexExpr IndexExpr::operator>=(IndexExpr const b) const {
  return compareOp(CmpIPredicate::sge, b);
}

IndexExpr IndexExpr::operator>=(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this >= bIndex;
}

IndexExpr IndexExpr::operator>(IndexExpr const b) const {
  return compareOp(CmpIPredicate::sgt, b);
}

IndexExpr IndexExpr::operator>(int64_t const b) const {
  IndexExpr bIndex = getContext().createLiteralIndex(b);
  return *this > bIndex;
}

IndexExpr IndexExpr::clamp(int64_t min, IndexExpr max) {
  IndexExpr minIndex = getContext().createLiteralIndex(min);
  return clamp(minIndex, max);
}

/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, IndexExpr const falseVal) {
  IndexExpr trueValIndex = compare.getContext().createLiteralIndex(trueVal);
  return select(compare, trueValIndex, falseVal);
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, IndexExpr const trueVal, int64_t const falseVal) {
  IndexExpr falseValIndex = compare.getContext().createLiteralIndex(falseVal);
  return select(compare, trueVal, falseValIndex);
}
/*static*/ IndexExpr IndexExpr::select(
    IndexExpr const compare, int64_t const trueVal, int64_t const falseVal) {
  IndexExpr trueValIndex = compare.getContext().createLiteralIndex(trueVal);
  IndexExpr falseValIndex = compare.getContext().createLiteralIndex(falseVal);
  return select(compare, trueValIndex, falseValIndex);
}

IndexExpr IndexExpr::selectOrSelf(
    IndexExpr const compare, IndexExpr const trueVal) const {
  return select(compare, trueVal, *this);
}

IndexExpr IndexExpr::selectOrSelf(
    IndexExpr const compare, int64_t const trueVal) const {
  return select(compare, trueVal, *this);
}
