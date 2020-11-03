//===----------------IndexExpr.hpp - Index expression---------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calcualtions using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

/*

1) IndexExpr
=============

IndexExpr is a single data structure that holds either an integer, an affine
expression or a Value. It is used to compute shape inference, loop bounds, and
index expressions in memory accesses. The main purpose of this data structure is
to use a single function to either determine shape inference or the actual
shapes when allocating/looping during the lowering.

During Shape inference, no code is generated; the IndexExpr will only be used to
eitehr determine the actual constant size or a questionmark (signifying unknown
at compile time).

During lowering, code can be generated, and if fact it must, to fill in the
information that might be missing at compile time. The same IndexExpression
computation are actually used to determine the sizes, indices, and access
functions. Because AffineExpr have several advantages over more generic Value
computations, the IndexExpr maintain computations as AffineExpr as long as
possible. For example a "dim / literal const" is affine and would be represented
as such in the IndexExpr, but if the denominator was in fact another symbol or
computation, such as "dim / shape[3]", then the same IndexExpr would lower
its representation to a Value computation.

IndexExpr can be querried to determine if they are represented at any times
as an Integer Literal, an AffineExpr, or a generic Value. IndexExpr can be
operated on with operations typically found in index computations, namely.

Add, Sub, Mult, Mod, CeilDiv, FloorDiv with the usual mathematical meanings.

Clamp(val, min, max) which forces val to be contained within inclusively min and
exclusively max. Clamp can use AffineMaxOp but the result is affine only when
all inputs are integer literals.

Select(compA, compareOperator, compB, trueVal, falseVal) which corresponds to
"compA operator compb ? trueVal : falseVal". The result can be statically
determined when the compare can be evaluated at compile time.

2) IndexExprContext
======================

Each IndexExpr must be part of a single context which holds all of the symbols
and Dim associated with them. Symbols are variables that are guaranteed to be
constant during the scope of the IndexExpre. Dim are typically runtime
dimensions of memrefs/tensors during computations to determine the shape of a
memref/tensor; or dims are typically the dynamic loop indices inside loop
structures.

A typical pattern is as follow for a kernel that a) determine the shape of the
output and computations, followed by b) the access pattern within the loop
iterations.

In a), the dims are runtime dimensions of inputs memrefs/tensors, and the
symbols are runtime parameters to the functions that are known to be constant.

In b), the dims are dynamic loop indices, and symbols are any of the
computations derived before the loop to compute the output bounds/shape of the
loop iterations.

When all the computations in a) are constant or affine, then the same
IndexExprContext can be reused between a) and b). It is recommended as it
enables bigger AffineExpr. But when the computations in a) are not affine, then
a new context can be started for the b) part. The non-affine parts of a)
becomes symbols.

Note that in a computation, all expressions must use IndexExpr originating from
the same context.

3) Code Sample
==============

3a) Create a context:

// During shape inference: no rewriter.

  IndexExprContext context(nullptr, getLoc());

// During lowering.

    IndexExprContext outerloopContex(&rewriter, sliceOp.getLoc());

3b) Computations on IndexExpr

// IN ONNXShapeHelper.cpp

// Get a value from an input operand (either a constant or a value to load).

    startInput = context.CreateSymbolIndexFromArrayAtIndex(
        op, operandAdaptor.starts(), i);

// Get a dimension from a memref.
    dimInput = context.CreateDimIndexFromMemref(data, dataShape, ii);

// Perform calculations.

    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);

3c) Look at Slice in ONNXOps.cpp on how to use IndexExpr for shape inferences.

// Extract the shape of the output.

  SmallVector<int64_t, 4> outputDims;
  IndexExprContext::GetOutputDimsForType(outputDimIndices, outputDims);
  getResult().setType(RankedTensorType::get(outputDims, elementType));

3d) Look at Slice.cpp on how to use IndexExpr for lowering.

// Create an alloc using dimensions as indices.

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, outputDims);

// Use indices to set loop sizes.

outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    for (int ii = 0; ii < outputRank; ++ii)
      outputLoops.pushBounds(outerloopContex, 0, outputDims[ii]);
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

// Create a sub-context for computations inside the loop iteration.
    IndexExprContext childContext(outerloopContex);

// Create indices with computations for a load.

    for (int ii = 0; ii < outputRank; ++ii) {
      Value loopVal = outputLoops.getInductionVar(ii);
      IndexExpr loopIndex = childContext.createLoopIterIndex(loopVal);
      IndexExpr start = childContext.createSymbolIndexFromParentContext(
          shapeHelper.starts[ii]);
      IndexExpr step = childContext.createSymbolIndexFromParentContext(
          shapeHelper.steps[ii]);
      IndexExpr actualIndex = (step * loopIndex) + start;
      loadIndices.emplace_back(actualIndex.getValue());
    }

*/

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>
#include <stdint.h>
#include <string>

namespace mlir {
class IndexExpr;
class IndexExprImpl;

class IndexExprContext {
public:
  // Constructor for a top level context.
  IndexExprContext(ConversionPatternRewriter *rewriter, Location loc);
  // Constructor for a child context.
  IndexExprContext(IndexExprContext &parentContext);
  // Destructor will also release all IndexExpr associated with this context.
  ~IndexExprContext();

  // IndexExpr builders.
  // Create a copy of the given index (deep copy).
  IndexExpr createIndex(IndexExpr const other);
  // Create an undefined index. Used to indicate undefined results.
  IndexExpr createUndefinedIndex();
  // Create a `?`, which indicates a runtime results during shape inference.
  IndexExpr createQuestionmarkIndex();
  // Create a literal index. Note that all integers are signed in IndexExpr.
  IndexExpr createLiteralIndex(int64_t const val);
  // Create a dimension index, typically associated with a dynamic memsize.
  IndexExpr createDimIndex(Value const val);
  // Create a loop iteration index, typically associated with a loop index.
  IndexExpr createLoopIterIndex(Value const val);
  // Create a symbol index, symbols must be constant in their of the contxt.
  IndexExpr createSymbolIndex(Value const val);
  // Create an index associated with an affine calculation.
  IndexExpr createAffineIndex(AffineExpr const val);
  // Create an index associated with a value (non affine/constant) calculation.
  IndexExpr createValueIndex(Value const val);
  // Create an index associated with a predicate value .
  IndexExpr createPredicateValueIndex(Value const val);
  // Scan a memref_shape[index] to generate an IndexExpr, typically used for
  // dimensions. Generate a literal when the memref dimension is known at
  // compile time, and otherwise a Dim Index.
  IndexExpr createDimIndexFromMemref(
      Value memref, ArrayRef<int64_t> memrefShape, int index);
  // Consider an op with operand "arrayOperand". We find this operand's defining
  // op: if it contains a literal at position "index", we generate an literal
  // IndexExpr; if its a tensor/memref, we load this value. If the index is out
  // of bound, we return an undefine IndexExpr.
  IndexExpr createSymbolIndexFromArrayAtIndex(
      Operation *op, Value array, uint64_t indexInArray);
  // Same as above, but return "defaultLitteral" when there are no defining op
  // or the index is out of bound.
  IndexExpr createSymbolIndexFromArrayAtIndex(Operation *op, Value array,
      uint64_t indexInArray, int64_t defaultLiteral);
  // Create an symbol index in the present context from the parent index in its
  // parent context. The parent index may be a dim/loop iteration index as long
  // as it is contant in the present context.
  IndexExpr createSymbolIndexFromParentContext(IndexExpr const parentIndexExpr);

  // Suppor functions for AffineExpr.
  int addDim(Value const value);
  int addSymbol(Value const value);

  // Querries and getters.
  bool isShapeInferencePass() const { return !rewriter; }
  void getDimAndSymbolList(SmallVectorImpl<Value> &list) const;
  int getNumDims() const { return dims.size(); }
  int getNumSymbols() const { return symbols.size(); }
  ConversionPatternRewriter &getRewriter() const;
  Location getLoc() const { return loc; }

  // Static helper functions.
  // Return true if all IndexExpr in the list are literals.
  bool static areAllLiteral(SmallVectorImpl<IndexExpr> &list);
  // Return true if all IndexExpr in the list are affine.
  bool static areAllAffine(SmallVectorImpl<IndexExpr> &list);
  // Transforms index into literal or -1 for, respectively, literal postive
  // values or runtime values.
  void static getOutputDimsForType(SmallVectorImpl<IndexExpr> &outputIndices,
      SmallVectorImpl<int64_t> &outputDims);

private:
  // Create a new IndexExprImpl and record it in the context's container.
  IndexExprImpl *createIndexExprImpl();
  // Dim and symbol mapping from index to value.
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  // Rewriter, null when during shape inference; otherwise used to create ops.
  ConversionPatternRewriter *rewriter;
  // Location for ops rewriting.
  Location loc;
  // Parent context (used when creating a child context).
  IndexExprContext *parentContext;
  // Container of all index expr implementation records, to simplify
  // live range analysis. ALl will be deleted upon context destruction.
  SmallVector<IndexExprImpl *, 20> container;
  // Enable reuse of constant literals.
  IndexExprImpl *zero, *one, *minusOne;
};

struct IndexExprImpl {
  IndexExprImpl(IndexExprContext *indexExprContext);

  // Higher-level basic initalization calls.
  void initAsUndefined();
  void initAsQuestionmark(IndexExprContext &context);
  void initAsLiteral(IndexExprContext &context, int64_t const val);
  void initAsSymbol(IndexExprContext &context, Value const val);
  void initAsDim(IndexExprContext &context, Value const val);
  void initAsValue(IndexExprContext &context, Value const val);
  void initAsPredicateValue(IndexExprContext &context, Value const val);
  void initAsAffineExpr(IndexExprContext &context, AffineExpr const val);
  // Higher-level initiation calls that extract info.
  void initAsDimFromMemref(IndexExprContext &context, Value memref,
      ArrayRef<int64_t> memrefShape, int index);
  void initAsSymbolFromArrayAtIndex(IndexExprContext &context, Operation *op,
      Value array, uint64_t indexInArray);
  void initAsSymbolFromArrayAtIndex(IndexExprContext &context, Operation *op,
      Value array, uint64_t indexInArray, int64_t defaultLiteral);
  // Lower-level initialization calls.
  void init(IndexExprContext *context, bool newIsDefined, bool newIsIntLit,
      bool newIsAffine, bool newIsSymbol, bool newIsDim,
      bool newIsPredicateType, int64_t const newIntLit,
      AffineExpr const newAffineExpr, Value const newValue);
  void initAsLitQuestionmarkOrValue(IndexExprContext &context, Value const val,
      bool isAffine, bool symbol, bool dim, bool predicateType);

  // Copy.
  void copy(IndexExprImpl const *other);

  // Data.
  IndexExprContext *context;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value.
  bool litteral;
  // Affine indicate that IndedExpr represent an affine expr, which is by
  // definition true for integer literals.
  bool affine;
  // Symbol indicates an IndexExpr representing a symbol; symbols are
  // expressions known to be constant in the context of an AffineExpr.
  bool symbol;
  // Dim indicates an IndexExpr representing a dim in an AffineExpr. Dim's
  // AffineExpr are used to represent tensor/memrefs runtime dimensions or loop
  // iterations, depending on the context in which they are used.
  bool dim;
  // IndexExpr always have an mlir::index type, except when representing the
  // output of a compare, in which case it is an int:1. Result of compares are
  // indicated by the "predicateType" boolean.
  bool predType;
  // Integer value, valid when "litteral" is true.
  int64_t intLit;
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr.
  AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  Value value;

private:
  IndexExprImpl() { llvm_unreachable("illegal"); }
};

class IndexExpr {
public:
  friend class IndexExprContext;

  // Contructors for undefined expressions.
  IndexExpr() : indexExprObj(nullptr) {}
  // Constructor that wraps an IndexExprObj. This is not a deep copy.
  IndexExpr(IndexExprImpl *obj) : indexExprObj(obj) {}

  // Shape inference querries.
  bool isDefined() const;
  bool isUndefined() const;
  bool isLiteral() const;
  bool isQuestionmark() const;
  bool isAffine() const;
  bool isSymbol() const;
  bool isDim() const;
  bool isPredType() const;
  bool isIndexType() const { return !isPredType(); }
  bool isShapeInferencePass() const;
  bool hasContext() const;
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Getters.
  int64_t getLiteral() const;
  AffineExpr getAffineExpr() const;
  Value getValue() const;
  IndexExprContext &getContext() const { return *getContextPtr(); }
  IndexExprContext *getContextPtr() const;
  ConversionPatternRewriter &getRewriter() const;
  Location getLoc() const { return getContext().getLoc(); }

  // Possibly Affine Operations. Return a new IndexExpr
  IndexExpr operator+(IndexExpr const b) const;
  IndexExpr operator+(int64_t const b) const;
  IndexExpr operator-(IndexExpr const b) const;
  IndexExpr operator-(int64_t const b) const;
  IndexExpr operator*(IndexExpr const b) const;
  IndexExpr operator*(int64_t const b) const;
  IndexExpr floorDiv(IndexExpr const b) const;
  IndexExpr ceilDiv(IndexExpr const b) const;
  IndexExpr operator%(IndexExpr const b) const;
  // Compare operations
  IndexExpr operator==(IndexExpr const b) const;
  IndexExpr operator==(int64_t const b) const;
  IndexExpr operator!=(IndexExpr const b) const;
  IndexExpr operator!=(int64_t const b) const;
  IndexExpr operator<=(IndexExpr const b) const;
  IndexExpr operator<=(int64_t const b) const;
  IndexExpr operator<(IndexExpr const b) const;
  IndexExpr operator<(int64_t const b) const;
  IndexExpr operator>=(IndexExpr const b) const;
  IndexExpr operator>=(int64_t const b) const;
  IndexExpr operator>(IndexExpr const b) const;
  IndexExpr operator>(int64_t const b) const;

  // Return a new IndexExpr in the range min/max inclusively.
  IndexExpr clamp(IndexExpr const min, IndexExpr const max) const;
  IndexExpr clamp(int64_t const min, IndexExpr const max);

  // Conditional setting of values: result = cond ? trueVal : falseVal
  static IndexExpr select(IndexExpr const compare, IndexExpr const trueVal,
      IndexExpr const falseVal);
  static IndexExpr select(
      IndexExpr const compare, int64_t const trueVal, IndexExpr const falseVal);
  static IndexExpr select(
      IndexExpr const compare, IndexExpr const trueVal, int64_t const falseVal);
  static IndexExpr select(
      IndexExpr const compare, int64_t const trueVal, int64_t const falseVal);
  // Conditional setting of value: result = cond ? trueVal : *this
  IndexExpr selectOrSelf(
      IndexExpr const compare, IndexExpr const trueVal) const;
  IndexExpr selectOrSelf(IndexExpr const compare, int64_t const trueVal) const;

  static IndexExpr min(SmallVectorImpl<IndexExpr> &vals);
  static IndexExpr max(SmallVectorImpl<IndexExpr> &vals);
  void debugPrint(const std::string &msg) const;

private:
  // Copy / private setters.
  IndexExprImpl &getObj() const;
  IndexExprImpl *getObjPtr() const;
  IndexExpr deepCopy() const;
  // Support for operations: lambda function types.
  typedef std::function<IndexExpr(IndexExpr const, IndexExpr const)> F2;
  typedef std::function<IndexExpr(IndexExpr, IndexExpr const)> F2Self;
  typedef std::function<IndexExpr(IndexExpr, SmallVectorImpl<IndexExpr> &)>
      Flist;
  typedef std::function<IndexExpr(
      IndexExpr const, IndexExpr const, IndexExpr const)>
      F3;
  // Support for operations: common handling for multiple operations.
  IndexExpr binaryOp(IndexExpr const b, bool affineWithLitB,
      bool affineExprCompatible, F2 finteger, F2 faffine, F2 fvalue) const;
  IndexExpr compareOp(CmpIPredicate comparePred, IndexExpr const b) const;
  static IndexExpr reductionOp(SmallVectorImpl<IndexExpr> &vals, F2Self litRed,
      Flist affineRed, F2Self valueRed);
  // Data: pointer to implemented object.
  IndexExprImpl *indexExprObj;
};

// Additional operators with integer first.
inline IndexExpr operator+(int64_t const a, const IndexExpr b) { return b + a; }
inline IndexExpr operator*(int64_t const a, const IndexExpr b) { return b * a; }
inline IndexExpr operator-(int64_t const a, const IndexExpr b) {
  return b.getContext().createLiteralIndex(a) - b;
}

} // namespace mlir
