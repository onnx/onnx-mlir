//===----------------IndexExpr.hpp - Index expression---------------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
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
either determine the actual constant size or a questionmark (signifying unknown
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

IndexExpr can be queried to determine if they are represented at any times
as an Integer Literal, an AffineExpr, or a generic Value. IndexExpr can be
operated on with operations typically found in index computations, namely.

Add, Sub, Mult, Mod, CeilDiv, FloorDiv with the usual mathematical meanings.

Clamp(val, min, max) which forces val to be contained within inclusively min and
exclusively max. Clamp can use AffineMaxOp but the result is affine only when
all inputs are integer literals.

Select(compA, compareOperator, compB, trueVal, falseVal) which corresponds to
"compA operator compB ? trueVal : falseVal". The result can be statically
determined when the compare can be evaluated at compile time.

2) IndexExprContext
======================

Each IndexExpr must be part of a single context which holds all of the symbols
and Dim associated with them. Symbols are variables that are guaranteed to be
constant during the scope of the IndexExprex. Dim are typically runtime
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

// Types that an dynamic (i.e. non constant, non literal) IndexExpr can be into.
enum class IndexExprType {
  // A dynamic value that is not affine (index type).
  NonAffine = 0x00,
  // A dynamic value during shape-inference pass (index type).
  QuestionMark = 0x01,
  // A dynamic value that is the result of a compare (1 bit int).
  Predicate = 0x02,
  // A dynamic value that is affine and represented by an AffineExpr.
  Affine = 0x10,
  // A dynamic dimensional identifier for AffineExpr representing a dimension.
  // There are no differences between the Dim and LoopInduction types, used for
  // readability.
  Dim = 0x11,
  // A dynamic dimensional identifier for an AffineExpr for a loop induction
  // variable.  There are no differences between the Dim and LoopInduction
  // types, used for readability.
  LoopInduction = 0x11,
  // A symbol identifier for an AffineExpr.
  Symbol = 0x12
};

// Data structure to hold all the IndexExpr in a given context. A context define
// a scope during which each of the dynamic dimensions are defined and all of
// the symbols hold constant in that scope.
class IndexExprContext {
public:
  // Constructor for a top level context.
  IndexExprContext(ConversionPatternRewriter *rewriter, Location loc);
  // Constructor for a child context.
  IndexExprContext(IndexExprContext &parentContext);
  // Destructor which release all IndexExpr associated with this context.
  ~IndexExprContext();

  // Individual IndexExpr builders.

  // Create an undefined index, which indicates undefined results.
  IndexExpr createUndefinedIndex();
  // Create an questionmark index, which indicates dynamic results during shape
  // inference.
  IndexExpr createQuestionmarkIndex();
  // Create a literal index. Note that all integers are signed in IndexExpr.
  IndexExpr createLiteralIndex(int64_t const val);
  // Create an index of type Affine, possibly reduced to a literal.
  IndexExpr createAffineIndex(AffineExpr const val);
  // Create an index of type Dim, possibly reduced to a literal.
  IndexExpr createDimIndex(Value const val);
  // Create an index of type Loop Iteration, possibly reduced to a literal.
  IndexExpr createLoopInductionIndex(Value const val);
  // Create an index of type Symbol, possibly reduced to a literal.
  IndexExpr createSymbolIndex(Value const val);
  // Create an index of type Non Affine, possibly reduced to a literal.
  IndexExpr createNonAffineIndex(Value const val);
  // Create an index of type Predicate, possibly reduced to a literal.
  IndexExpr createPredicateIndex(Value const val);
  // Create a copy of the given index (deep copy).
  IndexExpr createIndex(IndexExpr const other);

  // Builder that scan values to derive their IndexExpr.

  // Scan a memref_shape[index] to generate an IndexExpr, typically used for
  // dimensions. Generate a literal when the memref dimension is known at
  // compile time, and otherwise a Dim Index.
  IndexExpr createDimIndexFromShapedType(Value tensorOrMemref, int index);
  // Consider an op with operand "arrayOperand". We find this operand's defining
  // op: if it contains a literal at position "index", we generate an literal
  // IndexExpr; if its a tensor/memref, we issue a question mark during shape
  // inference, and if not, we load this value. If the index is out of bound, we
  // return an undefine IndexExpr.
  IndexExpr createSymbolIndexFromArrayValueAtIndex(
      Operation *op, Value array, uint64_t indexInArray);
  // Same as above, but return "defaultLiteral" when there are no defining op
  // or the index is out of bound.
  IndexExpr createSymbolIndexFromArrayValueAtIndex(Operation *op, Value array,
      uint64_t indexInArray, int64_t defaultLiteral);
  // Create an symbol index in the present context from the parent index in its
  // parent context. The parent index may be a dim/loop iteration index as long
  // as it is contant in the present context.
  IndexExpr createSymbolIndexFromParentContext(IndexExpr const parentIndexExpr);

  // Builder for lists of IndexExpr.

  // Scan a memref_shape to generate a list of IndexExpr, typically used for
  // dimensions. Generate a literal when the memref dimension is known at
  // compile time, and otherwise a Dim Index. Return true if every entry could
  // be successfully processed, false otherwise.
  bool createDimIndicesFromShapedType(
      Value tensorOrMemref, SmallVectorImpl<IndexExpr> &dimIndices);
  // Consider an op with operand "array". We find this operand's defining
  // op. For each of its value,  if it contains a literal, we generate an
  // literal IndexExpr; if its a tensor/memref, we issue a question mark during
  // shape inference, and if not, we load this value. Return true if every entry
  // could be successfully processed, false otherwise.
  bool createSymbolIndicesFromArrayValues(Operation *op, Value array,
      int arraySize, SmallVectorImpl<IndexExpr> &symbolIndices);
  bool createSymbolIndicesFromArrayValues(Operation *op, Value array,
      int arraySize, int64_t defaultLiteral,
      SmallVectorImpl<IndexExpr> &symbolIndices);
  // Create loop induction indices for each of hte block arguments.
  void createLoopInductionIndicesFromArrayValues(
      ArrayRef<BlockArgument> inductionVarArray,
      SmallVectorImpl<IndexExpr> &loopInductionIndices);

  // Code Create for possibly affine load and store. Memref shape is expected to
  // be of the same dimension than the indices array size. Each index expression
  // will be transformed to a value to be used as indices to the memref. When
  // all index expressions are affine, then an affine memory operation is
  // generated. Otherwise, a standard memory operation is generated.
  Value createLoadOp(Value memref, SmallVectorImpl<IndexExpr> &indices);
  void createStoreOp(
      Value val, Value memref, SmallVectorImpl<IndexExpr> &indices);

  // Support functions for AffineExpr.
  int addDim(Value const value);
  int addSymbol(Value const value);

  // Queries and getters.
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
  // Transforms index into literal or -1 for, respectively, literal positive
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

// Implementation of the IndexExpr. In nearly all cases, the value described by
// this data structure is constant. Sole exception is during the reduction
// operations. IndexExpr are simply a pointer to this data structure. This data
// structure is allocated in dynamic memory and resides in the context. It will
// be automaticaly destroyed at the same time as the context.
struct IndexExprImpl {
  // Public constructor.
  IndexExprImpl(IndexExprContext *indexExprContext);

  // Higher-level basic initialization calls.
  void initAsUndefined();
  void initAsQuestionmark(IndexExprContext &context);
  void initAsLiteral(IndexExprContext &context, int64_t const val);
  void initAsType(
      IndexExprContext &context, Value const val, IndexExprType type);
  void initAsAffineExpr(IndexExprContext &newContext, AffineExpr const val);

  // Higher-level initiation calls that extract info.
  void initAsDimFromShapedType(
      IndexExprContext &context, Value tensorOrMemref, int index);
  void initAsSymbolFromArrayAtIndex(IndexExprContext &context, Operation *op,
      Value array, uint64_t indexInArray);
  void initAsSymbolFromArrayAtIndex(IndexExprContext &context, Operation *op,
      Value array, uint64_t indexInArray, int64_t defaultLiteral);

  // Copy.
  void copy(IndexExprImpl const *other);

  // Data.
  IndexExprContext *context;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value.
  bool literal;
  // Type of IndexExpr. Literal are by default affine.
  IndexExprType type;
  // Integer value, valid when "literal" is true.
  int64_t intLit;
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr.
  AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  Value value;

private:
  // Init for internal use only.
  void init(IndexExprContext *context, bool isDefined, bool isIntLit,
      IndexExprType type, int64_t const intLit, AffineExpr const affineExpr,
      Value const value);
  // Default constructor is illegal to make sure context is always defined.
  IndexExprImpl() { llvm_unreachable("illegal"); }
};

// Data structure that is the public interface for IndexExpr. It is a shallow
// data structure that is simply a pointer to the actual data (IndexExprImpl).
class IndexExpr {
public:
  friend class IndexExprContext;

  // Contructors for undefined expressions.
  IndexExpr() : indexExprObj(nullptr) {}
  // Constructor that wraps an IndexExprObj. This is not a deep copy.
  IndexExpr(IndexExprImpl *obj) : indexExprObj(obj) {}

  // Shape inference queries.
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
  // Next calls: value/values has/have to be literal and satisfy the test.
  bool isLiteralAndIdenticalTo(int64_t b) const;           // Values equal.
  bool isLiteralAndIdenticalTo(IndexExpr const b) const;   // Values equal.
  bool isLiteralAndDifferentThan(int64_t b) const;         // Values unequal.
  bool isLiteralAndDifferentThan(IndexExpr const b) const; // Values unequal.

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
  // Compare operations, return a new IndexExpr that is either a literal or a
  // value expression of type predType.
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

  // Return an IndexExpr that is conditionally selected from two "true" and
  // "false" input value, namely: "result = cond ? trueVal : falseVal".
  static IndexExpr select(IndexExpr const compare, IndexExpr const trueVal,
      IndexExpr const falseVal);
  static IndexExpr select(
      IndexExpr const compare, int64_t const trueVal, IndexExpr const falseVal);
  static IndexExpr select(
      IndexExpr const compare, IndexExpr const trueVal, int64_t const falseVal);
  static IndexExpr select(
      IndexExpr const compare, int64_t const trueVal, int64_t const falseVal);
  // Return an IndexExpr that is conditionally selected from the "true" input
  // value, and the "this" value when the test is false: namely "result = cond
  // ? trueVal : *this".
  IndexExpr selectOrSelf(
      IndexExpr const compare, IndexExpr const trueVal) const;
  IndexExpr selectOrSelf(IndexExpr const compare, int64_t const trueVal) const;

  // Return min or max of a list of IndexExpr.
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
      bool affineExprCompatible, F2 fInteger, F2 fAffine, F2 fValue) const;
  IndexExpr compareOp(CmpIPredicate comparePred, IndexExpr const b) const;
  static IndexExpr reductionOp(SmallVectorImpl<IndexExpr> &vals, F2Self litRed,
      Flist affineRed, F2Self valueRed);
  // Data: pointer to implemented object.
  IndexExprImpl *indexExprObj;
};

// Additional operators with integer values in first position.
inline IndexExpr operator+(int64_t const a, const IndexExpr b) { return b + a; }
inline IndexExpr operator*(int64_t const a, const IndexExpr b) { return b * a; }
inline IndexExpr operator-(int64_t const a, const IndexExpr b) {
  return b.getContext().createLiteralIndex(a) - b;
}

} // namespace mlir
