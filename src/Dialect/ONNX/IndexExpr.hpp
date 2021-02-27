/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
either determine the actual constant size or a Questionmark (signifying unknown
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

Each IndexExpr must be part of a single scope which holds all of the symbols
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
a new scope can be started for the b) part. The non-affine parts of a)
becomes symbols.

Note that in a computation, all expressions must use IndexExpr originating from
the same scope.

3) Code Sample
==============

3a) Create a scope:

// During shape inference: no rewriter.

  IndexExprContext scope(nullptr, getLoc());

// During lowering.

    IndexExprContext outerloopContex(&rewriter, sliceOp.getLoc());

3b) Computations on IndexExpr

// IN ONNXShapeHelper.cpp

// Get a value from an input operand (either a constant or a value to load).

    startInput = scope.CreateSymbolIndexFromArrayAtIndex(
        op, operandAdaptor.starts(), i);

// Get a dimension from a memref.
    dimInput = scope.CreateDimIndexFromMemref(data, dataShape, ii);

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

// Create a sub-scope for computations inside the loop iteration.

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>
#include <stdint.h>
#include <string>

namespace mlir {
class IndexExpr;
class UndefinedIndexExpr;
class LiteralIndexExpr;
class NonAffineIndexExpr;
class QuestionmarkIndexExpr;
class PredicateIndexExpr;
class AffineIndexExpr;
class DimIndexExpr;
class SymbolIndexExpr;
class IndexExprImpl;

//===----------------------------------------------------------------------===//
// IndexExprKind
//===----------------------------------------------------------------------===//

// Types that an dynamic (i.e. non constant, non literal) IndexExpr can be into.
enum class IndexExprKind {
  // A dynamic value that is not affine (index type).
  NonAffine = 0x00,
  // A dynamic value during shape-inference pass (index type).
  Questionmark = 0x01,
  // A dynamic value that is the result of a compare (1 bit int).
  Predicate = 0x02,
  // A dynamic value that is affine and represented by an AffineExpr.
  Affine = 0x10,
  // A dynamic dimensional identifier for AffineExpr representing a dimension.
  Dim = 0x11,
  // A symbol identifier for an AffineExpr.
  Symbol = 0x12
};

//===----------------------------------------------------------------------===//
// IndexExprScope
//===----------------------------------------------------------------------===//

// Data structure to hold all the IndexExpr in a given scope. A scope define
// a scope during which each of the dynamic dimensions are defined and all of
// the symbols hold constant in that scope.
class IndexExprScope {
  friend class IndexExprImpl;
  friend class IndexExpr;
  friend class LiteralIndexExpr;

public:
  // Constructor for a scope. Top level scope must provide rewriter (possibly
  // null) and location.
  IndexExprScope(ConversionPatternRewriter *rewriter, Location loc);
  // Default constructor can be used for subsequent nested scopes.
  IndexExprScope();
  // While providing the parent scope is not necessary, it is offered to
  // generate more explicit code.
  IndexExprScope(IndexExprScope &explicitEnclosingScope);
  // Destructor which release all IndexExpr associated with this scope.
  ~IndexExprScope();

  // Public getters.
  static IndexExprScope &getCurrentScope();
  ConversionPatternRewriter &getRewriter() const;
  Location getLoc() const { return loc; }
  bool isShapeInferencePass() const { return !rewriter; }

  // Queries and getters.
  bool isCurrentScope();
  bool isEnclosingScope();
  void getDimAndSymbolList(SmallVectorImpl<Value> &list) const;
  int getNumDims() const { return dims.size(); }
  int getNumSymbols() const { return symbols.size(); }

private:
  static IndexExprScope *&getCurrentScopePtr() {
    thread_local IndexExprScope *scope = nullptr; // Thread local, null init.
    return scope;
  }

  // Add a new IndexExprImpl in the scope's container.
  void addIndexExprImpl(IndexExprImpl *obj);

  // Support functions for AffineExpr.
  int addDim(Value const value);
  int addSymbol(Value const value);

  // Support for cached literals.
  static IndexExprImpl *hasCachedLiteralIndexExp(int64_t value);
  static void cacheLiteralIndexExp(int64_t value, IndexExprImpl *obj);


  // Dim and symbol mapping from index to value.
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  // Rewriter, null when during shape inference; otherwise used to create ops.
  ConversionPatternRewriter *rewriter;
  // Location for ops rewriting.
  Location loc;
  // Parent scope (used when creating a child scope).
  IndexExprScope *parentScope;
  // Container of all index expr implementation records, to simplify
  // live range analysis. ALl will be deleted upon scope destruction.
  SmallVector<IndexExprImpl *, 20> container;
  // Cached literals.
  IndexExprImpl *zero, *minusOne, *one;
};

//===----------------------------------------------------------------------===//
// IndexExprImpl
//===----------------------------------------------------------------------===//

// Implementation of the IndexExpr. In nearly all cases, the value described by
// this data structure is constant. Sole exception is during the reduction
// operations. IndexExpr are simply a pointer to this data structure. This data
// structure is allocated in dynamic memory and resides in the scope. It will
// be automaticaly destroyed at the same time as the scope.
struct IndexExprImpl {
  // Public constructor.
  IndexExprImpl();

  // Basic initialization calls.
  void initAsUndefined();
  void initAsQuestionmark();
  void initAsLiteral(int64_t const value);
  void initAsKind(Value const value, IndexExprKind const kind);
  void initAsAffineExpr(AffineExpr const value);
  // Transformative initialization calls.
  void initAsKind(IndexExprImpl const *expr, IndexExprKind const kind);

  // Copy.
  void copy(IndexExprImpl const *other);

  // Data.
  IndexExprScope *scope;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value.
  bool literal;
  // Type of IndexExpr. Literal are by default affine.
  IndexExprKind kind;
  // Integer value, valid when "literal" is true.
  int64_t intLit;
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr.
  AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  Value value;

private:
  // Init for internal use only.
  void init(bool isDefined, bool isIntLit, IndexExprKind type,
      int64_t const intLit, AffineExpr const affineExpr, Value const value);
};

//===----------------------------------------------------------------------===//
// IndexExprExpr
//===----------------------------------------------------------------------===//

// Data structure that is the public interface for IndexExpr. It is a shallow
// data structure that is simply a pointer to the actual data (IndexExprImpl).
class IndexExpr {
public:
  friend class IndexExprScope;
  friend class NonAffineIndexExpr;
  friend class LiteralIndexExpr;
  friend class NonAffineIndexExpr;
  friend class QuestionmarkIndexExpr;
  friend class PredicateIndexExpr;
  friend class AffineIndexExpr;
  friend class DimIndexExpr;
  friend class SymbolIndexExpr;

  // Default and shallow copy constructors.
  IndexExpr() : indexExprObj(nullptr) {} // Undefined index expression.
  IndexExpr(IndexExprImpl *implObj) : indexExprObj(implObj) {}    // Shallow.
  IndexExpr(IndexExpr const &obj) : IndexExpr(obj.getObjPtr()) {} // Shallow.
  // To construct meaningful IndexExpr, use subclasses constructors.
  IndexExpr deepCopy() const;

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

  // Value/values has/have to be literal and satisfy the test.
  bool isLiteralAndIdenticalTo(int64_t b) const;           // Values equal.
  bool isLiteralAndIdenticalTo(IndexExpr const b) const;   // Values equal.
  bool isLiteralAndDifferentThan(int64_t b) const;         // Values unequal.
  bool isLiteralAndDifferentThan(IndexExpr const b) const; // Values unequal.

  // Helpers for IndexExpressions
  static void convertListOfIndexExprToIntegerDim(
      SmallVectorImpl<IndexExpr> &indexExprList,
      SmallVectorImpl<int64_t> &intDimList);

  // Getters.
  int64_t getLiteral() const;
  AffineExpr getAffineExpr() const;
  Value getValue() const;
  IndexExprScope &getScope() const { return *getScopePtr(); }
  IndexExprScope *getScopePtr() const;
  ConversionPatternRewriter &getRewriter() const;
  Location getLoc() const { return getScope().getLoc(); }

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

protected:
  // Copy / private setters.
  IndexExprImpl &getObj() const;
  IndexExprImpl *getObjPtr() const;
  IndexExprKind getKind() const;
  bool isInCurrentScope() const;
  bool canBeUsedInScope() const;

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

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing specific IndexExpr kinds.
//===----------------------------------------------------------------------===//

class UndefinedIndexExpr : public IndexExpr {
public:
  UndefinedIndexExpr();
};

// Subclass to explicitly create non affine IndexExpr
class LiteralIndexExpr : public IndexExpr {
public:
  LiteralIndexExpr(int64_t const value);
  LiteralIndexExpr(IndexExpr const otherIndexExpr);

private:
  void init(int64_t const value);
};

// Subclass to explicitly create non affine IndexExpr
class NonAffineIndexExpr : public IndexExpr {
public:
  NonAffineIndexExpr(Value const value);
  NonAffineIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Questionmark IndexExpr
class QuestionmarkIndexExpr : public IndexExpr {
public:
  QuestionmarkIndexExpr();
  QuestionmarkIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Predicate IndexExpr
class PredicateIndexExpr : public IndexExpr {
public:
  PredicateIndexExpr(Value const value);
  PredicateIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Affine IndexExpr
class AffineIndexExpr : public IndexExpr {
public:
  AffineIndexExpr(AffineExpr const value);
  AffineIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Dim IndexExpr
class DimIndexExpr : public IndexExpr {
public:
  DimIndexExpr(Value const value);
  DimIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create IndexExpr
class SymbolIndexExpr : public IndexExpr {
public:
  SymbolIndexExpr(Value const value);
  SymbolIndexExpr(IndexExpr const otherIndexExpr);
};

//===----------------------------------------------------------------------===//
// Capturing Index Expressions
//===----------------------------------------------------------------------===//

class ArrayValueIndexCapture {
public:
  ArrayValueIndexCapture(Operation *op, Value array);
  ArrayValueIndexCapture(Operation *op, Value array, int64_t defaultLiteral);

  IndexExpr getSymbol(uint64_t i);
  bool getSymbolList(int num, SmallVectorImpl<IndexExpr> &symbolList);

private:
  ArrayValueIndexCapture() { llvm_unreachable("forbidden constructor"); };

  Operation *op;
  Value array;
  int64_t defaultLiteral;
  bool hasDefault;
};

class MemRefBoundIndexCapture {
public:
  MemRefBoundIndexCapture(Value tensorOrMemref);

  IndexExpr getDim(uint64_t i);
  bool getDimList(SmallVectorImpl<IndexExpr> &dimList);

private:
  MemRefBoundIndexCapture() { llvm_unreachable("forbidden constructor"); };
  Value tensorOrMemref;
};

// Additional operators with integer values in first position.
inline IndexExpr operator+(int64_t const a, const IndexExpr b) { return b + a; }
inline IndexExpr operator*(int64_t const a, const IndexExpr b) { return b * a; }
inline IndexExpr operator-(int64_t const a, const IndexExpr b) {
  return LiteralIndexExpr(a) - b;
}

//===----------------------------------------------------------------------===//
// Processing lists
//===----------------------------------------------------------------------===//

template <class INDEXEXPR>
bool getIndexExprListFrom(
    ArrayRef<BlockArgument> inputList, SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  bool successful = true;
  for (auto item : inputList) {
    IndexExpr indexExpr = INDEXEXPR(item);
    if (indexExpr.isUndefined())
      successful = false;
    outputList.emplace_back(indexExpr);
  }
  return successful;
}

//===----------------------------------------------------------------------===//
// Generating Krnl Load / Store
//===----------------------------------------------------------------------===//

struct krnl_load {
  krnl_load(Value memref, SmallVectorImpl<IndexExpr> &indices);
  Value result;
  operator Value() { return result; }
};

struct krnl_store {
  krnl_store(Value val, Value memref, SmallVectorImpl<IndexExpr> &indices);
};

} // namespace mlir
