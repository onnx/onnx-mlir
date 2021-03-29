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
expression, or a Value. It is used to compute shape inference, loop bounds, and
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
determined when the compare can be evaluated at compile time. Note that results
of compare operations can either be literals (known at compile time) or runtime.
Runtime IndexExpr have a "predicate" implicit type (1 bit integer) that can only
be used where comparison are used (select, clamp).

Operators have been overloaded, with +, -, * as well as compare operators, and
accept mixtures of integers and index expressions.

Note that the IndexExpr type is used in all operations but the constructors. To
construct an IndexExpr, the IndexExpr subclasses should be used. There are 8
distinct subclasses.

*  UndefinedIndexExpr: used to generate an undefined value. They are typically
used when an error occurred.

* LiteralIndexExpr: used to generate a literal value.

* NonAffineIndexExpr: used to represent a general, non-affine result. Note that
if the result's value can be deduced at compile time, that constant value will
be used internally. So even though a user may attempt to create a non affine
index expression internally, the IndexExpr used to represent that value will be
actually the same as that of an LiteralIndexExpr.

* QuestionmarkIndexExpr: used to represent a runtime value during phases where
we are not interested in generating computations to represent the computed
values.

* PredicateIndexExpr: used to represent the outcome of a compare. Note that if
the outcome can be deduced at compile time, a literal index expression is
actually used to represent that compile time value.

* AffineIndexExpr: expression that the compiler is able to determine as an MLIR
affine expression. See affine dialect for more info about which expressions can
be represented as affine function. In general, though, affine computations
consits of Dim, Symbol, and literals.

* DimIndexExpr: Dim expressions represent "variable" in the affine dialects. In
our case, typical variables are runtime dimensions of memrefs/tensors during
shape inference. Other typical variables are runtime loop indices inside of the
loops. Note Dims are variable within a scope: a tensor dimension may be a
variable/symbol during the shape inference computations, but then treated as
constant/symbols within the loop iterations. Similarity, an index variable can
be considered a variable during its loop, but could be considered as a constant
in an inner-loop nested scope. For more details on the concept of Dim, please
refer to the affine MLIR dialect.

* SymbolIndexExpr: Symbols represent runtime values that are considered as a
constant in a given scope. See affine dialect for more info.


2) IndexExprScope
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
the same scope for Dim, Symbol, and Affine. Literal and non-affine can be from
enclosing scopes as well.

Note that the current scope is kept in a thread private variable and does not
need to be explicitly passed. It will be retrieved from the environment. Which
also means that one can only geneate code in one index scope at a time.


3) Code Sample
==============

3a) Create a scope:

    // During shape inference: no rewriter.

    IndexExprContext scope(nullptr, getLoc());

    // During lowering.

    IndexExprContext outerloopContex(&rewriter, sliceOp.getLoc());

3b) Computations on IndexExpr (examples from processing of ONNXSliceOp)

    // IN ONNXShapeHelper.cpp

    // Get a value from an input operand (either a constant or a value to load).

    ArrayValueIndexCapture startsCapture(genericOp, operandAdaptor.starts());
    SymbolIndexExpr startInput(startsCapture.getSymbol(i));

In the code above, we first capture the (hopefully compile time constant) values
of the "starts" array (limited to 1D arrays at this time). Then we create a
symbol index expression from the captured array value at index "i". When
constant, it will result in a literal. Otherwise it will result in a new Symbol
variable.

    // Get a dimension from a memref.
    MemRefBoundIndexCapture dataBounds(data);
    DimIndexExpr dimInput(dataBounds.getDim(ii));

In the code above, we first capture the (hopefully constant) bounds of hte
memref. We then create a Dim index expression from the memref's "ii" dimension.
When constant, this will result in a literal. Otherwise, it will result in a new
Dim variable.

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
    IndexExpr::getShape(
        shapeHelper.dimsForOutput(0), outputDims);
    getResult().setType(RankedTensorType::get(outputDims, elementType));

In this code, we convert the IndexExpressions back to integer dims (with >=0 for
compile time sizes, -1 for runtime sizes).

3d) Look at Slice.cpp on how to use IndexExpr for lowering.

    // Create an alloc using dimensions as indices.

    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, outputMemRefType, loc, outputDims);

    // Use indices to set loop sizes.

    outputLoops(rewriter, loc, outputRank);
      outputLoops.createDefineOp();
      outputLoops.pushAllBounds(shapeHelper.dimsForOutput(0));
      outputLoops.createIterateOp();
      rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Create a sub-scope for computations inside the loop iteration.

    IndexExprContext childContext(outerloopContex);

    // Create indices with computations for a load.

    for (int ii = 0; ii < outputRank; ++ii) {
      Value inductionVal = outputLoops.getInductionVar(ii);
      DimIndexExpr inductionIndex(inductionVal);
      IndexExpr start = SymbolIndexExpr(shapeHelper.starts[ii]);
      IndexExpr step = SymbolIndexExpr(shapeHelper.steps[ii]);
      loadIndices.emplace_back((step * inductionIndex) + start);
      storeIndices.emplace_back(inductionIndex);
    }

4) Scopes

  Here is an example of how scopes work:

    IndexExprScope outerScope; // outer scope
      DimIndexExpr d1(dataBounds.getDim(2); // outer scope variable
      // ...
      {
         IndexExprScope innerScope;
         SymbolIndexExpr s1(d1); // in the inner scope, make a symbol out of the
                                 // outer scope dim index expr d1.
        // ...
        // innerScope is deleted, and its enclosing scope becomes
        // the active scope again.
    }

    // Back to the outer scope

5) Additional infrastructure

   ArrayValueIndexCapture allows us to read 1D arrays and generate symbols out
of them expressions that are either literals or runtime values (symbols).

   MemRefBoundIndexCapture allows us to read memref or tensor 1D descriptors and
generate out of them expressions that are either literals or runtime values
(dims).

Note that in both case, runtime values may be "questionmarks" during the shape
inference part as no code may be generated during such phases.

krnl_load / krnl_store allow us to generate kernel loads or store where the
indices are represented by index expressions.

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
  // null if we cannot geneate code at this time) and location.
  IndexExprScope(OpBuilder *rewriter, Location loc);
  IndexExprScope(OpBuilder &rewriter, Location loc);
  // Constructor for subsequent nested scopes. Providing enclosing scope is not
  // necessary; it is provided for convenience if a user prefer to name the
  // enclosing scope explicitly.
  IndexExprScope();
  IndexExprScope(IndexExprScope &explicitEnclosingScope);
  // Destructor which release all IndexExpr associated with this scope.
  ~IndexExprScope();

  // Public getters.
  static IndexExprScope &getCurrentScope();
  OpBuilder &getRewriter() const;
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

  // Dim and symbol mapping from index to value.
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  // Rewriter, null when during shape inference; otherwise used to create ops.
  OpBuilder *rewriter;
  // Location for ops rewriting.
  Location loc;
  // Parent scope (used when creating a child scope).
  IndexExprScope *parentScope;
  // Container of all index expr implementation records, to simplify
  // live range analysis. ALl will be deleted upon scope destruction.
  SmallVector<IndexExprImpl *, 20> container;
};

//===----------------------------------------------------------------------===//
// IndexExprExpr
//===----------------------------------------------------------------------===//

// Data structure that is the public interface for IndexExpr. It is a shallow
// data structure that is simply a pointer to the actual data (IndexExprImpl).
class IndexExpr {

  friend class IndexExprScope;
  friend class NonAffineIndexExpr;
  friend class LiteralIndexExpr;
  friend class NonAffineIndexExpr;
  friend class QuestionmarkIndexExpr;
  friend class PredicateIndexExpr;
  friend class AffineIndexExpr;
  friend class DimIndexExpr;
  friend class SymbolIndexExpr;

public:
  // Default and shallow copy constructors. Index expressions are usually built
  // using the subclasses listed as friend above.
  IndexExpr() : indexExprObj(nullptr) {} // Undefined index expression.
  IndexExpr(IndexExprImpl *implObj) : indexExprObj(implObj) {}     // Shallow.
  IndexExpr(IndexExpr const &obj) : IndexExpr(obj.indexExprObj) {} // Shallow.
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
  bool isIndexType() const;
  bool isShapeInferencePass() const {
    return getScope().isShapeInferencePass();
  }
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Value/values has/have to be literal and satisfy the test.
  bool isLiteralAndIdenticalTo(int64_t b) const;           // Values equal.
  bool isLiteralAndIdenticalTo(IndexExpr const b) const;   // Values equal.
  bool isLiteralAndDifferentThan(int64_t b) const;         // Values unequal.
  bool isLiteralAndDifferentThan(IndexExpr const b) const; // Values unequal.

  // Getters.
  IndexExprScope &getScope() const { return *getScopePtr(); }
  OpBuilder &getRewriter() const { return getScope().getRewriter(); }
  Location getLoc() const { return getScope().getLoc(); }
  int64_t getLiteral() const;
  AffineExpr getAffineExpr() const;
  void getAffineMapAndOperands(
      AffineMap &map, SmallVectorImpl<Value> &operands) const;
  Value getValue() const;

  // Helpers for list of IndexExpressions
  static void getShape(SmallVectorImpl<IndexExpr> &indexExprList,
      SmallVectorImpl<int64_t> &intDimList);
  static void getValues(ArrayRef<IndexExpr> indexExprArray,
      SmallVectorImpl<Value> &valueList);

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
  // Compare and/or, only for predicate index expression, which are 1 bit 0
  // or 1. Because we use the comparison in non-branching code here, we perform
  // the "arithmetic and/or". Logical disjunctive comparison must be made
  // manually with branching code.
  IndexExpr operator&(IndexExpr const b) const;
  IndexExpr operator|(IndexExpr const b) const;
  IndexExpr operator!() const;

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
  static IndexExpr min(IndexExpr const first, IndexExpr const second);
  static IndexExpr min(IndexExpr const first, int64_t const second);
  static IndexExpr max(SmallVectorImpl<IndexExpr> &vals);
  static IndexExpr max(IndexExpr const first, IndexExpr const second);
  static IndexExpr max(IndexExpr const first, int64_t const second);

  // Debug (enable using DEBUG=1 at top of file).
  void debugPrint(const std::string &msg) const;

  bool retrieveAffineMinMax(
      bool &isMin, SmallVectorImpl<Value> &vals, AffineMap &map) const;

protected:
  // Private queries.
  bool hasScope() const;
  bool isInCurrentScope() const;
  bool canBeUsedInScope() const;
  // Copy / private setters.
  IndexExprScope *getScopePtr() const;
  IndexExprImpl &getObj() const;
  IndexExprImpl *getObjPtr() const;
  IndexExprKind getKind() const;

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

// Subclass to explicitely create undefined index expressions, typically used to
// return invalid values.
class UndefinedIndexExpr : public IndexExpr {
public:
  UndefinedIndexExpr();
};

// Subclass to explicitly create affine literal IndexExpr. For predicate literal
// values, use PredicateIndexExpr(true) or PredicateIndexExpr(false).
class LiteralIndexExpr : public IndexExpr {
public:
  LiteralIndexExpr() : IndexExpr() {}    // Make undefined.
  LiteralIndexExpr(int64_t const value); // Make an index constant value.
  LiteralIndexExpr(IndexExpr const otherIndexExpr);

private:
  void init(int64_t const value);
};

// Subclass to explicitly create non affine IndexExpr.
class NonAffineIndexExpr : public IndexExpr {
public:
  NonAffineIndexExpr() : IndexExpr() {} // Make undefined expression.
  NonAffineIndexExpr(Value const value);
  NonAffineIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Questionmark IndexExpr.
class QuestionmarkIndexExpr : public IndexExpr {
public:
  QuestionmarkIndexExpr();
  QuestionmarkIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Predicate IndexExpr.
class PredicateIndexExpr : public IndexExpr {
public:
  PredicateIndexExpr() : IndexExpr() {} // Make undefined predicate expression.
  PredicateIndexExpr(bool const value); // Make a predicate constant value.
  PredicateIndexExpr(Value const value);
  PredicateIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Affine IndexExpr.
class AffineIndexExpr : public IndexExpr {
public:
  AffineIndexExpr() : IndexExpr() {} // Make undefined expression.
  AffineIndexExpr(AffineExpr const value);
  AffineIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create Dim IndexExpr.
class DimIndexExpr : public IndexExpr {
public:
  DimIndexExpr() : IndexExpr() {} // Make undefined expression.
  DimIndexExpr(Value const value);
  DimIndexExpr(IndexExpr const otherIndexExpr);
};

// Subclass to explicitly create IndexExpr.
class SymbolIndexExpr : public IndexExpr {
public:
  SymbolIndexExpr() : IndexExpr() {} // Make undefined expression.
  SymbolIndexExpr(Value const value);
  SymbolIndexExpr(IndexExpr const otherIndexExpr);
};

//===----------------------------------------------------------------------===//
// Additional operators with integer values in first position
//===----------------------------------------------------------------------===//

inline IndexExpr operator+(int64_t const a, const IndexExpr b) { return b + a; }
inline IndexExpr operator*(int64_t const a, const IndexExpr b) { return b * a; }
inline IndexExpr operator-(int64_t const a, const IndexExpr b) {
  return LiteralIndexExpr(a) - b;
}

//===----------------------------------------------------------------------===//
// Capturing Index Expressions
//===----------------------------------------------------------------------===//

// Capture array of values given by an operand. Will find its definitition and
// use it locate its constant values, or load dynamically if they are not
// constant.
class ArrayValueIndexCapture {
public:
  ArrayValueIndexCapture(Operation *op, Value array);
  ArrayValueIndexCapture(Operation *op, Value array, int64_t defaultLiteral);

  IndexExpr getSymbol(uint64_t i);
  void getSymbolList(int num, SmallVectorImpl<IndexExpr> &symbolList);

private:
  ArrayValueIndexCapture() { llvm_unreachable("forbidden constructor"); };

  Operation *op;
  Value array;
  int64_t defaultLiteral;
  bool hasDefault;
};

// Capture array of values given by attributes.
class ArrayAttributeIndexCapture {
public:
  ArrayAttributeIndexCapture(ArrayAttr array);
  ArrayAttributeIndexCapture(ArrayAttr array, int64_t defaultLiteral);

  IndexExpr getLiteral(uint64_t i);
  int64_t size() { return arraySize; }

private:
  ArrayAttributeIndexCapture() { llvm_unreachable("forbidden constructor"); };

  ArrayAttr array;
  int64_t arraySize;
  int64_t defaultLiteral;
  bool hasDefault;
};

// Capture memory bounds give by a tensor or memref. Locate its shape, return
// constant values when available or generate the appropriate dim operation when
// they are not constant at compile time.
class MemRefBoundIndexCapture {
public:
  MemRefBoundIndexCapture(Value tensorOrMemref);

  int64_t getRank() { return memRank; }
  IndexExpr getLiteral(uint64_t i); // Assert if bound is not compile time.
  IndexExpr getDim(uint64_t i);
  IndexExpr getSymbol(uint64_t i);
  void getLiteralList(SmallVectorImpl<IndexExpr> &literalList);
  void getDimList(SmallVectorImpl<IndexExpr> &dimList);
  void getSymbolList(SmallVectorImpl<IndexExpr> &symbolList);

private:
  MemRefBoundIndexCapture() { llvm_unreachable("forbidden constructor"); };

  template <class INDEX>
  IndexExpr get(uint64_t i);
  template <class INDEX>
  void getList(SmallVectorImpl<IndexExpr> &dimList);

  Value tensorOrMemref;
  int64_t memRank;
};

//===----------------------------------------------------------------------===//
// Make IndexExpressions of a given type from provided input list/range
//===----------------------------------------------------------------------===//

template <class INDEXEXPR>
void getIndexExprList(
    ArrayRef<BlockArgument> inputList, SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : inputList)
    outputList.emplace_back(INDEXEXPR(item));
}

template <class INDEXEXPR>
void getIndexExprList(
    ValueRange range, SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : range)
    outputList.emplace_back(INDEXEXPR(item));
}

template <class INDEXEXPR>
void getIndexExprList(SmallVectorImpl<IndexExpr> &inputList,
    SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : inputList)
    outputList.emplace_back(INDEXEXPR(item));
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
