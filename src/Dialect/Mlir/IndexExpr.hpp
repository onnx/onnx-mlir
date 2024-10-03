/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------IndexExpr.hpp - Index expression---------------------=== //
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INDEX_EXPR_H
#define ONNX_MLIR_INDEX_EXPR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <functional>
#include <string>

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

During lowering, code can be generated, and in fact it must, to fill in the
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
consist of Dim, Symbol, and literals.

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

Note that in most cases, IndexExpr are built from MLIR Values, Types, or
Attributes. See Section 5) below for references.

2) IndexExprScope
======================

Each IndexExpr must be part of a single scope which holds all of the symbols
and Dim associated with them. Symbols are variables that are guaranteed to be
constant during the scope of the IndexExpr. Dim are typically runtime
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
IndexExprScope can be reused between a) and b). It is recommended as it
enables bigger AffineExpr. But when the computations in a) are not affine, then
a new scope can be started for the b) part. The non-affine parts of a)
becomes symbols.

Note that in a computation, all expressions must use IndexExpr originating from
the same scope for Dim, Symbol, and Affine. Literal and non-affine can be from
enclosing scopes as well.

Note that the current scope is kept in a thread private variable and does not
need to be explicitly passed. It will be retrieved from the environment. Which
also means that one can only generate code in one index scope at a time.


3) Code Sample
==============

3a) Create a scope:

During shape inference: no rewriter.

```
  IndexExprScope scope(nullptr, getLoc());
```

During lowering: rewriter/builder provided.

```
  IndexExprScope outerLoopContext(&rewriter, sliceOp.getLoc());
```

Note that the presence of a rewriter/builder in the constructor is what
indicates whether we are in "shape inference" or "code gen/lowering" mode.

3b) Computations on IndexExpr (examples from processing of ONNXSliceOp)

In src/Dialect/ONNX/ONNXOps/Tensor/Slice.cpp

```
    // Get a value from an input operand (either a constant or a value to load).
    SymbolIndexExpr startInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.starts(), i);
    if (startInput.isUndefined())
      return op->emitError("start input parameter could not be processed");
```

In the code above, we get a symbol index expression from the "starts" array
(limited to 1D arrays at this time). Specifically, we create a symbol index
expression from the captured array value at index "i". When constant, it will
result in a literal. Otherwise it will result in a new Symbol variable.

The `createIE` is a IndexExprBuilder that is used to scan MLIR structures to
generate index expressions. See Section 5 for references.

```
    // Get a dimension from a memref.
    DimIndexExpr dimInput = createIE->getShapeAsDim(data, ii);
```

In the code above, we create a Dim index expression from the memref's "ii"
dimension. When constant, this will result in a literal. Otherwise, it will
result in a new Dim variable.

```
    // Perform calculations.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);
```

3c) Look at Slice.cpp on how to use IndexExpr for lowering.

```
    // Create an alloc using dimensions as indices.

    Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);

    // Use indices to set loop sizes.

    outputLoops(rewriter, loc, outputRank);
    outputLoops.createDefineOp();
    outputLoops.pushAllBounds(shapeHelper.dimsForOutput());
    outputLoops.createIterateOp();
    rewriter.setInsertionPointToStart(outputLoops.getIterateBlock());

    // Create a sub-scope for computations inside the loop iteration.

    IndexExprScope childContext(outerLoopContext);

    // Create indices with computations for a load.

    for (int ii = 0; ii < outputRank; ++ii) {
      Value inductionVal = outputLoops.getInductionVar(ii);
      DimIndexExpr inductionIndex(inductionVal);
      IndexExpr start = SymIE(shapeHelper.starts[ii]);
      IndexExpr step = SymIE(shapeHelper.steps[ii]);
      loadIndices.emplace_back((step * inductionIndex) + start);
      storeIndices.emplace_back(inductionIndex);
    }
```

4) Scopes

  Here is an example of how scopes work:

  ```
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
  ```

5) Additional infrastructure

  Look at the IndexExprBuilder on how to create index expressions from MLIR
  Attributes and Values. There are basically 3 ways to get index expressions:

  * From attributes, see `getIntFromArrayAsLiteral` for example.
  * From values, see `getIntFromArrayAsSymbol` for example.
  * From shapes, see `getShapeAsDim` for example.

  Interface is described in src/Dialect/Mlir/IndexExprBuilder.hpp

*/

/* Dialect use in Index Expression stack (when generating ops)
   IndexExpr.cpp:
     Arithmetic: cmp, add, sub, mul, floor/ceil div, rem, and, or, select
     Affine: affine min/max
   IndexExprImpl.cpp:
     Arithmetic: constant, constant index, index cast.
     Affine: affine apply
*/

namespace onnx_mlir {

struct DialectBuilder;

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
  // null if we cannot generate code at this time) and location.
  IndexExprScope(mlir::OpBuilder *rewriter, mlir::Location loc);
  IndexExprScope(const DialectBuilder &db);
  // Constructor for subsequent nested scopes. Providing enclosing scope is
  // technically not necessary (nullptr can be passed); it is used to allow a
  // user to explicitly name the enclosing scope.
  IndexExprScope(mlir::OpBuilder *rewriter, IndexExprScope *enclosingScope);
  IndexExprScope(const DialectBuilder &db, IndexExprScope *enclosingScope);
  // Destructor which release all IndexExpr associated with this scope.
  virtual ~IndexExprScope();

  IndexExprScope() = delete;
  IndexExprScope(const IndexExprScope &) = delete;
  IndexExprScope(IndexExprScope &&) = delete;
  IndexExprScope &operator=(const IndexExprScope &) = delete;
  IndexExprScope &operator=(IndexExprScope &&) = delete;

  // Public getters.
  static IndexExprScope &getCurrentScope();
  mlir::OpBuilder &getRewriter() const;
  mlir::OpBuilder *getRewriterPtr() const;
  mlir::Location getLoc() const { return loc; }
  bool isShapeInferencePass() const { return !rewriter; }

  // Queries and getters.
  bool isCurrentScope() const;
  bool isEnclosingScope() const;
  void getDimAndSymbolList(llvm::SmallVectorImpl<mlir::Value> &list) const;
  int getNumDims() const { return dims.size(); }
  int getNumSymbols() const { return symbols.size(); }

  // Debug (enable using --debug-only=index-expr, for example).
  void debugPrint(const std::string &msg) const;

private:
  static IndexExprScope *&getCurrentScopePtr() {
    thread_local IndexExprScope *scope = nullptr; // Thread local, null init.
    return scope;
  }

  // Add a new IndexExprImpl in the scope's container.
  void addIndexExprImpl(IndexExprImpl *obj);

  // Support functions for AffineExpr.
  int indexInList(llvm::SmallVectorImpl<mlir::Value> const &list,
      mlir::Value const &value) const;
  int addDim(mlir::Value const value);
  int addSymbol(mlir::Value const value);

  // Dim and symbol mapping from index to value.
  llvm::SmallVector<mlir::Value, 4> dims;
  llvm::SmallVector<mlir::Value, 4> symbols;
  // Rewriter, null when during shape inference; otherwise used to create ops.
  mlir::OpBuilder *rewriter;
  // Parent scope (used when creating a child scope).
  IndexExprScope *parentScope;
  // Location for ops rewriting.
  mlir::Location loc;
  // Container of all index expr implementation records, to simplify
  // live range analysis. ALl will be deleted upon scope destruction.
  llvm::SmallVector<IndexExprImpl *, 20> container;
};
#define DETAILED_DEBUG_OF_SCOPE 0

//===----------------------------------------------------------------------===//
// IndexExprExpr
//===----------------------------------------------------------------------===//

using DimsExpr = llvm::SmallVector<IndexExpr, 4>;
using DimsExprRef = mlir::ArrayRef<IndexExpr>;

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
  IndexExpr() = default;
  IndexExpr(IndexExprImpl *implObj) : indexExprObj(implObj) {}     // Shallow.
  IndexExpr(IndexExpr const &obj) : IndexExpr(obj.indexExprObj) {} // Shallow.
  IndexExpr(IndexExpr &&obj) noexcept : indexExprObj(obj.indexExprObj) {
    obj.indexExprObj = nullptr;
  }
  virtual ~IndexExpr() = default;
  IndexExpr &operator=(const IndexExpr &) = default;
  IndexExpr &operator=(IndexExpr &&) = default;

  // To construct meaningful IndexExpr, use subclasses constructors.
  IndexExpr deepCopy() const;

  // Shape inference queries.
  bool isDefined() const;
  bool isUndefined() const;
  bool isLiteral() const;
  bool isFloat() const;
  bool areFloat(IndexExpr b) const; // True if both float; assert if different.
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
  bool isLiteralAndIdenticalTo(int b) const;               // Values equal.
  bool isLiteralAndIdenticalTo(int64_t b) const;           // Values equal.
  bool isLiteralAndIdenticalTo(double b) const;            // Values equal.
  bool isLiteralAndIdenticalTo(IndexExpr const b) const;   // Values equal.
  bool isLiteralAndDifferentThan(int b) const;             // Values unequal.
  bool isLiteralAndDifferentThan(int64_t b) const;         // Values unequal.
  bool isLiteralAndDifferentThan(double b) const;          // Values unequal.
  bool isLiteralAndDifferentThan(IndexExpr const b) const; // Values unequal.
  bool isLiteralAndGreaterThan(int b) const;               // Values greater.
  bool isLiteralAndGreaterThan(int64_t b) const;           // Values greater.
  bool isLiteralAndGreaterThan(double b) const;            // Values greater.
  bool isLiteralAndGreaterThan(IndexExpr const b) const;   // Values greater.
  bool isLiteralAndSmallerThan(int b) const;               // Values smaller.
  bool isLiteralAndSmallerThan(int64_t b) const;           // Values smaller.
  bool isLiteralAndSmallerThan(double b) const;            // Values smaller.
  bool isLiteralAndSmallerThan(IndexExpr const b) const;   // Values smaller.
  // Test if all element in list are literals.
  static bool isLiteral(mlir::ArrayRef<IndexExpr> list);
  static bool isNonNegativeLiteral(mlir::ArrayRef<IndexExpr> list);

  // Getters.
  IndexExprScope &getScope() const { return *getScopePtr(); }
  mlir::OpBuilder &getRewriter() const { return getScope().getRewriter(); }
  mlir::Location getLoc() const { return getScope().getLoc(); }
  int64_t getLiteral() const;
  double getFloatLiteral() const;
  int64_t getQuestionmark() const;
  mlir::AffineExpr getAffineExpr() const;
  void getAffineMapAndOperands(
      mlir::AffineMap &map, llvm::SmallVectorImpl<mlir::Value> &operands) const;
  mlir::Value getValue() const;
  int64_t getShape(bool uniqueQuestionMark = false) const;

  // Helpers for list of IndexExpressions: given a (list of) IndexExpr, provide
  // the (list of) Shape/Value/OpFoldResult corresponding to the original (list
  // of) IndexExpr.
  static void getLiteral(mlir::ArrayRef<IndexExpr> indexExprArray,
      llvm::SmallVectorImpl<int64_t> &intList);
  static void getShape(mlir::ArrayRef<IndexExpr> indexExprArray,
      llvm::SmallVectorImpl<int64_t> &intDimList,
      bool uniqueQuestionMark = false);
  static void getDynSymbols(
      mlir::ArrayRef<IndexExpr> indexExprArray,        // Input list.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols); // Symbol for dyn ref.
  static void getValues(mlir::ArrayRef<IndexExpr> indexExprArray,
      llvm::SmallVectorImpl<mlir::Value> &valueList);
  static void getOpOrFoldResults(mlir::ArrayRef<IndexExpr> indexExprArray,
      llvm::SmallVectorImpl<mlir::OpFoldResult> &resList);
  static void getAffineMapAndOperands(mlir::ArrayRef<IndexExpr> indexExprArray,
      mlir::AffineMap &map, llvm::SmallVectorImpl<mlir::Value> &operands);

  // Possibly Affine Operations. Return a new IndexExpr
  IndexExpr operator+(IndexExpr const b) const;
  IndexExpr operator+(int64_t const b) const;
  IndexExpr operator-(IndexExpr const b) const;
  IndexExpr operator-(int64_t const b) const;
  IndexExpr operator*(IndexExpr const b) const;
  IndexExpr operator*(int64_t const b) const;
  IndexExpr floorDiv(IndexExpr const b) const;  // Int only; float use `/`.
  IndexExpr floorDiv(int64_t const b) const;    // Int only; float use `/`.
  IndexExpr ceilDiv(IndexExpr const b) const;   // Int only; float use `/`.
  IndexExpr ceilDiv(int64_t const b) const;     // Int only; float use `/`.
  IndexExpr floor() const;                      // Float only.
  IndexExpr ceil() const;                       // Float only.
  IndexExpr operator/(IndexExpr const b) const; // Float; int ceil/floorDiv.
  IndexExpr operator%(IndexExpr const b) const; // Int only.
  IndexExpr operator%(int64_t const b) const;   // Int only.
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
  static IndexExpr min(mlir::ArrayRef<IndexExpr> vals);
  static IndexExpr min(IndexExpr const first, IndexExpr const second);
  static IndexExpr min(IndexExpr const first, int64_t const second);
  static IndexExpr max(mlir::ArrayRef<IndexExpr> vals);
  static IndexExpr max(IndexExpr const first, IndexExpr const second);
  static IndexExpr max(IndexExpr const first, int64_t const second);

  bool retrieveAffineMinMax(bool &isMin,
      llvm::SmallVectorImpl<mlir::Value> &vals, mlir::AffineMap &map) const;

  // Float related op. Note that float are currently MLIR f32.
  IndexExpr convertToFloat() const; // Int input only.
  IndexExpr convertToIndex() const; // Float input only.

  // Debug (enable running with --debug-only=index-expr, for example).
  void debugPrint(const std::string &msg) const;
  static void debugPrint(
      const std::string &msg, const mlir::ArrayRef<IndexExpr> list);

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
  using F1 = std::function<IndexExpr(IndexExpr const)>;
  using F2 = std::function<IndexExpr(IndexExpr const, IndexExpr const)>;
  using F2Self = std::function<IndexExpr(IndexExpr, IndexExpr const)>;
  using Flist = std::function<IndexExpr(IndexExpr, mlir::ArrayRef<IndexExpr>)>;
  using F3 = std::function<IndexExpr(
      IndexExpr const, IndexExpr const, IndexExpr const)>;
  // Support for operations: common handling for multiple operations.
  // When hasNeutralA, if a==*this is neutral literal value, then result is b.
  // When hasNeutralB, if b is neutral literal value, then result is a. We
  // represent the neutral value using a double, as it's value (0.0 or 1.0) can
  // be safely converted to int too.
  IndexExpr unaryOp(
      bool resIsFloat, F1 litFct, F1 affineExprFct, F1 valueFct) const;
  // Res is float is the same as a & b.
  IndexExpr binaryOp(IndexExpr const b, bool propagateIntoMinMax,
      bool affineWithLitB, bool hasNeutralA, bool hasNeutralB,
      double neutralVal, F2 fInteger, F2 fAffine, F2 fValue) const;
  IndexExpr compareOp(
      mlir::arith::CmpIPredicate comparePred, IndexExpr const b) const;
  IndexExpr compareOp(
      mlir::arith::CmpFPredicate comparePred, IndexExpr const b) const;
  static IndexExpr reductionOp(mlir::ArrayRef<IndexExpr> vals, F2Self litRed,
      Flist affineRed, F2Self valueRed);
  // Data: pointer to implemented object.
  IndexExprImpl *indexExprObj = nullptr;
};

//===----------------------------------------------------------------------===//
// IndexExpr Subclasses for constructing specific IndexExpr kinds.
//===----------------------------------------------------------------------===//

// Subclass to explicitly create undefined index expressions, typically used to
// return invalid values.
class UndefinedIndexExpr : public IndexExpr {
public:
  UndefinedIndexExpr();
  virtual ~UndefinedIndexExpr() = default;
};

// Subclass to explicitly create affine literal IndexExpr. For predicate literal
// values, use PredicateIndexExpr(true) or PredicateIndexExpr(false).
class LiteralIndexExpr : public IndexExpr {
public:
  LiteralIndexExpr() = default;
  LiteralIndexExpr(int const value);          // Make an index constant value.
  LiteralIndexExpr(unsigned int const value); // Make an index constant value.
  LiteralIndexExpr(int64_t const value);      // Make an index constant value.
  LiteralIndexExpr(uint64_t const value);     // Make an index constant value.
  LiteralIndexExpr(double const value);       // Make an index constant value.
  LiteralIndexExpr(IndexExpr const &o);
  LiteralIndexExpr(UndefinedIndexExpr const &o);
  LiteralIndexExpr(LiteralIndexExpr const &o);
  LiteralIndexExpr(NonAffineIndexExpr const &o);
  LiteralIndexExpr(QuestionmarkIndexExpr const &o);
  LiteralIndexExpr(PredicateIndexExpr const &o);
  LiteralIndexExpr(AffineIndexExpr const &o);
  LiteralIndexExpr(DimIndexExpr const &o);
  LiteralIndexExpr(SymbolIndexExpr const &o);

  LiteralIndexExpr(LiteralIndexExpr &&) = delete;
  virtual ~LiteralIndexExpr() = default;
  LiteralIndexExpr &operator=(const LiteralIndexExpr &) = delete;
  LiteralIndexExpr &operator=(LiteralIndexExpr &&) = delete;

private:
  void init(int64_t const value);
  void init(double const value);
};

// Subclass to explicitly create non affine IndexExpr.
class NonAffineIndexExpr : public IndexExpr {
public:
  NonAffineIndexExpr() = default;
  NonAffineIndexExpr(mlir::Value const value);
  NonAffineIndexExpr(IndexExpr const &o);
  NonAffineIndexExpr(UndefinedIndexExpr const &o);
  NonAffineIndexExpr(LiteralIndexExpr const &o);
  NonAffineIndexExpr(NonAffineIndexExpr const &o);
  NonAffineIndexExpr(QuestionmarkIndexExpr const &o);
  NonAffineIndexExpr(PredicateIndexExpr const &o);
  NonAffineIndexExpr(AffineIndexExpr const &o);
  NonAffineIndexExpr(DimIndexExpr const &o);
  NonAffineIndexExpr(SymbolIndexExpr const &o);

  NonAffineIndexExpr(NonAffineIndexExpr &&) = delete;
  virtual ~NonAffineIndexExpr() = default;
  NonAffineIndexExpr &operator=(const NonAffineIndexExpr &) = delete;
  NonAffineIndexExpr &operator=(NonAffineIndexExpr &&) = delete;

private:
  NonAffineIndexExpr(IndexExprImpl *otherObjPtr);
};

// Subclass to explicitly create Questionmark IndexExpr.
class QuestionmarkIndexExpr : public IndexExpr {
public:
  QuestionmarkIndexExpr() = default;
  QuestionmarkIndexExpr(bool isFloat);
  // Construct a question mark for an unknown dimension in a Tensor/Memref.
  // This constructor is needed for symbolic shape analysis where each
  // question mark is assigned to a unique value hashed from the given
  // tensorOrMemref and dimension index.
  QuestionmarkIndexExpr(mlir::Value tensorOrMemref, int64_t index);
  QuestionmarkIndexExpr(IndexExpr const &o);
  QuestionmarkIndexExpr(UndefinedIndexExpr const &o);
  QuestionmarkIndexExpr(LiteralIndexExpr const &o);
  QuestionmarkIndexExpr(NonAffineIndexExpr const &o);
  QuestionmarkIndexExpr(QuestionmarkIndexExpr const &o);
  QuestionmarkIndexExpr(PredicateIndexExpr const &o);
  QuestionmarkIndexExpr(AffineIndexExpr const &o);
  QuestionmarkIndexExpr(DimIndexExpr const &o);
  QuestionmarkIndexExpr(SymbolIndexExpr const &o);

  QuestionmarkIndexExpr(QuestionmarkIndexExpr &&) = delete;
  virtual ~QuestionmarkIndexExpr() = default;
  QuestionmarkIndexExpr &operator=(const QuestionmarkIndexExpr &) = delete;
  QuestionmarkIndexExpr &operator=(QuestionmarkIndexExpr &&) = delete;

  // Query functions.
  // Check if the question mark is specific so that it can be distinguished from
  // other question marks.
  bool specificQuestionmark() const;
  // Check if two question marks are the same or not. Two question marks are the
  // same if they are specific and have the same value.
  bool sameQuestionmark(IndexExpr const &o) const;

private:
  QuestionmarkIndexExpr(IndexExprImpl *otherObjPtr);
};

// Subclass to explicitly create Predicate IndexExpr.
class PredicateIndexExpr : public IndexExpr {
public:
  PredicateIndexExpr() = default;
  PredicateIndexExpr(bool const value); // Make a predicate constant value.
  PredicateIndexExpr(mlir::Value const value);
  PredicateIndexExpr(IndexExpr const &o);
  PredicateIndexExpr(UndefinedIndexExpr const &o);
  PredicateIndexExpr(LiteralIndexExpr const &o);
  PredicateIndexExpr(NonAffineIndexExpr const &o);
  PredicateIndexExpr(QuestionmarkIndexExpr const &o);
  PredicateIndexExpr(PredicateIndexExpr const &o);
  PredicateIndexExpr(AffineIndexExpr const &o);
  PredicateIndexExpr(DimIndexExpr const &o);
  PredicateIndexExpr(SymbolIndexExpr const &o);

  PredicateIndexExpr(PredicateIndexExpr &&) = delete;
  virtual ~PredicateIndexExpr() = default;
  PredicateIndexExpr &operator=(const PredicateIndexExpr &) = delete;
  PredicateIndexExpr &operator=(PredicateIndexExpr &&) = delete;

private:
  PredicateIndexExpr(IndexExprImpl *otherObjPtr);
};

// Subclass to explicitly create Affine IndexExpr.
class AffineIndexExpr : public IndexExpr {
public:
  AffineIndexExpr() = default;
  AffineIndexExpr(mlir::AffineExpr const value);
  AffineIndexExpr(IndexExpr const &o);
  AffineIndexExpr(UndefinedIndexExpr const &o);
  AffineIndexExpr(LiteralIndexExpr const &o);
  AffineIndexExpr(NonAffineIndexExpr const &o);
  AffineIndexExpr(QuestionmarkIndexExpr const &o);
  AffineIndexExpr(PredicateIndexExpr const &o);
  AffineIndexExpr(AffineIndexExpr const &o);
  AffineIndexExpr(DimIndexExpr const &o);
  AffineIndexExpr(SymbolIndexExpr const &o);

  AffineIndexExpr(AffineIndexExpr &&) = delete;
  virtual ~AffineIndexExpr() = default;
  AffineIndexExpr &operator=(const AffineIndexExpr &) = delete;
  AffineIndexExpr &operator=(AffineIndexExpr &&) = delete;

private:
  AffineIndexExpr(IndexExprImpl *otherObjPtr);
};

// Subclass to explicitly create Dim IndexExpr.
class DimIndexExpr : public IndexExpr {
public:
  DimIndexExpr() = default;
  DimIndexExpr(mlir::Value const value);
  DimIndexExpr(IndexExpr const &o);
  DimIndexExpr(UndefinedIndexExpr const &o);
  DimIndexExpr(LiteralIndexExpr const &o);
  DimIndexExpr(NonAffineIndexExpr const &o);
  DimIndexExpr(QuestionmarkIndexExpr const &o);
  DimIndexExpr(PredicateIndexExpr const &o);
  DimIndexExpr(AffineIndexExpr const &o);
  DimIndexExpr(DimIndexExpr const &o);
  DimIndexExpr(SymbolIndexExpr const &o);

  DimIndexExpr(DimIndexExpr &&) = default;
  virtual ~DimIndexExpr() = default;
  DimIndexExpr &operator=(const DimIndexExpr &) = default;
  DimIndexExpr &operator=(DimIndexExpr &&) = default;

private:
  DimIndexExpr(IndexExprImpl *otherObjPtr);
};

// Subclass to explicitly create IndexExpr.
class SymbolIndexExpr : public IndexExpr {
public:
  SymbolIndexExpr() = default;
  SymbolIndexExpr(mlir::Value const value);
  SymbolIndexExpr(IndexExpr const &o);
  SymbolIndexExpr(UndefinedIndexExpr const &o);
  SymbolIndexExpr(LiteralIndexExpr const &o);
  SymbolIndexExpr(NonAffineIndexExpr const &o);
  SymbolIndexExpr(QuestionmarkIndexExpr const &o);
  SymbolIndexExpr(PredicateIndexExpr const &o);
  SymbolIndexExpr(AffineIndexExpr const &o);
  SymbolIndexExpr(DimIndexExpr const &o);
  SymbolIndexExpr(SymbolIndexExpr const &o);

  SymbolIndexExpr(SymbolIndexExpr &&) = default;
  virtual ~SymbolIndexExpr() = default;
  SymbolIndexExpr &operator=(const SymbolIndexExpr &) = delete;
  SymbolIndexExpr &operator=(SymbolIndexExpr &&) = delete;

private:
  SymbolIndexExpr(IndexExprImpl *otherObjPtr);
};

//===----------------------------------------------------------------------===//
// Shortcuts for Index Expr subclasses, to render code more readable.
//===----------------------------------------------------------------------===//

using LitIE = LiteralIndexExpr;
using PredIE = PredicateIndexExpr;
using SymIE = SymbolIndexExpr;
using DimIE = DimIndexExpr;

//===----------------------------------------------------------------------===//
// Additional operators with integer values in first position
//===----------------------------------------------------------------------===//

inline IndexExpr operator+(int64_t const a, const IndexExpr &b) {
  return b + a;
}
inline IndexExpr operator*(int64_t const a, const IndexExpr &b) {
  return b * a;
}
inline IndexExpr operator-(int64_t const a, const IndexExpr &b) {
  return LitIE(a) - b;
}

//===----------------------------------------------------------------------===//
// Make IndexExpressions of a given type from provided input list/range
//===----------------------------------------------------------------------===//

// Create a list of IndexExpr of kind INDEX_EXPR from an ArrayRef of block args.
template <class INDEX_EXPR>
void getIndexExprList(mlir::ArrayRef<mlir::BlockArgument> inputList,
    llvm::SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : inputList)
    outputList.emplace_back(INDEX_EXPR(item));
}

// Create a list of IndexExpr of kind INDEX_EXPR from a value range (list of
// values).
template <class INDEX_EXPR>
void getIndexExprList(
    mlir::ValueRange range, llvm::SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : range)
    outputList.emplace_back(INDEX_EXPR(item));
}

inline llvm::SmallVector<IndexExpr, 4> DimListIE(mlir::ValueRange range) {
  llvm::SmallVector<IndexExpr, 4> outputList;
  getIndexExprList<DimIE>(range, outputList);
  return outputList;
}

inline llvm::SmallVector<IndexExpr, 4> SymListIE(mlir::ValueRange range) {
  llvm::SmallVector<IndexExpr, 4> outputList;
  getIndexExprList<SymIE>(range, outputList);
  return outputList;
}

// Create a list of IndexExpr of kind INDEX_EXPR from another list of IndexExpr.
template <class INDEX_EXPR>
void getIndexExprList(const mlir::ArrayRef<IndexExpr> inputList,
    llvm::SmallVectorImpl<IndexExpr> &outputList) {
  outputList.clear();
  for (auto item : inputList)
    outputList.emplace_back(INDEX_EXPR(item));
}

inline llvm::SmallVector<IndexExpr, 4> DimListIE(
    const mlir::ArrayRef<IndexExpr> inputList) {
  llvm::SmallVector<IndexExpr, 4> outputList;
  getIndexExprList<DimIE>(inputList, outputList);
  return outputList;
}

inline llvm::SmallVector<IndexExpr, 4> SymListIE(
    const mlir::ArrayRef<IndexExpr> inputList) {
  llvm::SmallVector<IndexExpr, 4> outputList;
  getIndexExprList<SymIE>(inputList, outputList);
  return outputList;
}

// Create a list of IndexExpr of kind LiteralIndexExpr from a list of integers.
void getIndexExprListFromInt(mlir::ArrayRef<int64_t> inputList,
    llvm::SmallVectorImpl<IndexExpr> &outputList);

// Create a list of IndexExpr of kind LiteralIndexExpr/Questionmark from a
// shape. Negative values are translated to Questionmarks.
void getIndexExprListFromShape(mlir::ArrayRef<int64_t> inputList,
    llvm::SmallVectorImpl<IndexExpr> &outputList);

} // namespace onnx_mlir
#endif
