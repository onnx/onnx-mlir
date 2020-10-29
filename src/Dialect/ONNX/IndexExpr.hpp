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

    startPlusDim.Add(startInput, dimInput);
    startPos.Select(startInput, CmpIPredicate::slt, 0, startPlusDim,
      startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    dimMinOneInput.Sub(dimInput, 1);
    neg.Clamp(startPos, 0, dimMinOneInput);
    pos.Clamp(startPos, 0, dimInput);
    startFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);

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
      IndexExpr loopIndex, start, step, actualIndex;
      loopIndex = childContext.CreateDimIndex(loopVal);
      start = childContext.CreateSymbolIndexFromParentContext(starts[ii]);
      step = childContext.CreateSymbolIndexFromParentContext(steps[ii]);
      actualIndex.Mult(step, loopIndex).IncBy(start);
      loadIndices.emplace_back(actualIndex.GetValue());
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

class IndexExprContext {
public:
  // Constructor for a top level context.
  IndexExprContext(ConversionPatternRewriter *rewriter, Location loc);
  // Constructor for a child context.
  IndexExprContext(IndexExprContext &parentContext);

  // IndexExpr basic builders.
  IndexExpr createUndefinedIndex();
  IndexExpr createQuestionmarkIndex();
  IndexExpr createLiteralIndex(int64_t val);
  IndexExpr createDimIndex(Value val);
  IndexExpr createSymbolIndex(Value val);

  // Scan a memref shape at index to generate an IndexExpr, typically used for
  // dimensions. Generate a literal when the memref dimension is known at
  // compile time.
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

  // Additional builder for repurposing IndexExpr from parent context.
  IndexExpr createSymbolIndexFromParentContext(IndexExpr &parentIndexExpr);

  // Actions for AffineExpr.
  int addDim(Value value);
  int addSymbol(Value value);

  // Querries.
  bool isShapeInferencePass() const { return !rewriter; }

  // Getters.
  void getDimAndSymbolList(SmallVectorImpl<Value> &list) const;
  int getNumDims() const { return dims.size(); }
  int getNumSymbols() const { return symbols.size(); }
  ConversionPatternRewriter &GetRewriter() const;
  Location GetLocation() const { return loc; }

  // Static helper functions.
  bool static areAllLiteral(SmallVectorImpl<IndexExpr> &list);
  bool static areAllAffine(SmallVectorImpl<IndexExpr> &list);
  void static getOutputDimsForType(SmallVectorImpl<IndexExpr> &outputIndices,
      SmallVectorImpl<int64_t> &outputDims);

private:
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  ConversionPatternRewriter *rewriter;
  Location loc;
  IndexExprContext *parentContext;
};

class IndexExpr {
public:
  friend class IndexExprContext;

  IndexExpr();

  // Shape inference querries.
  bool isDefined() const;
  bool isUndefined() const { return !isDefined(); }
  bool isLiteral() const;
  bool isQuestionmark() const;
  bool isAffine() const;
  bool isSymbol() const;
  bool isDim() const;
  bool isShapeInferencePass() const;
  bool hasContext() const;
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Getters.
  int64_t getLiteral() const;
  AffineExpr getAffineExpr();
  Value getValue();
  IndexExprContext &getContext() const;
  IndexExprContext *getContextPtr() const;
  Location getLoc() const;

  // Possibly Affine Operations.
  IndexExpr &add(IndexExpr &a, IndexExpr &b);
  IndexExpr &add(IndexExpr &a, int64_t b);
  IndexExpr &incBy(IndexExpr &b);
  IndexExpr &incBy(int64_t b);
  IndexExpr &sub(IndexExpr &a, IndexExpr &b);
  IndexExpr &sub(IndexExpr &a, int64_t b);
  IndexExpr &sub(int64_t a, IndexExpr &b);
  IndexExpr &decBy(IndexExpr &b);
  IndexExpr &decBy(int64_t b);
  IndexExpr &mult(IndexExpr &a, IndexExpr &b);
  IndexExpr &mult(IndexExpr &a, int64_t b);
  IndexExpr &multBy(IndexExpr &b);
  IndexExpr &multBy(int64_t b);
  IndexExpr &floorDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &floorDivBy(IndexExpr &b);
  IndexExpr &ceilDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &ceilDivBy(IndexExpr &b);
  IndexExpr &mod(IndexExpr &a, IndexExpr &b);
  IndexExpr &modBy(IndexExpr &a);
  IndexExpr &clamp(IndexExpr &val, IndexExpr &min, IndexExpr &max);
  IndexExpr &clamp(IndexExpr &val, int64_t min, IndexExpr &max);
  IndexExpr &select(IndexExpr &condA, CmpIPredicate comparePred,
      IndexExpr &condB, IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &select(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &select(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      int64_t trueVal, IndexExpr &falseVal);
  IndexExpr &assignIf(IndexExpr &condA, CmpIPredicate comparePred,
      int64_t condB, IndexExpr &trueVal);
  IndexExpr &assignIf(IndexExpr &condA, CmpIPredicate comparePred,
      int64_t condB, int64_t trueVal);
  IndexExpr &min(SmallVectorImpl<IndexExpr> &vals);
  IndexExpr &max(SmallVectorImpl<IndexExpr> &vals);
  void debugPrint(const std::string &msg);

private:
  // Higher-level basic initalization calls.
  IndexExpr &initAsUndefined();
  IndexExpr &initAsQuestionmark(IndexExprContext &context);
  IndexExpr &initAsLiteral(IndexExprContext &context, int64_t val);
  IndexExpr &initAsSymbol(IndexExprContext &context, Value val);
  IndexExpr &initAsDim(IndexExprContext &context, Value val);
  IndexExpr &initAsValue(IndexExprContext &context, Value val);
  IndexExpr &initAsAffineExpr(IndexExprContext &context, AffineExpr val);
  // Higher-level initiation calls that extract info
  IndexExpr &initAsDimFromMemref(IndexExprContext &context, Value memref,
      ArrayRef<int64_t> memrefShape, int index);
  IndexExpr &initAsSymbolFromArrayAtIndex(IndexExprContext &context,
      Operation *op, Value array, uint64_t indexInArray);
  IndexExpr &initAsSymbolFromArrayAtIndex(IndexExprContext &context,
      Operation *op, Value array, uint64_t indexInArray,
      int64_t defaultLiteral);
  // Lower-level initialization calls.
  IndexExpr &init(IndexExprContext *context, bool newIsDefined,
      bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
      int newIntLit, AffineExpr newAffineExpr, Value newValue);
  IndexExpr &initAsLitQuestionmarkOrValue(IndexExprContext &context, Value val,
      bool isAffine, bool symbol, bool dim);
  // Copy / private setters.
  IndexExpr &copy(IndexExpr &a);
  void setContext(IndexExprContext &context);
  // Support for Operations.
  typedef std::function<void(IndexExpr &, IndexExpr &, IndexExpr &)> F2;
  IndexExpr &BinaryOp(IndexExpr &a, IndexExpr &b, bool affineWithLitB,
      bool affineExprCompatible, F2 finteger, F2 faffine, F2 fvalue);
  typedef std::function<void(
      IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &)>
      F3;
  IndexExpr &TernaryOp(
      IndexExpr &a, IndexExpr &b, IndexExpr &c, F3 litFct, F3 valueFct);
  typedef std::function<void(
      IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &)>
      F4;
  IndexExpr &QuaternarySelectOp(IndexExpr &compA, IndexExpr &compB,
      IndexExpr &trueVal, IndexExpr &falseVal, F4 litFct, F4 valueFct);
  typedef std::function<void(IndexExpr &, SmallVectorImpl<IndexExpr> &)> Flist;
  IndexExpr &reductionOp(SmallVectorImpl<IndexExpr> &vals, F2 litRed,
      Flist affineRed, F2 valueRed);

  IndexExprContext *context;
  bool defined, litteral, affine, symbol, dim;
  int64_t intLit;
  AffineExpr affineExpr;
  Value value;
};

} // namespace mlir
