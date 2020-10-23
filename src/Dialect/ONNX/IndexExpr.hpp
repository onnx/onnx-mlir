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

IndexExpr is a single data structure that holds either an ineger, an affine
expression or a Value. It is used to compute shape inference, loop bounds, and
index expressions in memory accesses. The main purpose of this data structure is
to use a single function to either determine shape inference or the actual
shapes when allocating/looping during the lowering.

During Shape inference, no code is generated; the IndexExpr will only be used to
eitehr determine the actual constant size or a questionmark (signifying unknown
at compile time).

During lowering, code can be generated, and if fact it must, do fill in the
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

2) IndexExprContainer
======================

Each IndexExpr must be part of a single container which holds all of the symbols
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
IndexExprContainer can be reused between a) and b). It is recommended as it
enables bigger AffineExpr. But when the computations in a) are not affine, then
a new container can be started for the b) part. The non-affine parts of a)
becomes symbols.

Note that in a computation, all expressions must use IndexExpr originating from
the same container.

3) Code Sample
==============

3a) Scan parameters (typically arrays). Check ONNXShapeHelper.cpp

// Look at operation parameter "start" and extract an IndexExpr at location i.

startInput =
  GetIndexExprFromArrayAt(container, op, operandAdaptor.starts(), i);
if (startInput.IsUndefined())
  return sliceOp->emitError("start input parameter could not be processed");

// Scan a dim input (can be compile time or runtime)

dimInput = container.CreateDimIndexExpr(data, dataShape, ii);

// Perform some computations

startPlusDim.Add(startInput, dimInput);
startPos.Select(
startInput, CmpIPredicate::slt, 0, startPlusDim, startInput);
// Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
neg.Clamp(startPos, 0, dimMinOneInput);
pos.Clamp(startPos, 0, dimInput);
startFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);

3b) Look at Slice in ONNXOps.cpp on how to use IndexExpr for shape inferences.
3c) Look at Slice.cpp on how to use IndexExpr for lowering.

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

class IndexExprContainer {
public:
  // Constructor.
  IndexExprContainer(ConversionPatternRewriter *rewriter, Location loc);

  // IndexExpr builders.
  IndexExpr CreateUndefinedIndexExpr();
  IndexExpr CreateQuestionmarkIndexExpr();
  IndexExpr CreateLiteralIndexExpr(int64_t val);
  IndexExpr CreateDimIndexExpr(Value val);
  IndexExpr CreateDimIndexExpr(
      Value memref, ArrayRef<int64_t> memrefShape, int index);
  IndexExpr CreateSymbolIndexExpr(Value val);

  // Actions for AffineExpr.
  int AddDim(Value value);
  int AddSymbol(Value value);

  // Querries.
  bool IsShapeInferencePass() const { return !rewriter; }

  // Getters.
  void GetDimAndSymbolList(SmallVectorImpl<Value> &list) const;
  int GetDimSize() const { return dims.size(); }
  int GetSymbolSize() const { return symbols.size(); }
  ConversionPatternRewriter &GetRewriter() const;
  Location GetLocation() const { return loc; }

private:
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  ConversionPatternRewriter *rewriter;
  Location loc;
};

class IndexExpr {
public:
  friend class IndexExprContainer;

  IndexExpr();

  // Shape inference querries.
  bool IsDefined() const;
  bool IsUndefined() const { return !IsDefined(); }
  bool IsLiteral() const;
  bool IsQuestionmark() const;
  bool IsAffine() const;
  bool IsSymbol() const;
  bool IsDim() const;
  bool IsShapeInferencePass() const;
  bool HasContainer() const;
  bool HasAffineExpr() const;
  bool HasValue() const;
  // Shape inference querries on list of indices.
  bool static AreAllLiteral(SmallVectorImpl<IndexExpr> &list);
  bool static AreAllAffine(SmallVectorImpl<IndexExpr> &list);

  // Getters.
  int64_t GetLiteral() const;
  AffineExpr GetAffineExpr();
  Value GetValue();
  IndexExprContainer *GetContainer() const;
  Location GetLocation() const;

  // Possibly Affine Operations.
  IndexExpr &Add(IndexExpr &a, IndexExpr &b);
  IndexExpr &Add(IndexExpr &a, int64_t b);
  IndexExpr &IncBy(IndexExpr &b);
  IndexExpr &IncBy(int64_t b);
  IndexExpr &Sub(IndexExpr &a, IndexExpr &b);
  IndexExpr &Sub(IndexExpr &a, int64_t b);
  IndexExpr &Sub(int64_t a, IndexExpr &b);
  IndexExpr &DecBy(IndexExpr &b);
  IndexExpr &DecBy(int64_t b);
  IndexExpr &Mult(IndexExpr &a, IndexExpr &b);
  IndexExpr &Mult(IndexExpr &a, int64_t b);
  IndexExpr &MultBy(IndexExpr &b);
  IndexExpr &MultBy(int64_t b);
  IndexExpr &FloorDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &FloorDivBy(IndexExpr &a);
  IndexExpr &CeilDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &CeilDivBy(IndexExpr &a);
  IndexExpr &Mod(IndexExpr &a, IndexExpr &b);
  IndexExpr &ModBy(IndexExpr &a);
  IndexExpr &Clamp(IndexExpr &val, IndexExpr &min, IndexExpr &max);
  IndexExpr &Clamp(IndexExpr &val, int64_t min, IndexExpr &max);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred,
      IndexExpr &condB, IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      int64_t trueVal, IndexExpr &falseVal);
  IndexExpr &AssignIf(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      IndexExpr &trueVal);
  IndexExpr &AssignIf(IndexExpr &condA, CmpIPredicate comparePred, int64_t condB,
      int64_t trueVal);
  IndexExpr &Min(SmallVectorImpl<IndexExpr> &vals);
  IndexExpr &Max(SmallVectorImpl<IndexExpr> &vals);
  void DebugPrint(const std::string &msg);

private:
  // Higher-level initalization calls.
  IndexExpr &InitAsUndefined();
  IndexExpr &InitAsQuestionmark(IndexExprContainer *container);
  IndexExpr &InitAsIntLit(IndexExprContainer *container, int64_t val);
  IndexExpr &InitAsSymbol(IndexExprContainer *container, Value val);
  IndexExpr &InitAsDim(IndexExprContainer *container, Value val);
  IndexExpr &InitAsDim(IndexExprContainer *container, Value memref,
      ArrayRef<int64_t> memrefShape, int index);
  IndexExpr &InitAsAffineExpr(IndexExprContainer *container, AffineExpr val);
  // Lower-level initialization calls.
  IndexExpr &InitAsValue(IndexExprContainer *container, Value val);
  IndexExpr &Init(IndexExprContainer *container, bool newIsDefined,
      bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
      int newIntLit, AffineExpr newAffineExpr, Value newValue);
  IndexExpr &InitAsValueOrIntLit(IndexExprContainer *container, Value val,
      bool isAffine, bool isSymbol, bool isDim);
  // Copy.
  IndexExpr &Copy(IndexExpr &a);
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
  IndexExpr &ReductionOp(SmallVectorImpl<IndexExpr> &vals, F2 litRed,
      Flist affineRed, F2 valueRed);

  IndexExprContainer *container;
  bool isDefined, isIntLit, isAffine, isSymbol, isDim;
  int64_t intLit;
  AffineExpr affineExpr;
  Value value;
};

} // namespace mlir
