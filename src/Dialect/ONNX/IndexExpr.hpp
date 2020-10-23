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
  // Normal constructor.
  IndexExprContainer(ConversionPatternRewriter *rewriter, Location loc);

  // IndexExpr builders
  IndexExpr CreateUndefinedIndexExpr();
  IndexExpr CreateQuestionmarkIndexExpr();
  IndexExpr CreateIntLitIndexExpr(int64_t val);
  IndexExpr CreateDimIndexExpr(Value val);
  IndexExpr CreateDimIndexExpr(
      Value memref, ArrayRef<int64_t> memrefShape, int index);
  IndexExpr CreateSymbolIndexExpr(Value val);

  // Actions
  int AddDim(Value value);
  int AddSymbol(Value value);

  // querries
  bool IsShapeInferencePass() const { return !rewriter; }
  void DimAndSymbolList(SmallVectorImpl<Value> &list) const;
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
  IndexExpr();

  // Initalization.
  IndexExpr &InitAsUndefined();
  IndexExpr &InitAsQuestionmark(IndexExprContainer *container);
  IndexExpr &InitAsIntLit(IndexExprContainer *container, int64_t val);
  IndexExpr &InitAsSymbol(IndexExprContainer *container, Value val);
  IndexExpr &InitAsDim(IndexExprContainer *container, Value val);
  IndexExpr &InitAsDim(IndexExprContainer *container, Value memref,
      ArrayRef<int64_t> memrefShape, int index);
  IndexExpr &InitAsAffineExpr(IndexExprContainer *container, AffineExpr val);
  IndexExpr &InitAsValue(IndexExprContainer *container, Value val);

  // Shape inference querries.
  bool IsDefined() const;
  bool IsUndefined() const { return !IsDefined(); }
  bool IsIntLit() const;
  bool IsQuestionmark() const;
  bool IsAffine() const;
  bool IsSymbol() const;
  bool IsDim() const;
  bool IsShapeInferencePass() const;
  bool HasContainer() const;
  bool HasAffineExpr() const;
  bool HasValue() const;
  // Shape inference querries on list of indices.
  bool static AreAllIntLit(SmallVectorImpl<IndexExpr> &list);
  bool static AreAllAffine(SmallVectorImpl<IndexExpr> &list);

  // Getters.
  int64_t GetIntLit() const;
  AffineExpr GetAffineExpr();
  Value GetValue();
  IndexExprContainer *GetContainer() const;
  Location GetLocation() const;
  // Setters.
  void SetIntLit(int64_t val);
  void SetAffineExpr(AffineExpr expr) { affineExpr = expr; }
  void SetValue(Value val) { value = val; }

  // Possibly Affine Operations. For Mult, place lit in a position if possible.
  IndexExpr &Add(IndexExpr &a, IndexExpr &b);
  IndexExpr &Add(IndexExpr &a, int64_t b);
  IndexExpr &Sub(IndexExpr &a, IndexExpr &b);
  IndexExpr &Sub(IndexExpr &a, int64_t b);
  IndexExpr &Sub(int64_t a, IndexExpr &b);
  IndexExpr &Mult(IndexExpr &a, IndexExpr &b);
  IndexExpr &Mult(IndexExpr &a, int64_t b);
  IndexExpr &FloorDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &CeilDiv(IndexExpr &a, IndexExpr &b);
  IndexExpr &Mod(IndexExpr &a, IndexExpr &b);
  IndexExpr &Clamp(IndexExpr &val, IndexExpr &min, 
      IndexExpr &max);
  IndexExpr &Clamp(IndexExpr &val, int64_t min, IndexExpr &max);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred,
      IndexExpr &condB, IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred,
      int64_t condB, IndexExpr &trueVal, IndexExpr &falseVal);
  IndexExpr &Select(IndexExpr &condA, CmpIPredicate comparePred,
      int64_t condB, int64_t trueVal, IndexExpr &falseVal);
  IndexExpr &Min(SmallVectorImpl<IndexExpr> &vals);
  IndexExpr &Max(SmallVectorImpl<IndexExpr> &vals);
  void DebugPrint(const std::string &msg);

private:
  IndexExpr &Init(IndexExprContainer *container, bool newIsDefined,
      bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
      int newIntLit, AffineExpr newAffineExpr, Value newValue);
  IndexExpr &InitAsValueOrIntLit(IndexExprContainer *container, Value val,
      bool isAffine, bool isSymbol, bool isDim);

  IndexExpr &Copy(IndexExpr &a);
  typedef std::function<void(IndexExpr &, IndexExpr &, IndexExpr &)> F2;
  IndexExpr &BinaryOp(IndexExpr &a, IndexExpr &b, bool affineWithLitA,
      bool affineWithLitB, bool affineExprCompatible, F2 finteger, F2 faffine,
      F2 fvalue);
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
