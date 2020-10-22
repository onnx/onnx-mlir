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
  // Constructor used before a loop, where the previous index values are
  // becoming symbols.
  bool IsShapeInferencePass() const { return !rewriter; }
  int AddDim(Value value);
  int AddSymbol(Value value);
  void DimAndSymbolList(SmallVectorImpl<Value> &list) const;
  int GetDimSize() const { return dims.size(); }
  int GetSymbolSize() const { return symbols.size(); }
  ConversionPatternRewriter &GetRewriter() const;
  bool PerformShapeInference() const { return !rewriter; }
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
  IndexExpr(int64_t val); // Literal const.
  IndexExpr(IndexExprContainer &container, Value memref,
      ArrayRef<int64_t> memrefShape, int index); // Dim or Int Lit

  // Initalization.
  void InitAsUndefined();
  void InitAsQuestionmark();
  void InitAsIntLit(int64_t val);
  void InitAsSymbol(IndexExprContainer &container, Value val);
  void InitAsDim(IndexExprContainer &container, Value val);
  void InitAsDim(IndexExprContainer &container, Value memref,
      ArrayRef<int64_t> memrefShape, int index);
  void InitAsValue(IndexExprContainer &container, Value val);

  // Shape inference querries.
  bool IsDefined() const;
  bool IsIntLit() const;
  bool IsQuestionmark() const;
  bool IsAffine() const;
  bool IsSymbol() const;
  bool IsDim() const;
  bool HasAffineExpr() const;
  bool HasValue() const;
  // Shape inference querries on list of indices.
  bool static AreAllIntLit(SmallVectorImpl<IndexExpr> &list);
  bool static AreAllAffine(SmallVectorImpl<IndexExpr> &list);

  // Setter and getter.
  int64_t GetIntLit() const;
  AffineExpr GetAffineExpr(IndexExprContainer &container);
  Value GetValue(IndexExprContainer &container);
  void SetIntLit(int64_t val);
  void SetAffineExpr(AffineExpr expr) { affineExpr = expr; }
  void SetValue(Value val) { value = val; }

  // Possibly Affine Operations. For Mult, place lit in a position if possible.
  void Add(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Sub(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Mult(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void FloorDiv(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void CeilDiv(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Mod(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Clamp(IndexExprContainer &container, IndexExpr &val, IndexExpr &min,
      int64_t minInc, IndexExpr &max, int64_t maxInc);
  void Select(IndexExprContainer &container, IndexExpr &condA,
      CmpIPredicate comparePred, IndexExpr &condB, IndexExpr &trueVal,
      IndexExpr &falseVal);
  void Min(IndexExprContainer &container, SmallVectorImpl<IndexExpr> &vals);
  void Max(IndexExprContainer &container, SmallVectorImpl<IndexExpr> &vals);
  void DebugPrint(const std::string &msg);

private:
  void Init(bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim,
      bool newIsDefined, int newIntLit, AffineExpr newAffineExpr,
      Value newValue);
  void Init(bool isIntLit, bool isAffine);
  void InitAsValueOrIntLit(IndexExprContainer &container, Value val,
      bool isAffine, bool isSymbol, bool isDim);

  void Copy(IndexExpr &a);
  typedef std::function<void(IndexExpr &, IndexExpr &, IndexExpr &)> F2;
  void BinaryOp(IndexExprContainer &container, IndexExpr &a, IndexExpr &b,
      bool affineWithLitA, bool affineWithLitB, bool affineExprCompatible,
      F2 finteger, F2 faffine, F2 fvalue);
  typedef std::function<void(
      IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &)>
      F3;
  void TernaryOp(IndexExprContainer &container, IndexExpr &a, IndexExpr &b,
      IndexExpr &c, F3 litFct, F3 valueFct);
  typedef std::function<void(
      IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &, IndexExpr &)>
      F4;
  void QuaternarySelectOp(IndexExprContainer &container, IndexExpr &compA,
      IndexExpr &compB, IndexExpr &trueVal, IndexExpr &falseVal, F4 litFct,
      F4 valueFct);
  typedef std::function<void(IndexExpr &, SmallVectorImpl<IndexExpr> &)> Flist;
  void ReductionOp(IndexExprContainer &container,
      SmallVectorImpl<IndexExpr> &vals, F2 litRed, Flist affineRed,
      F2 valueRed);

  bool isIntLit, isAffine, isSymbol, isDim;
  bool isDefined; // For debug only.
  int64_t intLit;
  AffineExpr affineExpr;
  Value value;
};

} // namespace mlir
