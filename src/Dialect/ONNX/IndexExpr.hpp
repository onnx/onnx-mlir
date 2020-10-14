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
  IndexExprContainer(ConversionPatternRewriter *rewriter, Location loc);
  friend class IndexExpr;
  bool IsShapeInferencePass() const { return !rewriter; }
  int AddDim(Value value);
  int AddSymbol(Value value);
  void DimAndSymbolList(SmallVector<Value, 4> &list) const;
  int GetDimSize() const { return dims.size(); }
  int GetSymbolSize() const { return symbols.size(); }
  ConversionPatternRewriter *GetRewriter() const;
  bool PerformShapeInference() const { return ! rewriter; }
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
  void InitAsIntLit(int64_t val);
  void InitAsDim(Value val);
  void InitAsSymbol(Value val);
  void InitAsDimOrIntLit(IndexExprContainer &container, Value memref,
      ArrayRef<int64_t> memrefShape, int index);

  // Shape inference querries.
  bool IsIntLit() const { assert(isDefined); return isIntLit; }
  bool IsQuestionmark() const { return !IsIntLit(); }
  bool IsAffine() const { assert(isDefined); return isAffine; }

  bool IsSymbol() const { assert(isDefined); return isSymbol; }
  bool IsDim() const { assert(isDefined); return isDim; }
  bool HasAffineExpr() const { assert(isDefined); return !(!affineExpr); }
  bool HasValue() const { assert(isDefined); return !(!value); }

  // Shape inference querries on list of indices.
  bool static AreAllIntLit(SmallVectorImpl<IndexExpr> &list);
  bool static AreAllAffine(SmallVectorImpl<IndexExpr> &list);

  // Setter and getter.
  int64_t GetIntLit() const;
  AffineExpr GetAffineExpr(IndexExprContainer &container);
  Value GetValue(IndexExprContainer &container);
  void SetIntLiteral(int64_t val);
  void SetAffineExpr(AffineExpr expr) { affineExpr = expr; }
  void SetValue(Value val) { value = val; }

  // Possibly Affine Operations.
  void Add(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Sub(IndexExprContainer &container, IndexExpr &a, IndexExpr &b);
  void Mult(IndexExprContainer &container, IndexExpr &a, IndexExpr &b); // Lit in a.
  void FloorDiv(IndexExprContainer &container, IndexExpr &a, IndexExpr &b); // lit in b.
  void CeilDiv(IndexExprContainer &container, IndexExpr &a, IndexExpr &b); // Lit in b.
  void Mod(IndexExprContainer &container, IndexExpr &a, IndexExpr &b); // Lit in b.
  void Clamp(IndexExprContainer &container, IndexExpr &val, IndexExpr &min,
      int64_t minInc, IndexExpr &max, int64_t maxInc);
  void Select(IndexExprContainer &container, IndexExpr &condA,
      CmpIPredicate comparePred, IndexExpr &condB, IndexExpr &trueVal,
      IndexExpr &falseVal);
  void DebugPrint(const std::string &msg);

private:
  void Init(bool newIsIntLit, bool newIsAffine, bool newIsSymbol, bool newIsDim, bool newIsDefined, 
      int newIntLit, AffineExpr newAffineExpr, Value newValue);
  void Init(bool isIntLit, bool isAffine);

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

  bool isIntLit, isAffine, isSymbol, isDim;
  bool isDefined; // For debug only.
  int64_t intLit;
  AffineExpr affineExpr;
  Value value;
};

} // namespace mlir
