//===------- ShapeLoweringHelper.hpp - Helper functions for Lowering Shapes
//-------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering shapes for shape inference
// and ONNX lowering.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>
#include <stdint.h>

namespace mlir {

class ShapeValue {
public:
  // Constructors.
  ShapeValue(const ShapeValue &a);
  ShapeValue(int64_t val, ConversionPatternRewriter *rewriter);
  ShapeValue(Value val, bool affine, ConversionPatternRewriter *rewriter);
  // Copy.
  void Copy(const ShapeValue &a);
  // Status.
  bool IsConst() const { return isConst; }
  bool IsQuestionmark() const { return !isConst && !rewriter; }
  bool IsDynamic() const { return !IsConst(); }
  bool IsAffine() const { return isAffine; }
  // Getters/Setters.
  int64_t GetConstVal() const;
  void SetConstVal(int64_t val);
  Value GetDynVal(Location loc) const;
  void SetDynVal(Value val);
  ConversionPatternRewriter *GetRewriter() const;

  // Operators
  // Store the result of the addition or subtraction in current object.
  void Add(const ShapeValue &a, const ShapeValue &b, Location loc);
  void Sub(const ShapeValue &a, const ShapeValue &b, Location loc);
  void Mult(const ShapeValue &a, const ShapeValue &b, Location loc);
  void FloorDiv(const ShapeValue &a, const ShapeValue &b, Location loc);
  void CeilDiv(const ShapeValue &a, const ShapeValue &b, Location loc);
  // Store the result of the selection with true/false value.
  void Select(const ShapeValue &condA, const ShapeValue &condB,
      CmpIPredicate compare, const ShapeValue &trueVal,
      const ShapeValue &falseVal, Location loc);
  // Store val, clipped to be inside of min+minInc, max+maxInc bound.
  void Clip(const ShapeValue &val, const ShapeValue &min, const ShapeValue &max,
      int64_t minInc, int64_t maxInc, Location loc);

private:
  // Initializer
  void Init(int64_t val, ConversionPatternRewriter *rewriter);
  void Init(Value val, bool affine, ConversionPatternRewriter *rewriter);
  void Init(const ShapeValue &a);
  void Init(
      const ShapeValue &a, const ShapeValue &b, bool affineIfBConst = false);
  void Init(const ShapeValue &a, const ShapeValue &b, const ShapeValue &c);
  void Init(const ShapeValue &a, const ShapeValue &b, const ShapeValue &c,
      const ShapeValue &d);
  // Convert from constant value to dynamic.
  void MakeDynamic(Location loc);
  // Operator support.
  typedef std::function<void(ShapeValue &, const ShapeValue &)> F1;
  void UnaryOp(const ShapeValue &a, F1 finteger, F1 fvalue);
  typedef std::function<void(
      ShapeValue &, const ShapeValue &, const ShapeValue &)>
      F2;
  void BinaryOp(const ShapeValue &a, const ShapeValue &b, F2 finteger,
      F2 fvalue, bool affineIfBConst = false);
  typedef std::function<void(
      ShapeValue &, const ShapeValue &, const ShapeValue &, const ShapeValue &)>
      F3;
  void TernaryOp(const ShapeValue &a, const ShapeValue &b, const ShapeValue &c,
      F3 finteger, F3 fvalue);
  typedef std::function<void(ShapeValue &, const ShapeValue &,
      const ShapeValue &, const ShapeValue &, const ShapeValue &)>
      F4;
  void QuaternaryOp(const ShapeValue &a, const ShapeValue &b,
      const ShapeValue &c, const ShapeValue &d, F4 finteger, F4 fvalue);
  void QuaternarySelectOp(const ShapeValue &a, const ShapeValue &cb,
      const ShapeValue &c, const ShapeValue &d, F4 finteger, F4 fvalue);

  // Data.
  int64_t constVal;
  Value dynVal;
  ConversionPatternRewriter *rewriter;
  bool isConst;
  bool isAffine;
};

} // namespace mlir
