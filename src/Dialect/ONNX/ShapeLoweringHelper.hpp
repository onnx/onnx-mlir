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
  ShapeValue(ShapeValue &a);
  ShapeValue(int64_t val, ConversionPatternRewriter *rewriter);
  ShapeValue(Value val, bool affine, ConversionPatternRewriter *rewriter);
  // Copy.
  void Copy(ShapeValue &a);
  // Status.
  bool IsConst() { return isConst; }
  bool IsQuestionmark() { return !isConst && !rewriter; }
  bool IsDynamic() { return !IsConst(); }
  bool IsAffine() { return isAffine; }
  // Getters/Setters.
  int64_t GetConstVal();
  void SetConstVal(int64_t val);
  Value GetDynVal(Location loc);
  void SetDynVal(Value val);
  ConversionPatternRewriter *GetRewriter();

  // Operators
  // Store the result of the addition or subtraction in current object.
  void Add(ShapeValue &a, ShapeValue &b, Location loc);
  void Sub(ShapeValue &a, ShapeValue &b, Location loc);
  // Increment or decrement the current object by the provided value.
  void Inc(ShapeValue &a, Location loc);
  void Dec(ShapeValue &a, Location loc);
  // Store the result of the selection with true/false value.
  void Select(ShapeValue &condA, ShapeValue &condB, CmpIPredicate compare,
      ShapeValue &trueVal, ShapeValue &falseVal, Location loc);
  // Override current value with the true-value when the condition is satisfied.
  void Select(ShapeValue &condA, ShapeValue &condB, CmpIPredicate compare,
      ShapeValue &trueVal, Location loc);
  // Override the current values if outside of min+minInc, max+maxInc bound.
  void Clip(ShapeValue &min, ShapeValue &max, int64_t minInc, int64_t maxInc,
      Location loc);

private:
  // Initializer
  void Init(int64_t val, ConversionPatternRewriter *rewriter);
  void Init(Value val, bool affine, ConversionPatternRewriter *rewriter);
  void Init(ShapeValue &a);
  void Init(ShapeValue &a, ShapeValue &b, bool affineIfBConst = false);
  void Init(ShapeValue &a, ShapeValue &b, ShapeValue &c);
  void Init(ShapeValue &a, ShapeValue &b, ShapeValue &c, ShapeValue &d);
  // Convert from constant value to dynamic.
  void MakeDynamic(Location loc);
  // Operator support.
  typedef std::function<void(ShapeValue &, ShapeValue &)> F1;
  void UnaryOp(ShapeValue &a, F1 finteger, F1 fvalue);
  typedef std::function<void(ShapeValue &, ShapeValue &, ShapeValue &)> F2;
  void BinaryOp(ShapeValue &a, ShapeValue &b, F2 finteger, F2 fvalue,
      bool affineIfBConst = false);
  typedef std::function<void(
      ShapeValue &, ShapeValue &, ShapeValue &, ShapeValue &)>
      F3;
  void TernaryOp(
      ShapeValue &a, ShapeValue &b, ShapeValue &c, F3 finteger, F3 fvalue);
  typedef std::function<void(
      ShapeValue &, ShapeValue &, ShapeValue &, ShapeValue &, ShapeValue &)>
      F4;
  void QuaternaryOp(ShapeValue &a, ShapeValue &b, ShapeValue &c, ShapeValue &d,
      F4 finteger, F4 fvalue);
  void QuaternarySelectOp(ShapeValue &a, ShapeValue &cb, ShapeValue &c,
      ShapeValue &d, F4 finteger, F4 fvalue);

  // Data.
  int64_t constVal;
  Value dynVal;
  ConversionPatternRewriter *rewriter;
  bool isConst;
  bool isAffine;
};

} // namespace mlir
