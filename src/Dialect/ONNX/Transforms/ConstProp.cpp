/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to constprop an ONNX operation into
// composition of other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/Transforms/ConstProp.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <math.h>
#include <numeric>

#define DEBUG_TYPE "constprop-onnx"

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Instructions to add a constant operation.
//===----------------------------------------------------------------------===//
// There is currently support for adding constant propagation for unary and
// binary arithmetic ops (binary ops support broadcast). To add an operation,
// you simply have to add a templated method on how to compute the result in
// terms of one or two inputs.
//
// The methods are:
//
// ElementWiseBinaryOpImpl and ElementWiseUnaryOpImpl
// and they need to be templated with an ONNX Operation (presuably).
//
// Then you need to add rules on how to transform the patterns; look into
// ConstProp.td for example.
//

// Populated by configureConstPropONNXToONNXPass().
struct ConstPropONNXToONNXPassConfiguration {
  static bool roundFPToInt;
  static int expansionBound;
  static StringSet<> disabledPatterns;
  static bool constantPropIsDisabled;
};

bool ConstPropONNXToONNXPassConfiguration::roundFPToInt = false;
int ConstPropONNXToONNXPassConfiguration::expansionBound = -1; // -1 == no bound
StringSet<> ConstPropONNXToONNXPassConfiguration::disabledPatterns = {};
bool ConstPropONNXToONNXPassConfiguration::constantPropIsDisabled = false;

// Precondition: result has ranked tensor type with static shape and int or
// float element type.
bool satisfiesExpansionBound(Value result) {
  if (ConstPropONNXToONNXPassConfiguration::expansionBound < 0) {
    return true; // -1 == no bound
  }
  auto resultType = cast<RankedTensorType>(result.getType());
  assert(resultType.hasStaticShape() && "expansion bound needs static shape");
  int64_t sum = 0;
  for (auto operand : result.getDefiningOp()->getOperands()) {
    if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
      if (type.hasStaticShape())
        sum += getSizeInBytes(type);
  }
  return sum * ConstPropONNXToONNXPassConfiguration::expansionBound >=
         getSizeInBytes(resultType);
}

// We want to disable Constant Propagation when a user
// manually specifies the "disable-constant-prop" flag.
bool isConstantPropagationDisabled() {
  bool disable = (/*disableConstantProp*/ ConstPropONNXToONNXPassConfiguration::
          constantPropIsDisabled);
  return disable;
}

bool isNotDisabled(StringRef name) {
  bool ok =
      !ConstPropONNXToONNXPassConfiguration::disabledPatterns.contains(name);
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE " isNotDisabled " << name << " " << ok
                          << "\n");
  return ok;
}

ElementsAttr getConstValueElements(Value constValue) {
  ONNXConstantOp constOp = cast<ONNXConstantOp>(constValue.getDefiningOp());
  return mlir::cast<ElementsAttr>(constOp.getValueAttr());
}

// Creates ONNXConstantOp with the location from replacingValue.
Value createReplacingConstantOp(
    PatternRewriter &rewriter, Value replacingValue, ElementsAttr elements) {
  return OnnxBuilder(rewriter, replacingValue.getLoc()).constant(elements);
}

// Helper to restrict specialization to non-bool types.
template <typename T>
using EnableNotBool = std::enable_if_t<!std::is_same_v<T, bool>>;

/// Checks whether a variadic value is produced by dense ONNXConstantOps.
bool isVariadicOperandFromDenseONNXConstantOp(ValueRange operands) {
  return llvm::all_of(operands, [](Value v) { return isDenseONNXConstant(v); });
}

Value ConstZeroTensor(
    PatternRewriter &rewriter, Location loc, ShapedType type) {
  return OnnxBuilder(rewriter, loc)
      .constant(DenseElementsAttr::get(
          type, rewriter.getZeroAttr(type.getElementType())));
}

WideNum asWideNum(double n, Type elemType) {
  return wideZeroDispatch(elemType, [n](auto wideZero) {
    using cpptype = decltype(wideZero);
    constexpr BType TAG = toBType<cpptype>;
    return WideNum::widen<TAG>(static_cast<cpptype>(n));
  });
}

/// Checks whether a constant tensor's elements are all equal to a given scalar.
bool isConstOf(Value constValue, double n) {
  ElementsAttr constElements = getConstValueElements(constValue);
  Type elemType = constElements.getElementType();
  assert(!elemType.isInteger(1) && "booleans are not supported");
  WideNum w = asWideNum(n, elemType);
  return ElementsAttrBuilder::allEqual(constElements, w);
}

// Extracts number from a scalar constant value.
WideNum getScalarNum(Value constValue) {
  ElementsAttr elements = getConstValueElements(constValue);
  Type elementType = elements.getElementType();
  if (isa<FloatType>(elementType)) {
    APFloat f = *elements.value_begin<APFloat>();
    return WideNum::fromAPFloat(f);
  } else if (auto itype = dyn_cast<IntegerType>(elementType)) {
    APInt i = *elements.value_begin<APInt>();
    return WideNum::fromAPInt(i, !itype.isUnsigned());
  } else {
    llvm_unreachable("Only integer and float types are supported");
  }
}

ElementsAttr ConstPropReshapeImpl(PatternRewriter &rewriter,
    Value replacingValue, Value constValue, ArrayRef<int64_t> reshapedShape) {
  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  return elementsBuilder.reshape(constElements, reshapedShape);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as input the element
// type as well as the two element attributes for the operation, and return the
// result of the operation.

template <typename OP, typename T, class Enable = void>
struct ElementWiseBinaryOpImpl {
  static T eval(T lhs, T rhs) { llvm_unreachable("unsupported op or type"); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAddOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSubOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs - rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMulOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs * rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXDivOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs / rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMinOp, T> {
  static T eval(T lhs, T rhs) { return std::min<T>(lhs, rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMaxOp, T> {
  static T eval(T lhs, T rhs) { return std::max<T>(lhs, rhs); }
};

template <>
struct ElementWiseBinaryOpImpl<ONNXModOp, int64_t, EnableNotBool<int64_t>> {
  static int64_t eval(int64_t lhs, int64_t rhs) {
    // The original calculation for mod
    int64_t mod = lhs % rhs;
    // Handle the case when one of the int values are negative
    // If both int values are positive or multiples of each other, we can
    // calculate as normal
    if ((mod != 0) && ((lhs < 0) ^ (rhs < 0)))
      return (mod + rhs);
    return mod;
  }
};

template <>
struct ElementWiseBinaryOpImpl<ONNXModOp, double, EnableNotBool<double>> {
  static double eval(double lhs, double rhs) {
    // Rounding to match the results of the backend tests
    return (std::floor(fmod(lhs, rhs) * 1000000000) / 1000000000);
  }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXEqualOp, T> {
  static bool eval(T lhs, T rhs) { return lhs == rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXLessOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs < rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXGreaterOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs > rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXLessOrEqualOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs <= rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXGreaterOrEqualOp, T, EnableNotBool<T>> {
  static bool eval(T lhs, T rhs) { return lhs >= rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSumOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXPowOp, T, EnableNotBool<T>> {
  static T eval(T lhs, T rhs) { return std::pow(lhs, rhs); }
};

template <typename ElementwiseBinaryOp>
constexpr auto elementwiseBinaryOpCombiner(Type elemType) {
  return getWideNumWrappedTemplateFunction<ElementWiseBinaryOpImpl,
      ElementwiseBinaryOp>(elemType);
}

constexpr auto addCombiner(Type elemType) {
  return elementwiseBinaryOpCombiner<ONNXAddOp>(elemType);
}

constexpr auto subCombiner(Type elemType) {
  return elementwiseBinaryOpCombiner<ONNXSubOp>(elemType);
}

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  auto replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  ElementsAttr lhs = getConstValueElements(lhsValue);
  ElementsAttr rhs = getConstValueElements(rhsValue);
  Type operandsElemType = lhs.getElementType();
  assert(operandsElemType == rhs.getElementType() &&
         "all element-wise binary ops have matching operands element types");
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements = elementsBuilder.combine(lhs, rhs, replacingType,
      elementwiseBinaryOpCombiner<ElementwiseBinaryOp>(operandsElemType));
  return createReplacingConstantOp(rewriter, replacingValue, resultElements);
}

/// Do element-wise binary calculation of a variadic value and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
Value ConstPropVariadicElementwiseBinary(
    PatternRewriter &rewriter, Value replacingValue, ValueRange inputList) {
  assert(inputList.size() > 0 && "The variadic input is empty");
  auto replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  Value lhsValue = inputList[0];
  if (inputList.size() == 1)
    return lhsValue;

  ElementsAttr lhs = getConstValueElements(lhsValue);
  Type operandsElemType = lhs.getElementType();
  for (unsigned i = 1; i < inputList.size(); ++i) {
    Value rhsValue = inputList[i];
    ElementsAttr rhs = getConstValueElements(rhsValue);
    assert(operandsElemType == rhs.getElementType() &&
           "all element-wise binary ops have matching operands element types");
    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    lhs = elementsBuilder.combine(lhs, rhs, replacingType,
        elementwiseBinaryOpCombiner<ElementwiseBinaryOp>(operandsElemType));
  }
  return createReplacingConstantOp(rewriter, replacingValue, lhs);
}

//===----------------------------------------------------------------------===//
//// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP, typename T, class Enable = void>
struct ElementWiseUnaryOpImpl {
  static T eval(T val) { llvm_unreachable("unsupported op or type"); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T, EnableNotBool<T>> {
  static T eval(T val) { return -val; }
};

template <>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, double> {
  static double eval(double val) { return sqrt(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXReluOp, T, EnableNotBool<T>> {
  static T eval(T val) { return (val < 0) ? 0 : val; }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXReciprocalOp, T, EnableNotBool<T>> {
  static T eval(T val) { return (1 / val); }
};

template <typename ElementwiseUnaryOp>
auto elementwiseUnaryOpFunction(Type elemType) {
  return getWideNumWrappedTemplateFunction<ElementWiseUnaryOpImpl,
      ElementwiseUnaryOp>(elemType);
}

/// Do element-wise unary calculation of 'input' value and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseUnaryOp>
Value ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  Type replacingElemType =
      mlir::cast<ShapedType>(replacingValue.getType()).getElementType();

  ElementsAttr constElements = getConstValueElements(constValue);
  assert(replacingElemType == constElements.getElementType() &&
         "all element-wise unary ops preserve element type");
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr transposedElements =
      elementsBuilder.transform(constElements, replacingElemType,
          elementwiseUnaryOpFunction<ElementwiseUnaryOp>(replacingElemType));
  return createReplacingConstantOp(
      rewriter, replacingValue, transposedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ONNXWhereOp in presence of
// broadcast.
//
/// Does element-wise ternary (cond ? lhs : rhs) with broadcast on all inputs.
//===----------------------------------------------------------------------===//

Value ConstPropWhere(PatternRewriter &rewriter, Value replacingValue,
    Value condValue, Value lhsValue, Value rhsValue) {
  auto replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  ElementsAttr cond = getConstValueElements(condValue);
  assert(cond.getElementType().isInteger(1) &&
         "ONNXWhereOp condition has bool element type");
  ElementsAttr lhs = getConstValueElements(lhsValue);
  ElementsAttr rhs = getConstValueElements(rhsValue);
  Type operandsElemType = lhs.getElementType();
  assert(operandsElemType == rhs.getElementType() &&
         "ONNXWhereOp branches have matching element types");
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements =
      elementsBuilder.where(cond, lhs, rhs, replacingType);
  return createReplacingConstantOp(rewriter, replacingValue, resultElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for reduce ops.
//
// In the template helper methods ReduceOp is the corresponding element-wise op
// (ONNXAddOp for ONNXReduceSumOp, ONNXMaxOp for ONNXReduceMaxOp, etc) for
// ReduceSum/Prod/Min/Max, except it is ONNXReduceMeanOp for ONNXReduceMeanOp
// which is constant propagated in a special way: it is computed with
// ReduceSum followed by element-wise division to calculate the mean.
//===----------------------------------------------------------------------===//

int64_t getSIntAttr(Operation *op, StringRef attrName, int64_t deflt) {
  IntegerAttr iattr = op->getAttrOfType<IntegerAttr>(attrName);
  return iattr ? iattr.getSInt() : deflt;
}

template <typename ReduceOp>
Attribute getIdentity(Builder &builder, Type type) {
  if constexpr (std::is_same_v<ReduceOp, ONNXAddOp>) {
    return builder.getZeroAttr(type);
  } else if constexpr (std::is_same_v<ReduceOp, ONNXMulOp>) {
    if (auto itype = mlir::dyn_cast<IntegerType>(type))
      return builder.getIntegerAttr(type, APInt(itype.getWidth(), 1));
    assert(mlir::isa<FloatType>(type) &&
           "only supported types are integer, float");
    return builder.getFloatAttr(type, 1.0);
  } else {
    // Follow NumPy which doesn't support empty tensor for Min, Max, Mean.
    llvm_unreachable("reduce op has no identify, zero-size tensor unsupported");
  }
}

std::function<WideNum(WideNum)> divideBy(Type type, int64_t denominator) {
  return wideZeroDispatchNonBool(type, [denominator](auto wideZero) {
    using WideCppType = decltype(wideZero);
    return widenumWrapped<WideCppType, WideCppType>(
        [denominator](auto x) { return x / denominator; });
  });
}

template <typename ReduceOp, typename AxesRange = std::initializer_list<APInt>>
Value ConstPropReduceAxesRange(PatternRewriter &rewriter, Value replacingValue,
    Value dataValue, AxesRange axesRange) {
  Operation *op = replacingValue.getDefiningOp();

  // Find absoluteAxes, converting any negative axes to non-negative.
  SmallVector<unsigned, 4> absoluteAxes;
  ElementsAttr data = getConstValueElements(dataValue);
  int64_t rank = mlir::cast<ShapedType>(data.getType()).getRank();
  for (APInt a : axesRange) {
    int64_t axis = a.getSExtValue();
    assert(-rank <= axis && axis < rank && "axis out of range");
    if (axis < 0)
      axis += rank;
    assert(std::find(absoluteAxes.begin(), absoluteAxes.end(), axis) ==
               absoluteAxes.end() &&
           "duplicate axis");
    absoluteAxes.push_back(axis);
  }

  // If axes are empty and !noop_with_empty_axes, reduce over all dimensions.
  if (absoluteAxes.empty() &&
      getSIntAttr(op, "noop_with_empty_axes", /*default=*/0) == 0) {
    for (int64_t axis = 0; axis < rank; ++axis)
      absoluteAxes.push_back(axis);
  }

  // Compute the result.
  ElementsAttr reduced;
  Type elemType = data.getElementType();
  if (absoluteAxes.empty()) {
    reduced = data; // noop
  } else if (data.empty()) {
    Attribute identity = getIdentity<ReduceOp>(rewriter, elemType);
    reduced = DenseElementsAttr::get(
        mlir::cast<ShapedType>(replacingValue.getType()), {identity});
  } else {
    bool keepdims = getSIntAttr(op, "keepdims", /*default=*/1) != 0;
    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    if constexpr (std::is_same_v<ReduceOp, ONNXReduceMeanOp>) {
      // sum = ReduceSum(data)
      ElementsAttr sum = elementsBuilder.reduce(
          data, absoluteAxes, keepdims, addCombiner(elemType));
      assert(data.size() % sum.size() == 0 &&
             "ReduceSum reduces tensor size by integer factor");
      int64_t denominator = data.size() / sum.size();
      // reduced = sum / denominator
      reduced = elementsBuilder.transform(
          sum, elemType, divideBy(elemType, denominator));
    } else {
      reduced = elementsBuilder.reduce(data, absoluteAxes, keepdims,
          elementwiseBinaryOpCombiner<ReduceOp>(elemType));
    }
  }

  return createReplacingConstantOp(rewriter, replacingValue, reduced);
}

template <typename ReduceOp>
Value ConstPropReduce(PatternRewriter &rewriter, Value replacingValue,
    Value dataValue, Value axesValue) {
  if (isNoneValue(axesValue)) {
    return ConstPropReduceAxesRange<ReduceOp>(
        rewriter, replacingValue, dataValue, {});
  } else {
    ElementsAttr axes = getConstValueElements(axesValue);
    auto axesRange = axes.getValues<APInt>();
    return ConstPropReduceAxesRange<ReduceOp>(
        rewriter, replacingValue, dataValue, axesRange);
  }
}

template <typename ReduceOp>
Value ConstPropReduce(PatternRewriter &rewriter, Value replacingValue,
    Value dataValue, ArrayAttr axesArray) {
  if (axesArray) {
    auto axesRange = axesArray.getAsValueRange<IntegerAttr>();
    return ConstPropReduceAxesRange<ReduceOp>(
        rewriter, replacingValue, dataValue, axesRange);
  } else {
    return ConstPropReduceAxesRange<ReduceOp>(
        rewriter, replacingValue, dataValue, {});
  }
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for matrix multiplication.
//===----------------------------------------------------------------------===//

Value ConstPropMatMul(PatternRewriter &rewriter, Value replacingValue,
    Value lhsMatrixValue, Value rhsMatrixValue) {
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr lhs = getConstValueElements(lhsMatrixValue);
  ElementsAttr rhs = getConstValueElements(rhsMatrixValue);
  ElementsAttr matMulElements = elementsBuilder.matMul(lhs, rhs);
  return createReplacingConstantOp(rewriter, replacingValue, matMulElements);
}

// Takes the matrix shape and zero point for the LHS argument to MatMulInteger
// and returns the zero point if it broadcasts to the matrix shape or else
// returns the zero point reshaped so it broadcasts to the matrix shape.
ElementsAttr reshapeMatMulIntegerLhsZero(
    ArrayRef<int64_t> matrixShape, ElementsAttr zeroPoint) {
  ShapedType zeroPointType = zeroPoint.getShapedType();
  ArrayRef<int64_t> zeroPointShape = zeroPointType.getShape();
  size_t zeroPointRank = zeroPointShape.size();
  if (zeroPointRank == 0 || (zeroPointRank == 1 && zeroPointShape[0] == 1)) {
    // Scalar case is easy: zeroPoint trivially broadcasts to matrix's shape.
    // Scalars can be represented as singleton tensors with rank 0 or 1.
  } else if (zeroPointRank == 1) {
    // Vector with zero point scalar per row. Same shape as a matrix column.
    int64_t rows = zeroPointShape[0];
    // Per-row zero point is a proper vector we need to broadcast, unless
    // matrix is also a vector so the broadcasts cancel out.
    size_t matrixRank = matrixShape.size();
    if (matrixRank == 1) {
      // Broadcast of matrix and zero point vectors cancel out.
      assert(matrixShape == zeroPointShape &&
             "MatMulInteger LHS matrix, zero_point vectors mismatch");
    } else {
      assert(matrixRank > 1 && "MatMulInteger LHS matrix cannot be scalar");
      // When matrix is a proper tensor, reshape by appending zero point axis
      // with dim size 1 to broadcast to matrix's shape.
      assert(rows == matrixShape[matrixRank - 2] &&
             "MatMulInteger LHS matrix, zero_point rows mismatch");
      return OnnxElementsAttrBuilder(zeroPoint.getContext())
          .reshape(zeroPoint, {rows, 1});
    }
  } else {
    // Proper tensor is easy: last axis broadcasts to matrix's shape.
    assert(zeroPointShape.back() == 1 &&
           "last dim is 1 when LHS zero_point is a proper tensor");
    assert(zeroPointShape.drop_back() == matrixShape.drop_back() &&
           "MatMulInteger LHS matrix, zero_point tensors mismatch");
  }
  return zeroPoint;
}

// Rhs zero point scalar / vector / tensor always broadcasts to
// matrix's shape.
ElementsAttr reshapeMatMulIntegerRhsZero(
    ArrayRef<int64_t> matrixShape, ElementsAttr zeroPoint) {
  return zeroPoint;
}

bool isMatMulIntegerMatrixZero(Value matrixValue, Value zeroPointValue,
    function_ref<ElementsAttr(ArrayRef<int64_t>, ElementsAttr)> reshapeZero) {
  ElementsAttr matrix = getConstValueElements(matrixValue);
  assert(matrix.getElementType().isInteger(8) &&
         "MatMulInteger input element types must be u8 or i8");

  // An empty matrix is trivially zero.
  if (matrix.empty())
    return true;

  // If zeroPointValue is omitted, "zero" means all elements are zero.
  if (isNoneValue(zeroPointValue)) {
    WideNum zero = matrix.getElementType().isUnsignedInteger()
                       ? WideNum::widen<BType::UINT8>(0u)
                       : WideNum::widen<BType::INT8>(0);
    return ElementsAttrBuilder::allEqual(matrix, zero);
  }

  ElementsAttr zeroPoint = getConstValueElements(zeroPointValue);
  assert(zeroPoint.getElementType() == matrix.getElementType() &&
         "MatMulInteger matrix, zero_point element types mismatch");
  assert(!zeroPoint.empty() &&
         "MatMulInteger zero_point must be non-empty when matrix is");

  ElementsAttr reshapedZeroPoint =
      reshapeZero(matrix.getShapedType().getShape(), zeroPoint);
  return ElementsAttrBuilder::equal(matrix, reshapedZeroPoint);
}

bool isMatMulIntegerLhsZero(Value matrixValue, Value zeroPointValue) {
  return isMatMulIntegerMatrixZero(
      matrixValue, zeroPointValue, reshapeMatMulIntegerLhsZero);
}

bool isMatMulIntegerRhsZero(Value matrixValue, Value zeroPointValue) {
  return isMatMulIntegerMatrixZero(
      matrixValue, zeroPointValue, reshapeMatMulIntegerRhsZero);
}

ElementsAttr getMatMulIntegerMatrixElements(
    ElementsAttrBuilder &elementsBuilder, Value matrixValue,
    Value zeroPointValue,
    function_ref<ElementsAttr(ArrayRef<int64_t>, ElementsAttr)> reshapeZero) {
  auto I32 = IntegerType::get(matrixValue.getContext(), 32);
  ElementsAttr matrix8 = getConstValueElements(matrixValue);
  ElementsAttr matrix32 = elementsBuilder.castToIntElementType(matrix8, I32);
  if (isNoneValue(zeroPointValue)) {
    return matrix32;
  } else {
    ElementsAttr zeroPoint8 = getConstValueElements(zeroPointValue);
    ElementsAttr reshapedZeroPoint8 =
        reshapeZero(matrix8.getShapedType().getShape(), zeroPoint8);
    ElementsAttr reshapedZeroPoint32 =
        elementsBuilder.castToIntElementType(reshapedZeroPoint8, I32);
    return elementsBuilder.combine(matrix32, reshapedZeroPoint32,
        matrix32.getShapedType(),
        subCombiner(I32)); // elementwiseBinaryOpCombiner<ONNXSubOp>(I32));
  }
}

Value ConstPropMatMulInteger(PatternRewriter &rewriter, Value replacingValue,
    Value lhsMatrixValue, Value rhsMatrixValue, Value lhsZeroPointValue,
    Value rhsZeroPointValue) {
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr lhs = getMatMulIntegerMatrixElements(elementsBuilder,
      lhsMatrixValue, lhsZeroPointValue, reshapeMatMulIntegerLhsZero);
  ElementsAttr rhs = getMatMulIntegerMatrixElements(elementsBuilder,
      rhsMatrixValue, rhsZeroPointValue, reshapeMatMulIntegerRhsZero);
  ElementsAttr matMulElements = elementsBuilder.matMul(lhs, rhs);
  return createReplacingConstantOp(rewriter, replacingValue, matMulElements);
}

Value ConstPropGemm(PatternRewriter &rewriter, Value replacingValue,
    Value lhsMatrixValue, Value rhsMatrixValue, Value biasMatrixValue) {
  ONNXGemmOp gemmOp = cast<ONNXGemmOp>(replacingValue.getDefiningOp());
  float alpha = gemmOp.getAlpha().convertToFloat();
  float beta = gemmOp.getBeta().convertToFloat();
  constexpr std::array<uint64_t, 2> IDENTITY = {0, 1};
  constexpr std::array<uint64_t, 2> TRANSPOSE = {1, 0};
  ArrayRef<uint64_t> permLhs = gemmOp.getTransA() == 0 ? IDENTITY : TRANSPOSE;
  ArrayRef<uint64_t> permRhs = gemmOp.getTransB() == 0 ? IDENTITY : TRANSPOSE;
  FloatType F64 = rewriter.getF64Type();
  ShapedType resType = cast<ShapedType>(replacingValue.getType());
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr lhs = getConstValueElements(lhsMatrixValue);
  ElementsAttr rhs = getConstValueElements(rhsMatrixValue);
  ElementsAttr res =
      elementsBuilder.matMul(elementsBuilder.transpose(lhs, permLhs),
          elementsBuilder.transpose(rhs, permRhs));
  if (alpha != 1.0) {
    res = elementsBuilder.castToFPElementType(res, F64);
    res = elementsBuilder.transform(res, F64, [alpha](WideNum n) {
      return WideNum::widen<BType::DOUBLE>(alpha * n.narrow<BType::DOUBLE>());
    });
  }
  bool hasBias = !isa<NoneType>(biasMatrixValue.getType());
  if (hasBias) {
    ElementsAttr bias = getConstValueElements(biasMatrixValue);
    if (beta != 1.0) {
      bias = elementsBuilder.castToFPElementType(bias, F64);
      bias = elementsBuilder.transform(bias, F64, [beta](WideNum n) {
        return WideNum::widen<BType::DOUBLE>(beta * n.narrow<BType::DOUBLE>());
      });
    }
    // If one of res or bias has been cast to F64 then also cast the other.
    if (res.getElementType() != bias.getElementType()) {
      // One cast is unnecessary but ok: cast to the same type is free.
      res = elementsBuilder.castToFPElementType(res, F64);
      bias = elementsBuilder.castToFPElementType(bias, F64);
    }
    // elemType will be F64 if alpha != 1.0 or beta != 1.0.
    Type elemType = res.getElementType();
    res = elementsBuilder.combine(
        res, bias, resType.clone(elemType), addCombiner(elemType));
  }
  // Cast back in case res was cast to F64 somewhere along the way.
  res = elementsBuilder.castElementType(res, resType.getElementType());
  return createReplacingConstantOp(rewriter, replacingValue, res);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

Value ConstPropTranspose(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  // TODO: figure out if default may be omitted and what to do in that case
  ArrayAttr permAttr =
      mlir::cast<ArrayAttr>(replacingValue.getDefiningOp()->getAttr("perm"));
  SmallVector<uint64_t, 4> perm;
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(mlir::cast<IntegerAttr>(permVal).getInt());

  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr transposedElements =
      elementsBuilder.transpose(constElements, perm);
  return createReplacingConstantOp(
      rewriter, replacingValue, transposedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

Value ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  ArrayRef<int64_t> reshapedShape = getShape(replacingValue.getType());
  ElementsAttr reshapedElements =
      ConstPropReshapeImpl(rewriter, replacingValue, input, reshapedShape);
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Squeeze.
//===----------------------------------------------------------------------===//

Value ConstPropSqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  ArrayRef<int64_t> reshapedShape = getShape(replacingValue.getType());
  ElementsAttr reshapedElements =
      ConstPropReshapeImpl(rewriter, replacingValue, input, reshapedShape);
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ScatterND.
//===----------------------------------------------------------------------===//

Value ConstPropScatterND(PatternRewriter &rewriter, Value replacingValue,
    Value data, Value indices, Value updates) {
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr dataElements = getConstValueElements(data);
  ElementsAttr indicesElements = getConstValueElements(indices);
  ElementsAttr updatesElements = getConstValueElements(updates);
  ElementsAttr scatteredElements =
      elementsBuilder.scatterND(dataElements, indicesElements, updatesElements);
  return createReplacingConstantOp(rewriter, replacingValue, scatteredElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for CastOp.
//===----------------------------------------------------------------------===//

Value ConstPropCast(PatternRewriter &rewriter, Value replacingValue,
    Value constValue, IntegerAttr saturate, TypeAttr to) {
  Type toType = to.getValue();
  assert(toType == getElementType(replacingValue.getType()) &&
         "result element type mismatch");

  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr castElements;
  if (auto ftype = dyn_cast<FloatType>(toType)) {
    bool doSaturate = saturate.getSInt() != 0 && ftype.getWidth() == 8;
    castElements =
        elementsBuilder.castToFPElementType(constElements, ftype, doSaturate);
  } else if (auto itype = dyn_cast<IntegerType>(toType)) {
    // The onnx.Cast spec doesn’t say whether cast from floating point to
    // integer type should truncate towards zero or round but past discussions
    // (onnx issues #2285, #3776, #5004) point to truncation like numpy.
    // But round to nearest, ties to even, is preferable for numerics.
    bool round = ConstPropONNXToONNXPassConfiguration::roundFPToInt;
    castElements =
        elementsBuilder.castToIntElementType(constElements, itype, round);
  } else {
    llvm_unreachable("cast to unsupported type");
  }
  return createReplacingConstantOp(rewriter, replacingValue, castElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//===----------------------------------------------------------------------===//

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  Operation *op = replacingValue.getDefiningOp();

  // Get shape, starts, steps via ShapeHelper.
  ONNXSliceOpShapeHelper shapeHelper(op, {});
  auto outcome = shapeHelper.computeShape();
  assert(succeeded(outcome) && "Failed to scan slice op parameters");
  SmallVector<int64_t> shape, starts, steps;
  IndexExpr::getShape(shapeHelper.getOutputDims(), shape);
  IndexExpr::getLiteral(shapeHelper.starts, starts);
  IndexExpr::getLiteral(shapeHelper.steps, steps);

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr inputElements = getConstValueElements(constValue);
  ElementsAttr slicedElements =
      elementsBuilder.slice(inputElements, shape, starts, steps);
  return createReplacingConstantOp(rewriter, replacingValue, slicedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for PadOp.
//===----------------------------------------------------------------------===//

Value ConstPropPad(PatternRewriter &rewriter, Value replacingValue, Value data,
    Value padValue) {
  Operation *op = replacingValue.getDefiningOp();

  // Get pads via ShapeHelper.
  ONNXPadOpShapeHelper shapeHelper(op, {});
  auto outcome = shapeHelper.computeShape();
  assert(succeeded(outcome) && "Failed to scan pad op parameters");
  SmallVector<int64_t> shape, pads;
  IndexExpr::getLiteral(shapeHelper.pads, pads);

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr dataElements = getConstValueElements(data);
  WideNum padNum = isa<NoneType>(padValue.getType())
                       ? asWideNum(0, dataElements.getElementType())
                       : getScalarNum(padValue);
  ElementsAttr paddedElements = elementsBuilder.pad(dataElements, pads, padNum);
  return createReplacingConstantOp(rewriter, replacingValue, paddedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConcatOp.
//===----------------------------------------------------------------------===//

Value ConstPropConcat(PatternRewriter &rewriter, Value replacingValue,
    ValueRange operands, IntegerAttr axisAttr) {
  ShapedType outputType = mlir::cast<ShapedType>(replacingValue.getType());
  int64_t axis = axisAttr.getValue().getSExtValue();
  if (axis < 0)
    axis += outputType.getRank();

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  SmallVector<ElementsAttr, 4> inputElements;
  inputElements.reserve(operands.size());
  for (Value input : operands)
    inputElements.push_back(getConstValueElements(input));
  ElementsAttr concatenatedElements =
      elementsBuilder.concat(inputElements, axis);
  return createReplacingConstantOp(
      rewriter, replacingValue, concatenatedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ExpandOp.
//===----------------------------------------------------------------------===//

Value ConstPropExpand(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ArrayRef<int64_t> expandedShape = getShape(replacingValue.getType());

  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr expandedElements =
      elementsBuilder.expand(constElements, expandedShape);
  return createReplacingConstantOp(rewriter, replacingValue, expandedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for GatherOp.
//===----------------------------------------------------------------------===//

Value ConstPropGather(PatternRewriter &rewriter, Value replacingValue,
    Value inputValue, Value indicesValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);
  int64_t axis = gatherOp.getAxis();
  if (axis < 0)
    axis += mlir::cast<ShapedType>(inputValue.getType()).getRank();

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr inputElements = getConstValueElements(inputValue);
  ElementsAttr indicesElements = getConstValueElements(indicesValue);
  ElementsAttr gatheredElements =
      elementsBuilder.gather(inputElements, indicesElements, axis);
  return createReplacingConstantOp(rewriter, replacingValue, gatheredElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ReshapeOp.
//===----------------------------------------------------------------------===//

Value ConstPropReshape(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ArrayRef<int64_t> reshapedShape = getShape(replacingValue.getType());
  ElementsAttr reshapedElements =
      ConstPropReshapeImpl(rewriter, replacingValue, constValue, reshapedShape);
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConstantOfShape.
//===----------------------------------------------------------------------===//

Value ConstPropConstantOfShape(PatternRewriter &rewriter, Value replacingValue,
    Value shape, Attribute value) {
  ElementsAttr shapeElements = getConstValueElements(shape);
  llvm::SmallVector<int64_t, 4> shapeVector(shapeElements.getValues<int64_t>());

  // ONNXConstantOfShapeOp::inferShapes() makes sure that the 'value' attribute
  // here is specified
  ElementsAttr constElements = mlir::cast<ElementsAttr>(value);

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr expandedElements =
      shapeVector.empty() ? elementsBuilder.reshape(constElements, shapeVector)
                          : elementsBuilder.expand(constElements, shapeVector);
  return createReplacingConstantOp(rewriter, replacingValue, expandedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Range.
//===----------------------------------------------------------------------===//

Value ConstPropRange(PatternRewriter &rewriter, Value replacingValue,
    Value start, Value limit, Value delta) {
  ShapedType replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr rangeElements = elementsBuilder.range(
      replacingType, getScalarNum(start), getScalarNum(delta));
  return createReplacingConstantOp(rewriter, replacingValue, rangeElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for NonZero.
//===----------------------------------------------------------------------===//

Value ConstPropNonZero(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr nonZeroElements = elementsBuilder.nonZero(constElements);
  return createReplacingConstantOp(rewriter, replacingValue, nonZeroElements);
}

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/ONNXConstProp.inc"

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
// Not done with tablegen which doesn't support variadic results.
//===----------------------------------------------------------------------===//

std::vector<Value> ConstPropSplit(PatternRewriter &rewriter,
    ResultRange replacingValues, Value input, Value split, int64_t axis) {
  unsigned numResults = replacingValues.size();
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  int64_t splitAxisSize = inputShape[axis];
  SmallVector<int64_t> splitSizes(numResults, splitAxisSize / numResults);
  if (isa<NoneType>(split.getType())) {
    // If split attribute is not specified, split size is equally divided.
    // TODO: Follow the onnx spec which is more relaxed (albeit incomplete).
    assert(splitAxisSize % numResults == 0 &&
           "The dimension at the split axis is expected to be divisible by "
           "the number of results");
  } else {
    ElementsAttr splitElements = getConstValueElements(split);
    assert(splitElements.size() == numResults &&
           "split length should match the number of results");
    auto splitValues = splitElements.getValues<int64_t>();
    splitSizes.assign(splitValues.begin(), splitValues.end());
    // TODO: Figure out why std::reduce() doesn't work on Linux s390x. Until
    //       then we're using std::accumulate() instead.
    assert(splitAxisSize ==
               std::accumulate(splitSizes.begin(), splitSizes.end(), 0) &&
           "split values must sum to axis size");
  }

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr inputElements = getConstValueElements(input);
  std::vector<ElementsAttr> resElements =
      elementsBuilder.split(inputElements, axis, splitSizes);
  std::vector<Value> resValues;
  resValues.reserve(numResults);
  for (unsigned int i = 0; i < numResults; ++i) {
    ElementsAttr splitElements = resElements[i];
    resValues.push_back(
        createReplacingConstantOp(rewriter, replacingValues[i], splitElements));
  }
  return resValues;
}

class SplitOfConst : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult match(ONNXSplitOp splitOp) const override {
    if (!isDenseONNXConstant(splitOp.getInput()))
      return failure();
    Value split = splitOp.getSplit();
    if (!(isa<NoneType>(split.getType()) || isDenseONNXConstant(split)))
      return failure();
    return success();
  }

  void rewrite(ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {
    rewriter.replaceOp(splitOp,
        ConstPropSplit(rewriter, splitOp.getResults(), splitOp.getInput(),
            splitOp.getSplit(), splitOp.getAxis()));
  }
};

class IfOfConst : public OpRewritePattern<ONNXIfOp> {
public:
  using OpRewritePattern<ONNXIfOp>::OpRewritePattern;

  LogicalResult match(ONNXIfOp ifOp) const override {
    if (!isDenseONNXConstant(ifOp.getCond()))
      return failure();
    return success();
  }

  void rewrite(ONNXIfOp ifOp, PatternRewriter &rewriter) const override {
    Value cond = ifOp.getCond();
    ElementsAttr condElements = getConstValueElements(cond);
    auto splitValues = condElements.getValues<bool>();
    Region *region;
    if (splitValues[0] == 0) {
      region = &ifOp.getElseBranch();
    } else {
      region = &ifOp.getThenBranch();
    }

    assert(
        region->hasOneBlock() && "Then/Else region should have only one block");

    Operation *yieldOp = region->front().getTerminator();
    ValueRange yields = yieldOp->getOperands();
    SmallVector<Value, 4> outputs(yields.begin(), yields.end());
    Block *newBlock =
        rewriter.splitBlock(&region->front(), region->front().begin());

    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(newBlock, ifOp);
    rewriter.replaceOp(ifOp, outputs);
  }
};

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstPropONNXToONNXPass)

  StringRef getArgument() const override { return "constprop-onnx"; }

  StringRef getDescription() const override {
    return "ConstProp ONNX operations into composition of "
           "other ONNX operations.";
  }
  void runOnOperation() final;
};

void ConstPropONNXToONNXPass::runOnOperation() {
  auto function = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  getConstPropONNXToONNXPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
    signalPassFailure();
}

} // end anonymous namespace.

void onnx_mlir::getConstPropONNXToONNXPatterns(RewritePatternSet &patterns) {
  if (isConstantPropagationDisabled())
    return;
  populateWithGenerated(patterns);
  if (isNotDisabled("SplitOfConst"))
    patterns.insert<SplitOfConst>(patterns.getContext());
  patterns.insert<IfOfConst>(patterns.getContext());
}

void onnx_mlir::configureConstPropONNXToONNXPass(bool roundFPToInt,
    int expansionBound, ArrayRef<std::string> disabledPatterns,
    bool constantPropIsDisabled) {
  ConstPropONNXToONNXPassConfiguration::roundFPToInt = roundFPToInt;
  ConstPropONNXToONNXPassConfiguration::expansionBound = expansionBound;
  ConstPropONNXToONNXPassConfiguration::disabledPatterns.insert(
      disabledPatterns.begin(), disabledPatterns.end());
  ConstPropONNXToONNXPassConfiguration::constantPropIsDisabled =
      constantPropIsDisabled;
}

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
