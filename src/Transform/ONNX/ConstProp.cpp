/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXConstProp.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
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

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <math.h>
#include <numeric>
#include <unordered_map>

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

// Collects stats on the amount of constant propagation.
// The onnx-mlir binary dumps the stats if run with --onnx-const-prop-report.
struct ConstPropCounters {
  size_t invocations = 0;
  size_t input_elms = 0;

  static void count(const std::string &name, ValueRange operands) {
    auto &counters = map[name];
    counters.invocations += 1;
    for (auto oprnd : operands)
      if (!isa<NoneType>(oprnd.getType()))
        counters.input_elms +=
            getNumberOfElements(oprnd.getType().cast<ShapedType>());
  }

  static void dump(llvm::raw_ostream &os) {
    size_t total_invocations = 0, total_input_elms = 0;
    for (auto &entry : map)
      total_invocations += entry.second.invocations,
          total_input_elms += entry.second.input_elms;
    os << "constprop report (cumulative), entries: " << map.size()
       << ", total invocations:" << total_invocations
       << ", total input elements:" << total_input_elms << "\n";
    for (auto &entry : map)
      os << "  " << entry.first << " invocations:" << entry.second.invocations
         << " input elements:" << entry.second.input_elms << "\n";
  }

  static std::unordered_map<std::string, ConstPropCounters> map;
};

std::unordered_map<std::string, ConstPropCounters> ConstPropCounters::map;

ElementsAttr getConstValueElements(Value constValue) {
  ONNXConstantOp constOp = cast<ONNXConstantOp>(constValue.getDefiningOp());
  return constOp.getValueAttr().cast<ElementsAttr>();
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

/// Checks whether a constant tensor's elements are all equal to a given scalar.
bool isConstOf(Value constValue, double n) {
  ElementsAttr constElements = getConstValueElements(constValue);
  Type elemType = constElements.getElementType();
  assert(!elemType.isInteger(1) && "booleans are not supported");
  WideNum w = wideZeroDispatchNonBool(elemType, [n](auto wideZero) {
    using cpptype = decltype(wideZero);
    constexpr BType TAG = toBType<cpptype>;
    return WideNum::widen<TAG>(static_cast<cpptype>(n));
  });
  return ElementsAttrBuilder::allEqual(constElements, w);
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
  ConstPropCounters::count("ElementwiseBinary", {lhsValue, rhsValue});
  auto replacingType = replacingValue.getType().cast<ShapedType>();

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
  ConstPropCounters::count("VariadicElementwiseBinary", inputList);
  auto replacingType = replacingValue.getType().cast<ShapedType>();

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
  ConstPropCounters::count("ElementwiseUnary", {constValue});
  Type replacingElemType =
      replacingValue.getType().cast<ShapedType>().getElementType();

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
  ConstPropCounters::count("Where", {condValue, lhsValue, rhsValue});
  auto replacingType = replacingValue.getType().cast<ShapedType>();

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
    if (auto itype = type.dyn_cast<IntegerType>())
      return builder.getIntegerAttr(type, APInt(itype.getWidth(), 1));
    assert(type.isa<FloatType>() && "only supported types are integer, float");
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
  ConstPropCounters::count("Reduce", {dataValue});
  Operation *op = replacingValue.getDefiningOp();

  // Find absoluteAxes, converting any negative axes to non-negative.
  SmallVector<unsigned, 4> absoluteAxes;
  ElementsAttr data = getConstValueElements(dataValue);
  int64_t rank = data.getType().cast<ShapedType>().getRank();
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
        replacingValue.getType().cast<ShapedType>(), {identity});
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
  ConstPropCounters::count("MatMul", {lhsMatrixValue, rhsMatrixValue});
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
  Type I32 = IntegerType::get(matrixValue.getContext(), 32);
  ElementsAttr matrix8 = getConstValueElements(matrixValue);
  ElementsAttr matrix32 = elementsBuilder.castElementType(matrix8, I32);
  if (isNoneValue(zeroPointValue)) {
    return matrix32;
  } else {
    ElementsAttr zeroPoint8 = getConstValueElements(zeroPointValue);
    ElementsAttr reshapedZeroPoint8 =
        reshapeZero(matrix8.getShapedType().getShape(), zeroPoint8);
    ElementsAttr reshapedZeroPoint32 =
        elementsBuilder.castElementType(reshapedZeroPoint8, I32);
    return elementsBuilder.combine(matrix32, reshapedZeroPoint32,
        matrix32.getShapedType(),
        subCombiner(I32)); // elementwiseBinaryOpCombiner<ONNXSubOp>(I32));
  }
}

Value ConstPropMatMulInteger(PatternRewriter &rewriter, Value replacingValue,
    Value lhsMatrixValue, Value rhsMatrixValue, Value lhsZeroPointValue,
    Value rhsZeroPointValue) {
  ConstPropCounters::count("MatMulInteger",
      {lhsMatrixValue, rhsMatrixValue, lhsZeroPointValue, rhsZeroPointValue});
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
  ConstPropCounters::count(
      "MatMul", {lhsMatrixValue, rhsMatrixValue, biasMatrixValue});
  ONNXGemmOp gemmOp = cast<ONNXGemmOp>(replacingValue.getDefiningOp());
  float alpha = gemmOp.getAlpha().convertToFloat();
  float beta = gemmOp.getBeta().convertToFloat();
  constexpr std::array<uint64_t, 2> IDENTITY = {0, 1};
  constexpr std::array<uint64_t, 2> TRANSPOSE = {1, 0};
  ArrayRef<uint64_t> permLhs = gemmOp.getTransA() == 0 ? IDENTITY : TRANSPOSE;
  ArrayRef<uint64_t> permRhs = gemmOp.getTransB() == 0 ? IDENTITY : TRANSPOSE;
  Type F64 = rewriter.getF64Type();
  ShapedType resType = cast<ShapedType>(replacingValue.getType());
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr lhs = getConstValueElements(lhsMatrixValue);
  ElementsAttr rhs = getConstValueElements(rhsMatrixValue);
  ElementsAttr res =
      elementsBuilder.matMul(elementsBuilder.transpose(lhs, permLhs),
          elementsBuilder.transpose(rhs, permRhs));
  if (alpha != 1.0) {
    res = elementsBuilder.castElementType(res, F64);
    res = elementsBuilder.transform(res, F64, [alpha](WideNum n) {
      return WideNum::widen<BType::DOUBLE>(alpha * n.narrow<BType::DOUBLE>());
    });
  }
  bool hasBias = !isa<NoneType>(biasMatrixValue.getType());
  if (hasBias) {
    ElementsAttr bias = getConstValueElements(biasMatrixValue);
    if (beta != 1.0) {
      bias = elementsBuilder.castElementType(bias, F64);
      bias = elementsBuilder.transform(bias, F64, [beta](WideNum n) {
        return WideNum::widen<BType::DOUBLE>(beta * n.narrow<BType::DOUBLE>());
      });
    }
    // If one of res or bias has been cast to F64 then also cast the other.
    if (res.getElementType() != bias.getElementType()) {
      // One cast is unnecessary but ok: cast to the same type is free.
      res = elementsBuilder.castElementType(res, F64);
      bias = elementsBuilder.castElementType(bias, F64);
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
  ConstPropCounters::count("Transpose", {constValue});
  // TODO: figure out if default may be omitted and what to do in that case
  ArrayAttr permAttr =
      replacingValue.getDefiningOp()->getAttr("perm").cast<ArrayAttr>();
  SmallVector<uint64_t, 4> perm;
  for (auto permVal : permAttr.getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

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
  ConstPropCounters::count("Unsqueeze", {input});
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
  ConstPropCounters::count("Squeeze", {input});
  ArrayRef<int64_t> reshapedShape = getShape(replacingValue.getType());
  ElementsAttr reshapedElements =
      ConstPropReshapeImpl(rewriter, replacingValue, input, reshapedShape);
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult ConstPropSplitPatternCommon(
    Op splitOp, PatternRewriter &rewriter, std::optional<ArrayAttr> splitAttr) {
  // Basic info.
  unsigned numResults = splitOp.getNumResults();
  Value input = splitOp.getInput();
  if (!isDenseONNXConstant(input))
    return failure();
  ConstPropCounters::count("Split", {input});
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  uint64_t splitAxis = splitOp.getAxis();
  int64_t splitAxisSize = inputShape[splitAxis];
  std::vector<int64_t> splitSizes(numResults, splitAxisSize / numResults);
  if (splitAttr.has_value()) {
    for (unsigned int i = 0; i < numResults; ++i)
      splitSizes[i] = ArrayAttrIntVal(splitAttr, i);
    // TODO: Figure out why std::reduce() doesn't work on Linux s390x. Until
    //       then we're using std::accumulate() instead.
    assert(splitAxisSize ==
               std::accumulate(splitSizes.begin(), splitSizes.end(), 0) &&
           "split values must sum to axis size");
  } else {
    // If split attribute is not specified, split size is equally divided.
    // TODO: Follow the onnx spec which is more relaxed (albeit incomplete).
    assert(splitAxisSize % numResults == 0 &&
           "The dimension at the split axis is expected to be divisible by "
           "the number of results");
  }

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr inputElements = getConstValueElements(input);
  std::vector<ElementsAttr> resElements =
      elementsBuilder.split(inputElements, splitAxis, splitSizes);
  std::vector<Value> resValues;
  resValues.reserve(numResults);
  for (unsigned int i = 0; i < numResults; ++i) {
    Value replacingValue = splitOp.getResult(i);
    ElementsAttr splitElements = resElements[i];
    resValues.push_back(
        createReplacingConstantOp(rewriter, replacingValue, splitElements));
  }
  rewriter.replaceOp(splitOp, resValues);
  return success();
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {
    std::optional<ArrayAttr> optionalAttr;

    auto split = splitOp.getSplit();
    if (auto splitConstOp = getONNXConstantOp(split)) {
      // Checking value of split parameter.
      optionalAttr.emplace(createArrayAttrFromConstantOp(splitConstOp));
    } else if (!split.getType().isa<NoneType>()) {
      llvm_unreachable("dynamic split not yet supported");
    }

    return ConstPropSplitPatternCommon(splitOp, rewriter, optionalAttr);
  }
};

class ConstPropSplitV11Pattern : public OpRewritePattern<ONNXSplitV11Op> {
public:
  using OpRewritePattern<ONNXSplitV11Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitV11Op splitOp, PatternRewriter &rewriter) const override {
    return ConstPropSplitPatternCommon(splitOp, rewriter, splitOp.getSplit());
  }
};

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ScatterND.
//===----------------------------------------------------------------------===//

class ConstPropScatterNDPattern : public OpRewritePattern<ONNXScatterNDOp> {
public:
  using OpRewritePattern<ONNXScatterNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXScatterNDOp scatterNdOp, PatternRewriter &rewriter) const override {
    // Match
    if (!scatterNdOp.getResult().getType().isa<RankedTensorType>())
      return failure();

    if (!isDenseONNXConstant(scatterNdOp.getData()))
      return failure();

    if (!isDenseONNXConstant(scatterNdOp.getIndices()))
      return failure();

    if (!isDenseONNXConstant(scatterNdOp.getUpdates()))
      return failure();

    ConstPropCounters::count(
        "Scatter", {scatterNdOp.getData(), scatterNdOp.getIndices(),
                       scatterNdOp.getUpdates()});

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    ElementsAttr dataElements = getConstValueElements(scatterNdOp.getData());
    ElementsAttr indicesElements =
        getConstValueElements(scatterNdOp.getIndices());
    ElementsAttr updatesElements =
        getConstValueElements(scatterNdOp.getUpdates());
    ElementsAttr scatteredElements = elementsBuilder.scatterND(
        dataElements, indicesElements, updatesElements);
    Value constOpResult = createReplacingConstantOp(
        rewriter, scatterNdOp.getData(), scatteredElements);

    rewriter.replaceOp(scatterNdOp, constOpResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for CastOp.
//===----------------------------------------------------------------------===//

Value ConstPropCast(PatternRewriter &rewriter, Value replacingValue,
    Value constValue, IntegerAttr saturate, TypeAttr to) {
  ConstPropCounters::count("Cast", {constValue});
  Type toType = to.getValue();
  assert(toType == getElementType(replacingValue.getType()) &&
         "result element type mismatch");

  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr castElements =
      elementsBuilder.castElementType(constElements, toType);

  // 'saturate' is ignored unless toType is a 8 bits float type.
  if (saturate && isa<FloatType>(toType) &&
      toType.getIntOrFloatBitWidth() == 8) {
    float max =
        dispatchByBType(btypeOfMlirType(toType), [&](auto btype) -> float {
          using cpptype = CppType<btype>;
          if constexpr (isSmallFPType<cpptype>) {
            return cpptype::max;
          } else {
            llvm_unreachable("unsupported 8 bits floating point type");
          }
        });
    // Clipping after cast relies on that cast is lazy and represents
    // elements as doubles until they are materialized, so it's not too
    // late to clip them here.
    // TODO: Clean up the contracts to make it clearer what's going on.
    //
    // Note that we saturate by clipping which isn't 100% faithful to the
    // onnx spec here: https://onnx.ai/onnx/technical/float8.html
    // and here: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
    // which, in the case of E4M3FNUZ and E5M2FNUZ, requires infinite values
    // to saturate to NaN, whereas we saturate them to lowest/highest with
    // clipping. Our clipping implementation matchint the reference
    // implementation in onnx/reference/ops/op_cast.py.
    // TODO: Change our implementation to match the spec, or change the spec.
    WideNum lowest = WideNum::widen<BType::FLOAT>(-max);
    WideNum highest = WideNum::widen<BType::FLOAT>(max);
    castElements = elementsBuilder.clip(castElements, lowest, highest);
  }

  return createReplacingConstantOp(rewriter, replacingValue, castElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//===----------------------------------------------------------------------===//

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ConstPropCounters::count("Slice", {constValue});
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
// Code to perform constant propagation for ConcatOp.
//===----------------------------------------------------------------------===//

Value ConstPropConcat(PatternRewriter &rewriter, Value replacingValue,
    ValueRange operands, IntegerAttr axisAttr) {
  ConstPropCounters::count("Concat", operands);
  ShapedType outputType = replacingValue.getType().cast<ShapedType>();
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
  ConstPropCounters::count("Expand", {constValue});
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
  ConstPropCounters::count("Gather", {inputValue, indicesValue});
  Operation *op = replacingValue.getDefiningOp();
  ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);
  int64_t axis = gatherOp.getAxis();
  if (axis < 0)
    axis += inputValue.getType().cast<ShapedType>().getRank();

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
  ConstPropCounters::count("Reshape", {constValue});
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
  ConstPropCounters::count("ConstantOfShape", {shape});
  ElementsAttr shapeElements = getConstValueElements(shape);
  llvm::SmallVector<int64_t, 4> shapeVector(shapeElements.getValues<int64_t>());

  // ONNXConstantOfShapeOp::inferShapes() makes sure that the 'value' attribute
  // here is specified
  ElementsAttr constElements = value.cast<ElementsAttr>();

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr expandedElements =
      shapeVector.empty() ? elementsBuilder.reshape(constElements, shapeVector)
                          : elementsBuilder.expand(constElements, shapeVector);
  return createReplacingConstantOp(rewriter, replacingValue, expandedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Range.
//===----------------------------------------------------------------------===//

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

Value ConstPropRange(PatternRewriter &rewriter, Value replacingValue,
    Value start, Value limit, Value delta) {
  ConstPropCounters::count("Range", {start});
  ShapedType replacingType = replacingValue.getType().cast<ShapedType>();

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
  ConstPropCounters::count("NonZero", {constValue});
  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr nonZeroElements = elementsBuilder.nonZero(constElements);
  return createReplacingConstantOp(rewriter, replacingValue, nonZeroElements);
}

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ONNXConstProp.inc"

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

  ConstPropONNXToONNXPass(bool report) : report(report) {}

  void runOnOperation() final;

private:
  bool report;
};
} // end anonymous namespace.

void ConstPropONNXToONNXPass::runOnOperation() {
  auto function = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());
  patterns.insert<ConstPropSplitV11Pattern>(&getContext());
  patterns.insert<ConstPropScatterNDPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns))))
    signalPassFailure();

  if (report)
    ConstPropCounters::dump(llvm::outs());
}

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass(
    bool report) {
  return std::make_unique<ConstPropONNXToONNXPass>(report);
}
