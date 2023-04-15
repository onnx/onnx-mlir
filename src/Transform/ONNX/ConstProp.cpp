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

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
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
      counters.input_elms += getNumberOfElements(oprnd.getType());
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

template <typename ElementwiseBinaryOp>
constexpr auto elementwiseBinaryOpCombiner(Type elemType) {
  return getWideNumWrappedTemplateFunction<ElementWiseBinaryOpImpl,
      ElementwiseBinaryOp>(elemType);
}

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  ConstPropCounters::count("ElementwiseBinary", {lhsValue, rhsValue});
  Type replacingType = replacingValue.getType().cast<ShapedType>();

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
  Type replacingType = replacingValue.getType().cast<ShapedType>();

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
  int64_t rank = data.getType().getRank();
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
    reduced = DenseElementsAttr::get(replacingValue.getType(), {identity});
  } else {
    bool keepdims = getSIntAttr(op, "keepdims", /*default=*/1) != 0;
    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    if constexpr (std::is_same_v<ReduceOp, ONNXReduceMeanOp>) {
      // sum = ReduceSum(data)
      ElementsAttr sum = elementsBuilder.reduce(data, absoluteAxes, keepdims,
          elementwiseBinaryOpCombiner<ONNXAddOp>(elemType));
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

// Takes the matrix shape and zero point for the LHS argument to MatMulInteger
// and returns the zero point if it broadcasts to the matrix shape or else
// returns the zero point reshaped so it broadcasts to the matrix shape.
ElementsAttr reshapeMatMulIntegerLhsZero(
    ArrayRef<int64_t> matrixShape, ElementsAttr zeroPoint) {
  ShapedType zeroPointType = zeroPoint.getType();
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
      reshapeZero(matrix.getType().getShape(), zeroPoint);
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
        reshapeZero(matrix8.getType().getShape(), zeroPoint8);
    ElementsAttr reshapedZeroPoint32 =
        elementsBuilder.castElementType(reshapedZeroPoint8, I32);
    return elementsBuilder.combine(matrix32, reshapedZeroPoint32,
        matrix32.getType(), elementwiseBinaryOpCombiner<ONNXSubOp>(I32));
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
LogicalResult ConstPropSplitPatternCommon(Op splitOp, PatternRewriter &rewriter,
    llvm::Optional<ArrayAttr> splitAttr) {
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
    llvm::Optional<ArrayAttr> optionalAttr;

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

/// Compute strides for a given shape.
std::vector<int64_t> getStrides(ArrayRef<int64_t> shape) {
  int rank = shape.size();
  std::vector<int64_t> strides;
  int64_t count = 1;
  for (int i = rank - 1; i >= 0; i--) {
    strides.insert(strides.begin(), count);
    count *= shape[i];
  }
  return strides;
}

/// Compute the linear access index.
int64_t getLinearAccessIndex(
    ArrayRef<int64_t> indices, ArrayRef<int64_t> strides) {
  int64_t index = 0;
  for (unsigned int i = 0; i < strides.size(); ++i)
    index += indices[i] * strides[i];
  return index;
}

// https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-13
/*
 * output = np.copy(data)
 * update_indices = indices.shape[:-1]
 * for idx in np.ndindex(update_indices):
 *     output[indices[idx]] = updates[idx]
 *
 * TODO: Move this to a scatterND method in ElementsAttrBuilder.
 */
void ScatterNDImpl(ElementsAttr dataElements, ElementsAttr indicesElements,
    ElementsAttr updatesElements, MutableArrayRef<WideNum> output) {
  readElementsWideNums(dataElements, output);
  ArrayBuffer<int64_t> indicesBuffer =
      getElementsArray<int64_t>(indicesElements);
  ArrayRef<int64_t> indices = indicesBuffer.get();
  ArrayBuffer<WideNum> updatesBuffer = getElementsWideNums(updatesElements);
  ArrayRef<WideNum> updates = updatesBuffer.get();

  auto dataShape = dataElements.getType().getShape();
  auto indicesShape = indicesElements.getType().getShape();
  auto updatesShape = updatesElements.getType().getShape();

  int64_t indices_nd = indicesShape.back();
  auto outer = indicesShape.drop_back();
  int64_t n_slices = ShapedType::getNumElements(outer);
  int64_t slice_size =
      ShapedType::getNumElements(updatesShape.drop_front(outer.size()));
  auto dataStrides = getStrides(dataShape);
  auto sliceStrides = llvm::ArrayRef(dataStrides).take_front(indices_nd);

  auto indicesIter = indices.begin();
  auto updatesIter = updates.begin();
  for (int64_t i = 0; i < n_slices; ++i) {
    ArrayRef<int64_t> idxs(indicesIter, indices_nd);
    int64_t pos = getLinearAccessIndex(idxs, sliceStrides);
    std::copy_n(updatesIter, slice_size, output.begin() + pos);
    indicesIter += indices_nd;
    updatesIter += slice_size;
  }
}

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
    ElementsAttr scatteredElements = elementsBuilder.fromWideNums(
        dataElements.getType(), [&](MutableArrayRef<WideNum> dst) {
          ScatterNDImpl(dataElements, indicesElements, updatesElements, dst);
        });
    Value constOpResult = createReplacingConstantOp(
        rewriter, scatterNdOp.getData(), scatteredElements);

    rewriter.replaceOp(scatterNdOp, constOpResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for CastOp.
//===----------------------------------------------------------------------===//

Value ConstPropCast(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ConstPropCounters::count("Cast", {constValue});
  Type replacingElemType =
      replacingValue.getType().cast<ShapedType>().getElementType();

  ElementsAttr constElements = getConstValueElements(constValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr castElements =
      elementsBuilder.castElementType(constElements, replacingElemType);
  return createReplacingConstantOp(rewriter, replacingValue, castElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//
// TODO: Move this to a slice method in ElementsAttrBuilder.
//===----------------------------------------------------------------------===//

void ConstPropSliceImpl(ShapedType outputType,
    const ONNXSliceOpShapeHelper &shapeHelper, ElementsAttr inputElements,
    MutableArrayRef<WideNum> outputData) {
  size_t rank = outputType.getRank();
  auto outputShape = outputType.getShape();
  std::vector<int64_t> inputStrides =
      getStrides(inputElements.getType().getShape());
  ArrayBuffer<WideNum> inputBuffer = getElementsWideNums(inputElements);
  const WideNum *start = inputBuffer.get().begin();
  SmallVector<int64_t, 4> steps(rank, 0);
  for (size_t axis = 0; axis < rank; ++axis) {
    start += shapeHelper.starts[axis].getLiteral() * inputStrides[axis];
    steps[axis] = shapeHelper.steps[axis].getLiteral() * inputStrides[axis];
  }
  for (StridesIterator<1> it(outputShape, {steps}), end(outputShape); it != end;
       ++it)
    outputData[it->flattenedIndex] = *(start + it->pos[0]);
}

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ConstPropCounters::count("Slice", {constValue});
  Operation *op = replacingValue.getDefiningOp();
  ONNXSliceOp sliceOp = cast<ONNXSliceOp>(op);

  // Get starts, ends, axes and steps via ShapeHelper.
  ONNXSliceOpShapeHelper shapeHelper(op, {});
  if (failed(shapeHelper.computeShape())) {
    sliceOp.emitError("Failed to scan " + ONNXSliceOp::getOperationName() +
                      " parameters successfully");
    return nullptr;
  }

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr inputElements = getConstValueElements(constValue);
  ShapedType outputType = replacingValue.getType().cast<ShapedType>();
  ElementsAttr slicedElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropSliceImpl(outputType, shapeHelper, inputElements, dst);
      });
  return createReplacingConstantOp(rewriter, replacingValue, slicedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConcatOp.
//
// TODO: Move this to a concat method in ElementsAttrBuilder.
//===----------------------------------------------------------------------===//

void ConstPropConcatImpl(ShapedType outputType,
    ArrayRef<ElementsAttr> inputElements, int64_t axis,
    MutableArrayRef<WideNum> outputData) {
  ArrayRef<int64_t> outputShape = outputType.getShape();
  size_t stride = ShapedType::getNumElements(outputShape.drop_front(axis));
  size_t start = 0;
  auto out = outputData.begin();
  for (ElementsAttr input : inputElements) {
    ArrayRef<int64_t> inputShape = input.getType().getShape();
    size_t len = ShapedType::getNumElements(inputShape.drop_front(axis));
    ArrayBuffer<WideNum> inputData = getElementsWideNums(input);
    auto in = inputData.get().begin();
    for (size_t offset = start; offset < outputData.size(); offset += stride) {
      std::copy_n(in, len, out + offset);
      in += len;
    }
    assert(in == inputData.get().end() && "input num elements mismatch");
    start += len;
  }
}

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
  ElementsAttr concatenatedElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropConcatImpl(outputType, inputElements, axis, dst);
      });
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
//
// TODO: Move this to a gather method in ElementsAttrBuilder.
//===----------------------------------------------------------------------===//

void ConstPropGatherImpl(ShapedType outputType, ElementsAttr inputElements,
    ElementsAttr indicesElements, int64_t axis,
    MutableArrayRef<WideNum> outputData) {
  ArrayBuffer<WideNum> inputData = getElementsWideNums(inputElements);
  ArrayBuffer<int64_t> indicesData = getElementsArray<int64_t>(indicesElements);
  auto inputShape = inputElements.getType().getShape();
  size_t axisSize = inputShape[axis];
  size_t inputStride = ShapedType::getNumElements(inputShape.drop_front(axis));
  size_t len = inputStride / axisSize;
  auto outputShape = outputType.getShape();
  size_t outputStride =
      ShapedType::getNumElements(outputShape.drop_front(axis));
  assert(outputStride == indicesData.get().size() * len);
  size_t start = 0;
  auto out = outputData.begin();
  for (int64_t idx : indicesData.get()) {
    int64_t adjustedIdx = idx < 0 ? idx + axisSize : idx;
    auto in = inputData.get().begin() + adjustedIdx * len;
    for (size_t offset = start; offset < outputData.size();
         offset += outputStride) {
      std::copy_n(in, len, out + offset);
      in += inputStride;
    }
    start += len;
  }
}

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
  ShapedType outputType = replacingValue.getType().cast<ShapedType>();
  ElementsAttr gatheredElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropGatherImpl(
            outputType, inputElements, indicesElements, axis, dst);
      });
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
  ElementsAttr shapeAttr =
      getONNXConstantOp(shape).getValueAttr().cast<ElementsAttr>();
  llvm::SmallVector<int64_t, 4> shapeVector(shapeAttr.getValues<int64_t>());

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
