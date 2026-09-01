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
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/Transforms/ConstProp.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
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
  static int64_t maxTileFoldSize;
  static StringSet<> disabledPatterns;
  static bool constantPropIsDisabled;
};

bool ConstPropONNXToONNXPassConfiguration::roundFPToInt = false;
int ConstPropONNXToONNXPassConfiguration::expansionBound = -1; // -1 == no bound
int64_t ConstPropONNXToONNXPassConfiguration::maxTileFoldSize = 0;
StringSet<> ConstPropONNXToONNXPassConfiguration::disabledPatterns = {};
bool ConstPropONNXToONNXPassConfiguration::constantPropIsDisabled = false;

bool satisfiesMaxTileFoldSize(Value result) {
  int64_t maxSize = ConstPropONNXToONNXPassConfiguration::maxTileFoldSize;
  if (maxSize <= 0)
    return true;
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  return getSizeInBytes(resultType) <= maxSize;
}

// Precondition: result has ranked tensor type with static shape and int or
// float element type.
bool satisfiesExpansionBound(Value result) {
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType || !resultType.hasStaticShape())
    return true;
  // SmallVector<WideNum> uses uint32_t for capacity when sizeof(WideNum) >= 4
  // thus capping at UINT32_MAX elements.
  constexpr auto kMaxConstPropElements =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  if (resultType.getNumElements() > kMaxConstPropElements)
    return false;
  if (ConstPropONNXToONNXPassConfiguration::expansionBound < 0) {
    return true; // -1 == no bound
  }
  int64_t sum = 0;
  for (auto operand : result.getDefiningOp()->getOperands()) {
    if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
      if (type.hasStaticShape())
        sum += getSizeInBytes(type);
  }
  return sum * ConstPropONNXToONNXPassConfiguration::expansionBound >=
         getSizeInBytes(resultType);
}

/// True if the transpose result's element type matches the constant input's
/// element type after remapping per-axis quantization through `perm` (same
/// rule as ConvertToChannelLast: output axis i reads input axis perm[i]).
/// Requires an explicit `perm` attribute; ONNX's default (reverse axes) is
/// expected to be materialized during import or canonicalization.
bool valuesHaveSameDTypeForTransposeOfConst(
    Value transposeResult, Value input) {
  auto transposeOp = dyn_cast<ONNXTransposeOp>(transposeResult.getDefiningOp());
  if (!transposeOp)
    return false;

  auto inRanked = dyn_cast<RankedTensorType>(input.getType());
  auto outRanked = dyn_cast<RankedTensorType>(transposeResult.getType());
  if (!inRanked || !outRanked)
    return false;

  Type inElem = inRanked.getElementType();
  Type outElem = outRanked.getElementType();

  if (!transposeOp.getPermAttr())
    return false;
  SmallVector<int64_t, 8> perm;
  for (int64_t p :
      extractFromIntegerArrayAttr<int64_t>(transposeOp.getPermAttr()))
    perm.push_back(p);
  // Ranked input: perm length must match tensor rank (ONNX Transpose).
  if (static_cast<int64_t>(perm.size()) != inRanked.getRank())
    return false;

  Type transposedInElem = inElem;
  if (auto perAxis =
          dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inElem)) {
    int32_t oldAxis = perAxis.getQuantizedDimension();
    if (oldAxis < 0 || oldAxis >= inRanked.getRank())
      return false;
    // ONNX: output axis i reads input axis perm[i]; inverse maps input axis
    // -> output axis (same as ConvertToChannelLast::remapPerAxisQuantType).
    SmallVector<int64_t> invPerm = invertPermutationVector(perm);
    int32_t newAxis =
        static_cast<int32_t>(invPerm[static_cast<size_t>(oldAxis)]);
    if (newAxis != oldAxis) {
      transposedInElem =
          mlir::quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(),
              perAxis.getStorageType(), perAxis.getExpressedType(),
              perAxis.getScales(), perAxis.getZeroPoints(), newAxis,
              perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
    }
  }

  return transposedInElem == outElem;
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
  ElementsAttr elements = getDenseOrDisposableConstLikeElements(constValue);
  assert(elements && "getConstValueElements: value is not a dense constant");
  return elements;
}

// Creates ONNXConstantOp with the location from replacingValue.
Value createReplacingConstantOp(
    PatternRewriter &rewriter, Value replacingValue, ElementsAttr elements) {
  mlir::Value result =
      OnnxBuilder(rewriter, replacingValue.getLoc()).constant(elements);

  auto tensorType = cast<mlir::TensorType>(replacingValue.getType());
  if (isa<mlir::quant::QuantizedType>(tensorType.getElementType()))
    result.setType(replacingValue.getType());

  return result;
}

// Helper to restrict specialization to non-bool types.
template <typename T>
using EnableNotBool = std::enable_if_t<!std::is_same_v<T, bool>>;

template <typename T>
using EnableBool = std::enable_if_t<std::is_same_v<T, bool>>;

template <typename T>
using EnableInteger =
    std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>;

template <typename T>
using EnableFloatingPoint = std::enable_if_t<std::is_floating_point_v<T>>;

/// Checks whether all values in a variadic operand are dense ConstantLike.
bool isVariadicOperandFromDenseONNXConstantOp(ValueRange operands) {
  return llvm::all_of(operands, [](Value v) {
    return static_cast<bool>(getDenseOrDisposableConstLikeElements(v));
  });
}

Value ConstZeroTensor(
    PatternRewriter &rewriter, Location loc, ShapedType type) {
  return OnnxBuilder(rewriter, loc)
      .constant(DenseElementsAttr::get(
          type, rewriter.getZeroAttr(type.getElementType())));
}

template <typename GetFPConstFunc =
              std::function<APFloat(const llvm::fltSemantics &, bool)>,
    typename GetIntConstFunc = std::function<APInt(unsigned)>>
Value getClipConstantOfType(PatternRewriter &rewriter, ShapedType type,
    Location loc, GetFPConstFunc fpConstantFunc, bool isNegative,
    GetIntConstFunc intConstantFunc) {
  OnnxBuilder create(rewriter, loc);
  auto elemType = type.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    auto fpValue =
        fpConstantFunc(floatType.getFloatSemantics(), /*Negative=*/isNegative);
    return create.constant(DenseElementsAttr::get(
        RankedTensorType::get({}, elemType), llvm::ArrayRef(fpValue)));
  }
  auto intValue = intConstantFunc(elemType.getIntOrFloatBitWidth());
  return create.constant(DenseElementsAttr::get(
      RankedTensorType::get({}, elemType), llvm::ArrayRef(intValue)));
}

Value createMaximumValueForClip(
    PatternRewriter &rewriter, ShapedType type, Value value) {

  // Return 'value' if exists, as there is no need to clip to largest.
  if (!isNoneValue(value))
    return value;

  return getClipConstantOfType(rewriter, type, value.getLoc(),
      llvm::APFloat::getLargest, false, llvm::APInt::getMaxValue);
}

Value createMinimumValueForClip(
    PatternRewriter &rewriter, ShapedType type, Value value) {

  // Return 'value' if exists, as there is no need to clip to lowest.
  if (!isNoneValue(value))
    return value;

  return getClipConstantOfType(rewriter, type, value.getLoc(),
      llvm::APFloat::getLargest, true, llvm::APInt::getMinValue);
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
  static T eval(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T>) {
      if (rhs == 0) {
        // Undefined behavior. We can return any value.
        // Performing the divison would crash.
        return lhs;
      }
    }
    return lhs / rhs;
  }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXBitwiseAndOp, T, EnableInteger<T>> {
  static T eval(T lhs, T rhs) { return lhs & rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXBitwiseOrOp, T, EnableInteger<T>> {
  static T eval(T lhs, T rhs) { return lhs | rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAndOp, T, EnableBool<T>> {
  static T eval(T lhs, T rhs) { return lhs && rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXOrOp, T, EnableBool<T>> {
  static T eval(T lhs, T rhs) { return lhs || rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXXorOp, T, EnableBool<T>> {
  static T eval(T lhs, T rhs) { return lhs != rhs; }
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
struct ElementWiseUnaryOpImpl<ONNXBitwiseNotOp, T, EnableInteger<T>> {
  static T eval(T val) { return ~val; }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXAbsOp, T, EnableNotBool<T>> {
  static T eval(T val) {
    if constexpr (std::is_integral_v<T>) {
      // Cast to int64_t to disambiguate abs if T is signed.
      // Otherwise, just return the value.
      if constexpr (std::is_signed_v<T>) {
        return std::abs(static_cast<int64_t>(val));
      } else {
        return val;
      }
    } else {
      return std::fabs(val);
    };
  }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXCeilOp, T, EnableNotBool<T>> {
  static T eval(T val) { return ceil(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXCosOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return cos(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXErfOp, T, EnableNotBool<T>> {
  static T eval(T val) { return std::erf(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXExpOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return std::exp(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXFloorOp, T, EnableNotBool<T>> {
  static T eval(T val) { return floor(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXLogOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return std::log(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T, EnableNotBool<T>> {
  static T eval(T val) { return -val; }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNotOp, T, EnableBool<T>> {
  static T eval(T val) { return !val; }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSigmoidOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return 1 / (1 + std::exp(-val)); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSinOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return sin(val); }
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
struct ElementWiseUnaryOpImpl<ONNXReciprocalOp, T, EnableFloatingPoint<T>> {
  static T eval(T val) { return (1 / val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXRoundOp, T, EnableNotBool<T>> {
  static T eval(T val) { return std::nearbyint(val); }
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

  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(replacingElemType))
    replacingElemType = quantType.getStorageType();

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

bool canOpLikelyBeFusedWithBias(Value opValue) {
  return llvm::isa_and_nonnull<ONNXConvOp, ONNXMatMulOp, ONNXGemmOp>(
      opValue.getDefiningOp());
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
// Code to perform constant propagation for reverseSequence.
//===----------------------------------------------------------------------===//

Value ConstPropReverseSequence(PatternRewriter &rewriter, Value replacingValue,
    Value inputValue, Value sequenceValue) {

  ONNXReverseSequenceOp reverseSequenceOP = cast<ONNXReverseSequenceOp>(
      replacingValue.getDefiningOp<ONNXReverseSequenceOp>());

  auto batchAxis = reverseSequenceOP.getBatchAxis();

  ElementsAttr inputElements = getConstValueElements(inputValue);
  ElementsAttr sequenceElements = getConstValueElements(sequenceValue);
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr reverseSequencedElements = elementsBuilder.reverseSequence(
      inputElements, sequenceElements, batchAxis);
  return createReplacingConstantOp(
      rewriter, replacingValue, reverseSequencedElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

Value ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  assert(llvm::cast<ShapedType>(replacingValue.getType()).hasStaticShape());
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
  assert(llvm::cast<ShapedType>(replacingValue.getType()).hasStaticShape());
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
// Constant propagation for TileOp.
//===----------------------------------------------------------------------===//

Value ConstPropTile(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  auto inputType = mlir::cast<ShapedType>(constValue.getType());
  auto outputType = mlir::cast<ShapedType>(replacingValue.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  // A zero repeat (or zero-sized input) yields an empty tensor.
  if (outputType.getNumElements() == 0)
    return createReplacingConstantOp(rewriter, replacingValue,
        DenseElementsAttr::get(outputType, ArrayRef<Attribute>{}));

  ElementsAttr elements = getConstValueElements(constValue);

  if (elements.isSplat())
    return createReplacingConstantOp(rewriter, replacingValue,
        DenseElementsAttr::get(
            outputType, {elements.getSplatValue<Attribute>()}));

  // Collect the elements of TileOp as a broadcast:
  // - Insert a singleton dimension in front of each axis
  // - Expand those dimensions up to the per-axis repeat counts
  // - Flatten back to the tiled shape
  // This way we only allocate an ElementsAttr once.
  SmallVector<int64_t> singletonShape;
  SmallVector<int64_t> broadcastShape;
  for (auto [in, out] : llvm::zip(inputShape, outputShape)) {
    singletonShape.append({1, in});
    broadcastShape.append({out / in, in});
  }

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  elements = elementsBuilder.reshape(elements, singletonShape);
  elements = elementsBuilder.expand(elements, broadcastShape);
  elements = elementsBuilder.reshape(elements, outputShape);
  return createReplacingConstantOp(rewriter, replacingValue, elements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ShapeOp.
//===----------------------------------------------------------------------===//

/// Folds onnx.Shape(input) into a constant int64 tensor when the input has a
/// statically known shape. The optional start/end attributes are respected.
Value ConstPropShape(
    PatternRewriter &rewriter, Value replacingValue, Value inputValue) {
  auto inputType = mlir::cast<ShapedType>(inputValue.getType());
  assert(inputType.hasStaticShape() && "expected statically shaped input");

  int64_t rank = inputType.getRank();
  Operation *op = replacingValue.getDefiningOp();
  ONNXShapeOp shapeOp = cast<ONNXShapeOp>(op);

  // Normalize start: default 0, support negative indices.
  int64_t start = shapeOp.getStart();
  if (start < 0)
    start += rank;
  start = std::clamp(start, int64_t(0), rank);

  // Normalize end: default rank, support negative indices.
  int64_t end = rank;
  if (auto endAttr = shapeOp.getEnd())
    end = *endAttr < 0 ? *endAttr + rank : *endAttr;
  end = std::clamp(end, int64_t(0), rank);

  auto shape =
      inputType.getShape().slice(start, std::max(end - start, int64_t(0)));
  auto resultType = RankedTensorType::get(
      {static_cast<int64_t>(shape.size())}, rewriter.getI64Type());
  auto elements = DenseElementsAttr::get(
      resultType, ArrayRef<int64_t>(shape.begin(), shape.end()));
  return createReplacingConstantOp(rewriter, replacingValue, elements);
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
  assert(llvm::cast<ShapedType>(replacingValue.getType()).hasStaticShape());
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
  Type resultElementType = getElementType(replacingValue.getType());
  if (auto resultIntTy = dyn_cast<IntegerType>(resultElementType)) {
    if (nonZeroElements.getElementType() != resultElementType)
      nonZeroElements = elementsBuilder.castToIntElementType(
          nonZeroElements, resultIntTy, /*round=*/false);
  }
  return createReplacingConstantOp(rewriter, replacingValue, nonZeroElements);
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Resize.
//
// Resize with a constant data input (and constant/absent roi, scales, sizes)
// is a pure compile-time computation. onnx-mlir has no ElementsAttrBuilder
// primitive for it, so the resampling is implemented here directly. Only a
// subset of the Resize spec is handled; isResizeConstPropagatable() gates the
// pattern so any unsupported configuration is simply left untouched.
//
// ONNX Resize spec (attribute semantics, CTM/mode formulas) reference:
//   https://onnx.ai/onnx/operators/onnx__Resize.html
//
// Mental model:
//
//     output pixel  --CTM-->  input coordinate  --mode-->  output value
//
//   * coordinate_transformation_mode (CTM) answers WHERE to look: given an
//     output index, what (fractional) input coordinate does it map to?
//     -> resizeSourceCoord().
//   * mode answers HOW to compute the value once there:
//       nearest -> copy the 1 closest sample
//       linear  -> blend the 2 surrounding samples  (2x2 in 2D)
//       cubic   -> smooth curve over 4 samples       (4x4 in 2D)
//     -> resizeNearestIndex() / (linear inline) / resizeCubicWeight().
//
// Resize is separable: we resample one axis at a time (1-D interpolation along
// that axis, repeated), which composes into bilinear / bicubic for images.
//===----------------------------------------------------------------------===//

// CTM ("where to look"): maps an output index x back to the (fractional) input
// coordinate per the ONNX coordinate_transformation_mode. The variants differ
// only in how they align the two grids:
//   asymmetric          - samples at integer coords, no half-pixel shift:
//                         src = x / scale.
//   half_pixel          - samples are cell centers (coord = idx + 0.5); the
//                         standard, keeps the image centered when scaling.
//   pytorch_half_pixel  - half_pixel, but maps to 0 when outLen == 1.
//   align_corners       - pins the first/last output onto the first/last input
//                         exactly (corners line up).
//   half_pixel_symmetric- half_pixel with a centering correction when
//                         scale*inLen != outLen.
// tf_crop_and_resize is intentionally unsupported (gated out).
double resizeSourceCoord(
    int64_t x, double scale, int64_t inLen, int64_t outLen, StringRef ctm) {
  if (ctm == "asymmetric")
    return static_cast<double>(x) / scale;
  if (ctm == "align_corners")
    return outLen == 1 ? 0.0
                       : static_cast<double>(x) * (inLen - 1) / (outLen - 1);
  if (ctm == "pytorch_half_pixel")
    return outLen > 1 ? (x + 0.5) / scale - 0.5 : 0.0;
  if (ctm == "half_pixel_symmetric") {
    double adjustment = static_cast<double>(outLen) / (scale * inLen);
    double center = inLen / 2.0;
    double offset = center * (1.0 - adjustment);
    return offset + (x + 0.5) / scale - 0.5;
  }
  // "half_pixel" (default).
  return (x + 0.5) / scale - 0.5;
}

// mode=nearest ("how", part 1): once CTM has given the fractional source
// coordinate, snap it to a single integer input index and copy that sample
// (weight 1.0 - nearest computes no new value). nearest_mode only decides the
// tie-break / rounding direction when the coordinate falls between two samples.
int64_t resizeNearestIndex(double src, StringRef nearestMode) {
  if (nearestMode == "floor")
    return static_cast<int64_t>(std::floor(src));
  if (nearestMode == "ceil")
    return static_cast<int64_t>(std::ceil(src));
  if (nearestMode == "round_prefer_ceil")
    return static_cast<int64_t>(std::floor(src + 0.5));
  // "round_prefer_floor" (default).
  return static_cast<int64_t>(std::ceil(src - 0.5));
}

// mode=cubic ("how", part 2): returns the weight for ONE input sample given
// `distance` = signed distance from the (fractional) source coordinate to that
// sample (the body uses dist = |distance| and cubC = cubicCoeffA to match the
// standard cubic-convolution formula).
//
// Bigger picture: for an output that lands at source coord `src`, cubic uses
// the 4 nearest input samples at indices floor(src)-1 .. floor(src)+2. This
// helper is called once per sample to get its weight; resizeBuildAxisTaps then
// records the 4 {index, weight} taps and the value is sum(weight_i * sample_i)
// (weights are renormalized to sum to 1 so a flat input is preserved).
//
// The kernel is a piecewise cubic in |x| (the two branches below), so the
// weight depends only on distance, and larger distance => smaller weight. The
// 4 samples always fall into two distance bands. With `frac = src - floor(src)`
// (so frac is in [0,1)):
//
//   sample:    P0        P1      [src]   P2        P3
//   distance:  1+frac     frac  .        1-frac    2-frac    <- |x|, from src
//   band:      (1,2)      [0,1]          [0,1]     (1,2)
//   -> weight: small/-ve  large          large     small/-ve <- kernel(|x|)
//              (far)       (near)         (near)    (far)
//
// So P0 and P3 have the LARGEST distances (1+frac and 2-frac, both in (1,2)),
// which is exactly why their WEIGHTS come out small/negative: distance and
// weight are inverse.
//
//   * |x| <= 1  -> first branch, the two NEAR samples (P1, P2): weight peaks
//                  at x=0 and tapers to 0 by |x|=1.
//   * 1<|x|<2   -> second branch, the two FAR samples (P0, P3): small and,
//                  for the usual a=-0.75, negative. Those negative lobes let
//                  cubic sharpen edges and slightly overshoot, unlike linear.
//   * |x| >= 2  -> 0 (sample too far to contribute).
//
// cubicCoeffA is ONNX cubic_coeff_a (default -0.75) and sets how pronounced
// those lobes are. (linear needs no such helper: its 2 weights are just
// 1-frac and frac, computed inline in resizeBuildAxisTaps.)
double resizeCubicWeight(double distance, double cubicCoeffA) {
  double absDistance = std::abs(distance);
  if (absDistance <= 1.0)
    return ((cubicCoeffA + 2.0) * absDistance - (cubicCoeffA + 3.0)) *
               absDistance * absDistance +
           1.0;
  if (absDistance < 2.0)
    return (((absDistance - 5.0) * absDistance + 8.0) * absDistance - 4.0) *
           cubicCoeffA;
  return 0.0;
}

struct ResizeTap {
  int64_t index;
  double weight;
};

// Precomputes, for a single spatial axis, how every output position is built
// from the input samples along that axis. A "tap" is one {input index, weight}
// contribution; interpolating an output position is just the weighted sum of
// its taps over the input:  out[o] = sum_t weight_t * in[index_t].
//
// Called once per resampled axis (the separable resize loop in ConstPropResize
// invokes it for each axis whose length changes), then resizeAlongAxis applies
// the returned taps along that axis.
//
// The weight is the fractional contribution of that input sample to the output
// value - i.e. the interpolation kernel evaluated at the distance between the
// output's (fractional) source coordinate and that input sample. It is a pure
// geometric coefficient (independent of the data): larger weight = the sample
// is closer / more influential. For a well-formed position the weights sum to
// 1.0, so a flat input is reproduced exactly (a region of constant value stays
// that value). Examples:
//   nearest -> the single tap has weight 1.0 (copy the closest sample).
//   linear  -> two taps with weights (1 - frac) and frac, where frac is how far
//              the source coord sits between the two neighbors (e.g. exactly
//              halfway -> 0.5 / 0.5, a plain average).
//   cubic   -> four taps whose weights come from the cubic kernel; the nearer
//              two are positive and the outer two are typically small/negative.
//
// Return value: taps[o] is the list of taps for output position o, so the
// result has one inner list per output position (outer size == outLen). The
// number of taps per position depends only on the mode:
//   nearest -> 1 tap   (the single closest sample, weight 1.0)
//   linear  -> 2 taps  (the two neighbors bracketing the source coord)
//   cubic   -> 4 taps  (the cubic kernel's 4-sample support)
//
// Mental model: we walk the OUTPUT positions and, for each, ask "where does
// this land in the INPUT?" (its fractional "source coordinate"), then gather
// the nearby input samples as taps. coordinate_transformation_mode is simply
// the formula for that output -> input mapping; mode is how we blend around it.
//
// Example 1 - asymmetric, nearest. inLen=4, outLen=2,
// scale=0.5 (downsample by 2). "asymmetric" puts input samples at integer
// coordinates and maps  src = out_index / scale, with no half-pixel shift:
//
//   input idx:      0     1     2     3      (values, say:  a  b  c  d)
//   input coord:    0     1     2     3
//                   |     |     |     |
//   out[0] src=0    •                        nearest -> idx 0    taps: {0, 1.0}
//   out[1] src=2                •            nearest -> idx 2    taps: {2, 1.0}
//
//   returns: [ [{0,1.0}], [{2,1.0}] ]        -> output = [a, c]
//
// Example 2 - linear, half_pixel. Same sizes/scale, but "half_pixel" treats
// samples as cell centers (coord = idx + 0.5) and maps
// src = (out_index + 0.5) / scale - 0.5, so an output can land *between* two
// input samples and we take 2 taps that blend them:
//
//   input idx:      0     1     2     3
//   input coord:   0.5   1.5   2.5   3.5
//                   |     |     |     |
//   out[0] src=0.5  •-----'                 taps: {0, 0.5}, {1, 0.5}
//   out[1] src=2.5              •-----'      taps: {2, 0.5}, {3, 0.5}
//
//   returns: [ [{0,0.5},{1,0.5}],           <- taps for out[0]
//              [{2,0.5},{3,0.5}] ]           <- taps for out[1]
//
// Out-of-range indices (near the borders) are clamped into [0, inLen-1]; with
// excludeOutside those outside taps get zero weight and the remaining weights
// are renormalized so each output position's weights still sum to 1.
SmallVector<SmallVector<ResizeTap>> resizeBuildAxisTaps(int64_t inLen,
    int64_t outLen, double scale, StringRef mode, StringRef ctm,
    StringRef nearestMode, double cubicA, bool excludeOutside) {
  SmallVector<SmallVector<ResizeTap>> taps(outLen);
  for (int64_t outIdx = 0; outIdx < outLen; ++outIdx) {
    double src = resizeSourceCoord(outIdx, scale, inLen, outLen, ctm);
    SmallVector<ResizeTap> &posTaps = taps[outIdx];

    // Records one interpolation tap (input sample `idx` contributing `weight`)
    // and accumulates `weightSum` for later normalization. Handles the border:
    // an out-of-range sample either contributes nothing (exclude_outside) or is
    // pulled to the nearest edge pixel (clamp).
    double weightSum = 0.0;
    auto addTap = [&](int64_t idx, double weight) {
      if (idx < 0 || idx >= inLen) {
        if (excludeOutside)
          weight = 0.0;
        idx = std::clamp<int64_t>(idx, 0, inLen - 1);
      }
      posTaps.push_back({idx, weight});
      weightSum += weight;
    };

    // An interpolated value is a weighted average, so the tap weights must add
    // up to 1. Near a border some taps get dropped (exclude_outside) or merged
    // by clamping, which can make the total drift from 1; dividing every weight
    // by the actual total restores that. Without it, e.g. an all-5.0 input
    // region could resize to something other than 5.0.
    auto normalizeTaps = [&]() {
      if (weightSum != 0.0)
        llvm::for_each(
            posTaps, [&](ResizeTap &tap) { tap.weight /= weightSum; });
    };

    if (mode == "nearest") {
      // Pick the single closest sample; weight 1.0, nothing to normalize.
      int64_t idx = std::clamp<int64_t>(
          resizeNearestIndex(src, nearestMode), 0, inLen - 1);
      posTaps.push_back({idx, 1.0});
    } else if (mode == "linear") {
      // Blend the two samples bracketing src: leftIdx and leftIdx+1 (right).
      // `frac` in [0,1) is how far src lies past leftIdx, so the right sample
      // gets weight `frac` and the left gets the rest.
      auto leftIdx = static_cast<int64_t>(std::floor(src));
      double frac = src - leftIdx;
      addTap(leftIdx, 1.0 - frac);
      addTap(leftIdx + 1, frac);
      // Interior linear taps already sum to 1; only exclude_outside (which can
      // zero a tap) requires renormalizing.
      if (excludeOutside)
        normalizeTaps();
    } else { // "cubic"
      // Blend the 4 samples baseIdx-1 .. baseIdx+2 with the cubic kernel
      // evaluated at each sample's distance from src.
      auto baseIdx = static_cast<int64_t>(std::floor(src));
      double frac = src - baseIdx;
      for (int k = -1; k <= 2; ++k)
        addTap(baseIdx + k, resizeCubicWeight(frac - k, cubicA));
      normalizeTaps();
    }
  }
  return taps;
}

// Resamples one axis of a row-major flat buffer using precomputed 1-D taps.
//
// The trick is to view the N-D row-major tensor as just three grouped ranges,
// [outerLen, axisLen, innerLen], where:
//   outerLen = product of the dims BEFORE `axis`  (slices that repeat the pass)
//   innerLen = product of the dims AFTER  `axis`   (contiguous block per elem)
// Only the middle range changes length (inAxisLen -> outAxisLen)
// This makes the resample axis-agnostic: the same code handles H, W, or any
// other axis.
//
//   in  [outerLen][ inAxisLen][innerLen]
//                --taps-->
//   out [outerLen][outAxisLen][innerLen]
//
// Each output element is the tap-weighted sum of input samples along the axis
SmallVector<double> resizeAlongAxis(ArrayRef<double> in,
    ArrayRef<int64_t> inShape, int64_t axis, int64_t outAxisLen,
    ArrayRef<SmallVector<ResizeTap>> taps) {
  int64_t inAxisLen = inShape[axis];
  // outerLen/innerLen are the products of the dims before/after `axis`.
  auto product = [](ArrayRef<int64_t> dims) {
    return std::accumulate(
        dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
  };
  int64_t outerLen = product(inShape.take_front(axis));
  // If the axis is innermost dim, the range below is empty and innerLen will
  // be 1.
  int64_t innerLen = product(inShape.drop_front(axis + 1));

  // Flat offset of element (outerIdx, axisIdx, innerIdx) in an
  // [outerLen][axisLen][innerLen] row-major buffer. Input and output share this
  // layout and differ only in the axis length (inAxisLen vs outAxisLen).
  auto offset = [&](int64_t outerIdx, int64_t axisIdx, int64_t axisLen,
                    int64_t innerIdx) {
    size_t outerStride = static_cast<size_t>(axisLen) * innerLen;
    size_t axisStride = innerLen;
    return outerIdx * outerStride + axisIdx * axisStride + innerIdx;
  };

  SmallVector<double> out(
      static_cast<size_t>(outerLen) * outAxisLen * innerLen, 0.0);
  for (int64_t outerIdx : llvm::seq<int64_t>(0, outerLen)) {
    for (int64_t outAxisIdx : llvm::seq<int64_t>(0, outAxisLen)) {
      ArrayRef<ResizeTap> positionTaps = taps[outAxisIdx];
      for (int64_t innerIdx : llvm::seq<int64_t>(0, innerLen)) {
        out[offset(outerIdx, outAxisIdx, outAxisLen, innerIdx)] =
            std::accumulate(positionTaps.begin(), positionTaps.end(), 0.0,
                [&](double acc, const ResizeTap &tap) {
                  return acc + tap.weight * in[offset(outerIdx, tap.index,
                                                inAxisLen, innerIdx)];
                });
      }
    }
  }
  return out;
}

bool isResizeConstPropagatable(Operation *op) {
  auto resizeOp = cast<ONNXResizeOp>(op);
  auto inType = dyn_cast<RankedTensorType>(resizeOp.getX().getType());
  auto outType = dyn_cast<RankedTensorType>(resizeOp.getResult().getType());
  if (!inType || !outType || !inType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;
  if (inType.getRank() != outType.getRank())
    return false;
  // Restrict to float element types
  if (!isa<FloatType>(inType.getElementType()))
    return false;
  // roi is only considered in tf_crop_and_resize mode.
  // Since tf_crop_and_resize mode is not supported,
  // roi is not supported.
  if (!isa<NoneType>(resizeOp.getRoi().getType()))
    return false;
  if (resizeOp.getAxesAttr())
    return false;
  if (resizeOp.getAntialias() != 0)
    return false;
  static constexpr StringRef kSupportedModes[] = {"nearest", "linear", "cubic"};
  if (!llvm::is_contained(kSupportedModes, resizeOp.getMode()))
    return false;
  static constexpr StringRef kSupportedCTMs[] = {"half_pixel",
      "half_pixel_symmetric", "pytorch_half_pixel", "align_corners",
      "asymmetric"};
  return llvm::is_contained(
      kSupportedCTMs, resizeOp.getCoordinateTransformationMode());
}

Value ConstPropResize(
    PatternRewriter &rewriter, Value replacingValue, Value dataValue) {
  auto resizeOp = cast<ONNXResizeOp>(replacingValue.getDefiningOp());
  auto inType = cast<RankedTensorType>(dataValue.getType());
  auto outType = cast<ShapedType>(replacingValue.getType());
  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  int64_t rank = inType.getRank();

  StringRef mode = resizeOp.getMode();
  StringRef ctm = resizeOp.getCoordinateTransformationMode();
  StringRef nearestMode = resizeOp.getNearestMode();
  double cubicA = resizeOp.getCubicCoeffAAttr().getValueAsDouble();
  bool excludeOutside = resizeOp.getExcludeOutside() != 0;

  // Per-axis scale: prefer explicit 'scales' when present, else derive from
  // the output/input shapes (the 'sizes' path).
  SmallVector<double> scales = llvm::to_vector(
      llvm::map_range(llvm::zip_equal(outShape, inShape), [](const auto &dims) {
        return static_cast<double>(std::get<0>(dims)) / std::get<1>(dims);
      }));
  Value scalesVal = resizeOp.getScales();
  if (!isa<NoneType>(scalesVal.getType())) {
    ArrayBuffer<WideNum> scalesBuf =
        getElementsWideNums(getConstValueElements(scalesVal));
    ArrayRef<WideNum> s = scalesBuf.get();
    if (static_cast<int64_t>(s.size()) == rank)
      scales = llvm::to_vector(
          llvm::map_range(s, [](WideNum num) { return num.dbl; }));
  }

  // Read the constant data as doubles.
  ArrayBuffer<WideNum> dataBuf =
      getElementsWideNums(getConstValueElements(dataValue));
  SmallVector<double> buffer = llvm::to_vector(
      llvm::map_range(dataBuf.get(), [](WideNum num) { return num.dbl; }));

  // Separable resize: an N-D interpolation is done as independent 1-D passes,
  // one axis at a time, each consuming the previous pass's output. E.g. 2-D
  // bilinear = interpolate along X, then interpolate along Y of that result:
  //   10 20      x@0.25   12.5     y@0.25
  //   30 40   ---------->  32.5  ---------->  17.5
  // The per-axis weights are independent (they only depend on that axis's
  // in/out length); the data flows through the axes sequentially. `curShape`
  // tracks the running shape so each pass strides over the intermediate buffer.
  SmallVector<int64_t> curShape(inShape.begin(), inShape.end());
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (scales[axis] == 1.0)
      continue;
    SmallVector<SmallVector<ResizeTap>> taps =
        resizeBuildAxisTaps(curShape[axis], outShape[axis], scales[axis], mode,
            ctm, nearestMode, cubicA, excludeOutside);
    buffer = resizeAlongAxis(buffer, curShape, axis, outShape[axis], taps);
    curShape[axis] = outShape[axis];
  }

  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements =
      elementsBuilder.fromWideNums(outType, [&](MutableArrayRef<WideNum> dst) {
        for (size_t i = 0; i < dst.size(); ++i)
          dst[i] = WideNum::widen<BType::DOUBLE>(buffer[i]);
      });
  return createReplacingConstantOp(rewriter, replacingValue, resultElements);
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

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {
    if (!isDenseONNXConstant(splitOp.getInput()))
      return failure();
    Value split = splitOp.getSplit();
    if (!(isa<NoneType>(split.getType()) || isDenseONNXConstant(split)))
      return failure();

    rewriter.replaceOp(splitOp,
        ConstPropSplit(rewriter, splitOp.getResults(), splitOp.getInput(),
            splitOp.getSplit(), splitOp.getAxis()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Loop unrolling: LoopUnroll.
//
// Unrolls an onnx.Loop with a statically-known, bounded trip count into N
// copies of the loop body inlined into the parent block.  After unrolling,
// the standard constprop patterns (ConstPropRange, ConstPropGather, …) handle
// the folding of the resulting ops automatically, without any per-op special
// casing here.
//
// Match conditions:
//   • NoneType condition input (always run for exactly M trips)
//   • Constant dense trip-count M in (0, maxLoopUnrollCount]
//===----------------------------------------------------------------------===//

class LoopUnroll : public OpRewritePattern<ONNXLoopOp> {
  int64_t maxLoopUnrollCount;

public:
  LoopUnroll(MLIRContext *context, int64_t maxLoopUnrollCount)
      : OpRewritePattern<ONNXLoopOp>(context),
        maxLoopUnrollCount(maxLoopUnrollCount) {}

  LogicalResult matchAndRewrite(
      ONNXLoopOp loopOp, PatternRewriter &rewriter) const override {
    // The loop unrolling works only if MaxTripCount is a constant i64 scalar
    // (as required by the ONNX spec: M is tensor of int64).
    SmallVector<int64_t, 1> mVals;
    if (!getI64ValuesFromONNXConstantOp(loopOp.getM(), mVals))
      return rewriter.notifyMatchFailure(
          loopOp, "trip count must be a constant i64 scalar");
    int64_t M = mVals[0];

    if (M < 0 || M > maxLoopUnrollCount)
      return rewriter.notifyMatchFailure(
          loopOp, "M is out of the configured unrollable range");

    MLIRContext *ctx = rewriter.getContext();
    Location loc = loopOp.getLoc();
    Block &body = loopOp.getRegion().front();
    auto yieldOp = cast<ONNXYieldOp>(body.getTerminator());

    // Determine whether the loop is guaranteed to run exactly M iterations.
    // Two accepted forms:
    //   1. NoneType condition: ONNX spec says the loop runs exactly M trips.
    //   2. Constant-true initial condition AND body always yields constant
    //   true:
    //      semantically equivalent to NoneType when both are statically known.
    bool condIsNone = isa<NoneType>(loopOp.getCond().getType());
    auto getConstBool = [](Value v) -> std::optional<bool> {
      if (!isDenseONNXConstant(v))
        return std::nullopt;
      auto elems = getConstValueElements(v);
      return (*elems.value_begin<APInt>()).getBoolValue();
    };
    // The body's yielded condition (operand 0) guarantees always-true when it
    // is either a constant true, or the same block argument the body received
    // (passthrough) — in which case the initial true propagates unchanged.
    Value yieldedCond = yieldOp.getOperand(0);
    bool yieldAlwaysTrue =
        getConstBool(yieldedCond) == std::optional<bool>(true) ||
        yieldedCond == body.getArgument(1);
    bool condIsAlwaysTrue =
        getConstBool(loopOp.getCond()) == std::optional<bool>(true) &&
        yieldAlwaysTrue;
    if (!condIsNone && !condIsAlwaysTrue)
      return rewriter.notifyMatchFailure(loopOp,
          "Condition must be NoneType or a constant-true value with the body "
          "always yielding true");
    const auto numCarried = static_cast<int64_t>(loopOp.getVInitial().size());
    const auto numResults = static_cast<int64_t>(loopOp.getNumResults());
    const int64_t numScanOutputs = numResults - numCarried;

    // M = 0: the loop body never executes.
    // Carried outputs are the unchanged v_initial values.
    // Scan outputs are zero-element tensors with the correct element type.
    if (M == 0) {
      rewriter.setInsertionPoint(loopOp);
      OnnxBuilder ob0(rewriter, loc);
      SmallVector<Value> zeroOutputs(
          loopOp.getVInitial().begin(), loopOp.getVInitial().end());
      for (int64_t k = 0; k < numScanOutputs; ++k) {
        Type scanResultTy = loopOp.getResult(numCarried + k).getType();
        Type elemTy = rewriter.getF32Type(); // fallback
        SmallVector<int64_t> shape = {0};
        if (auto rt = dyn_cast<RankedTensorType>(scanResultTy)) {
          elemTy = rt.getElementType();
          // Shape: [0, D1, ..., Dn] (leading dim = 0, inner dims preserved).
          shape.append(rt.getShape().begin() + 1, rt.getShape().end());
        } else if (auto st = dyn_cast<ShapedType>(scanResultTy)) {
          elemTy = st.getElementType();
        }
        auto emptyTy = RankedTensorType::get(shape, elemTy);
        zeroOutputs.push_back(ob0.constant(
            DenseElementsAttr::get(emptyTy, llvm::ArrayRef<Attribute>{})));
      }
      rewriter.replaceOp(loopOp, zeroOutputs);
      return success();
    }

    OnnxBuilder ob(rewriter, loc);

    // Helper: scalar i64 constant for the loop-iteration counter.
    auto makeIterConst = [&ob, ctx](int64_t i) -> Value {
      auto ty = RankedTensorType::get({}, IntegerType::get(ctx, 64));
      return ob.constant(
          DenseElementsAttr::get(ty, APInt(64, i, /*isSigned=*/true)));
    };
    // Helper: scalar bool constant (true) for the loop condition arg.
    auto makeTrueConst = [&ob, ctx]() -> Value {
      auto ty = RankedTensorType::get({}, IntegerType::get(ctx, 1));
      return ob.constant(DenseElementsAttr::get(ty, APInt(1, 1)));
    };

    // Current loop-carried values start as the loop's v_initial operands.
    SmallVector<Value> carried(
        loopOp.getVInitial().begin(), loopOp.getVInitial().end());

    // Per-scan-output: one Value per iteration, to be concatenated afterwards.
    SmallVector<SmallVector<Value>> scanContribs(numScanOutputs);

    // --- Unroll M iterations ---
    rewriter.setInsertionPoint(loopOp);
    for (int64_t i = 0; i < M; ++i) {
      IRMapping map;
      // arg0 = iteration counter (i64 scalar)
      // arg1 = loop condition (always true for NoneType cond loops)
      // arg2 … = loop-carried values
      map.map(body.getArgument(0), makeIterConst(i));
      map.map(body.getArgument(1), makeTrueConst());
      for (int64_t j = 0; j < numCarried; ++j)
        map.map(body.getArgument(2 + j), carried[j]);

      // Clone every op in the body except the yield terminator.
      // Set each cloned op's location to the original loop's location so that
      // diagnostic messages and debug info refer to the loop, not the body op.
      for (Operation &op : body.getOperations()) {
        if (&op == yieldOp.getOperation())
          break;
        Operation *cloned = rewriter.clone(op, map);
        cloned->setLoc(loc);
      }

      // Advance carried values and record scan contributions.
      // Yield operand layout: [cond, carried…, scan…]
      for (int64_t j = 0; j < numCarried; ++j)
        carried[j] = map.lookupOrDefault(yieldOp.getOperand(1 + j));
      for (int64_t k = 0; k < numScanOutputs; ++k)
        scanContribs[k].push_back(
            map.lookupOrDefault(yieldOp.getOperand(1 + numCarried + k)));
    }

    // --- Build replacement values ---
    // Loop-carried results: the final `carried` values after M iterations.
    SmallVector<Value> outputs(carried.begin(), carried.end());

    // Scan outputs: unsqueeze each per-iteration contribution (adding a
    // leading axis-0 dimension) then concatenate into one tensor.
    // These Unsqueeze/Concat ops will be folded by subsequent constprop
    // passes if the contributions turn out to be constants.
    //
    // Type strategy: anchor on the original loop result type so downstream
    // ops see the same static type they had before unrolling, without waiting
    // for shape inference to propagate through the new Unsqueeze/Concat.
    //   loopResultTy  = tensor<M x D1 x ... x Dn>  (concat output)
    //   unsqueezedTy  = tensor<1 x D1 x ... x Dn>  (each iteration's slice)
    // Fall back to contribution-derived or unranked types when the loop
    // result type isn't ranked (should not happen in practice).
    Value axes0 = ob.constantInt64({0}); // axes = [0] for unsqueeze
    for (int64_t k = 0; k < numScanOutputs; ++k) {
      SmallVector<Value> &contribs = scanContribs[k];
      assert(!contribs.empty() &&
             "scan output must have at least one contribution");

      // Derive types from the original loop result.
      Type loopResultTy = loopOp.getResult(numCarried + k).getType();
      Type concatTy = loopResultTy;

      // Each per-iteration contribution, after unsqueeze(axes=[0]), has the
      // same shape as loopResultTy but with the leading dim clamped to 1.
      Type unsqueezedTy;
      if (auto rt = dyn_cast<RankedTensorType>(loopResultTy)) {
        SmallVector<int64_t> sliceShape = {1};
        sliceShape.append(rt.getShape().begin() + 1, rt.getShape().end());
        unsqueezedTy = RankedTensorType::get(sliceShape, rt.getElementType());
      } else if (auto st = dyn_cast<ShapedType>(contribs[0].getType())) {
        // Contribution is already ranked: derive unsqueezed shape from it.
        SmallVector<int64_t> sliceShape = {1};
        sliceShape.append(st.getShape().begin(), st.getShape().end());
        unsqueezedTy = RankedTensorType::get(sliceShape, st.getElementType());
        concatTy = UnrankedTensorType::get(st.getElementType());
      } else {
        Type fallback = UnrankedTensorType::get(rewriter.getF32Type());
        unsqueezedTy = fallback;
        concatTy = fallback;
      }

      SmallVector<Value> unsqueezed;
      unsqueezed.reserve(contribs.size());
      for (Value contrib : contribs)
        unsqueezed.push_back(ob.unsqueeze(unsqueezedTy, contrib, axes0));

      Value scanOut = (unsqueezed.size() == 1)
                          ? unsqueezed[0]
                          : ob.concat(concatTy, unsqueezed, /*axis=*/0);
      outputs.push_back(scanOut);
    }

    rewriter.replaceOp(loopOp, outputs);
    return success();
  }
};

class IfOfConst : public OpRewritePattern<ONNXIfOp> {
public:
  using OpRewritePattern<ONNXIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXIfOp ifOp, PatternRewriter &rewriter) const override {
    if (!isDenseONNXConstant(ifOp.getCond()))
      return failure();

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
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant propagation for ConcatFromSequenceOp.
//
// When the input sequence is built entirely from constant tensors via a
// SequenceEmpty → SequenceInsert … chain, fold the whole ConcatFromSequence
// into a single constant tensor.
//===----------------------------------------------------------------------===//

/// Walk a SequenceInsert chain and collect element ElementsAttrs in order
/// (first-inserted first).  Only handles append-mode inserts (NoneType
/// position).  Returns false on failure.
static bool collectConstSequenceElems(
    Value seq, SmallVectorImpl<ElementsAttr> &elems) {
  SmallVector<Value> elemsRev;
  while (true) {
    if (isa_and_nonnull<ONNXSequenceEmptyOp>(seq.getDefiningOp()))
      break;
    auto ins = dyn_cast_or_null<ONNXSequenceInsertOp>(seq.getDefiningOp());
    if (!ins)
      return false;
    if (!isa<NoneType>(ins.getPosition().getType()))
      return false; // only handle append, not random-access insert
    if (!isDenseONNXConstant(ins.getTensor()))
      return false;
    elemsRev.push_back(ins.getTensor());
    seq = ins.getInputSequence();
  }
  for (Value v : llvm::reverse(elemsRev))
    elems.push_back(getConstValueElements(v));
  return true;
}

class ConstPropConcatFromSequence
    : public OpRewritePattern<ONNXConcatFromSequenceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatFromSequenceOp op, PatternRewriter &rewriter) const override {
    SmallVector<ElementsAttr> elems;
    if (!collectConstSequenceElems(op.getInputSequence(), elems))
      return failure();
    if (elems.empty())
      return failure();

    int64_t axis = op.getAxis();
    bool newAxisFlag = op.getNewAxis() != 0;
    int64_t rank = cast<ShapedType>(elems[0].getType()).getRank();
    if (axis < 0)
      axis += rank + (newAxisFlag ? 1 : 0);

    OnnxElementsAttrBuilder elemBuilder(rewriter.getContext());
    SmallVector<ElementsAttr> toConcat;
    toConcat.reserve(elems.size());
    for (auto &e : elems) {
      if (newAxisFlag) {
        // Stack: insert a size-1 dimension at `axis`.
        auto shape = cast<ShapedType>(e.getType()).getShape();
        SmallVector<int64_t> newShape(shape.begin(), shape.begin() + axis);
        newShape.push_back(1);
        newShape.append(shape.begin() + axis, shape.end());
        toConcat.push_back(elemBuilder.reshape(e, newShape));
      } else {
        toConcat.push_back(e);
      }
    }
    ElementsAttr result = elemBuilder.concat(toConcat, (unsigned)axis);
    Value constVal =
        createReplacingConstantOp(rewriter, op.getResult(), result);
    rewriter.replaceOp(op, constVal);
    return success();
  }
};

// Q-DQ Removal to enable const-folding through data reformatting ops
template <typename ONNXOp>
class RemoveQDQForConst : public OpRewritePattern<ONNXOp> {
public:
  using OpRewritePattern<ONNXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXOp op, PatternRewriter &rewriter) const override {
    // Only first operand is considered
    auto dqOp =
        op->getOperand(0).template getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return rewriter.notifyMatchFailure(op, "DQ not found");

    auto constOp = dqOp.getX().template getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return rewriter.notifyMatchFailure(op, "Not a constant input");

    // Only first result is considered
    auto result = op->getResult(0);
    auto qOp = dyn_cast<ONNXQuantizeLinearOp>(*result.user_begin());
    if (!qOp || !result.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "Q not found or has multiple uses");

    if (dqOp.getXScale() != qOp.getYScale() ||
        dqOp.getXZeroPoint() != qOp.getYZeroPoint() ||
        getElementTypeOrSelf(dqOp.getX()) != getElementTypeOrSelf(qOp.getY()))
      return rewriter.notifyMatchFailure(op, "Q & DQ are not equivalent");

    SmallVector<Value> operands = op->getOperands();
    operands[0] = constOp;
    SmallVector<NamedAttribute> attrs(op->getAttrs());
    llvm::erase_if(attrs, [](NamedAttribute attr) {
      auto strRef = attr.getName().strref();
      return (strRef == "onnx_node_name" || strRef == "ResultNames");
    });

    rewriter.replaceOpWithNewOp<ONNXOp>(qOp, qOp.getType(), operands, attrs);

    return success();
  }
};

// Fold Const(fp) -> QuantizeLinear into Const(int)
class ConstFoldQuantizeLinearOnConst
    : public OpRewritePattern<ONNXQuantizeLinearOp> {
public:
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {
    // Blocked quantization is out of scope.
    if (qOp.getBlockSize() != 0)
      return rewriter.notifyMatchFailure(qOp, "blocked quantization");

    // The input and quantization parameters must all be constants.
    ElementsAttr xElems = getDenseOrDisposableConstLikeElements(qOp.getX());
    if (!xElems)
      return rewriter.notifyMatchFailure(qOp, "x is not a constant");
    if (!isa<FloatType>(xElems.getElementType()))
      return rewriter.notifyMatchFailure(qOp, "x is not floating point");

    ElementsAttr scaleElems =
        getDenseOrDisposableConstLikeElements(qOp.getYScale());
    if (!scaleElems)
      return rewriter.notifyMatchFailure(qOp, "y_scale is not a constant");

    Value zpValue = qOp.getYZeroPoint();
    bool hasZeroPoint = !isNoneValue(zpValue);
    ElementsAttr zpElems;
    if (hasZeroPoint) {
      zpElems = getDenseOrDisposableConstLikeElements(zpValue);
      if (!zpElems)
        return rewriter.notifyMatchFailure(
            qOp, "y_zero_point is not a constant");
    }

    auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(qOp.getY()));
    if (!intType)
      return rewriter.notifyMatchFailure(qOp, "non-integer output type");

    auto xType = cast<ShapedType>(qOp.getX().getType());
    if (!xType.hasStaticShape())
      return rewriter.notifyMatchFailure(qOp, "x has dynamic shape");
    int64_t xRank = xType.getRank();
    ArrayRef<int64_t> xShape = xType.getShape();

    int64_t scaleElemCount =
        cast<ShapedType>(scaleElems.getType()).getNumElements();
    int64_t zpElemCount =
        hasZeroPoint ? cast<ShapedType>(zpElems.getType()).getNumElements() : 1;
    bool isPerAxis = scaleElemCount > 1 || zpElemCount > 1;

    int64_t axis = qOp.getAxis();
    if (isPerAxis) {
      if (axis < 0)
        axis += xRank;
      if (axis < 0 || axis >= xRank)
        return rewriter.notifyMatchFailure(qOp, "axis out of range");

      auto isFoldablePerAxisParam = [&](ElementsAttr elems,
                                        int64_t count) -> bool {
        if (count == 1)
          return true; // Scalar broadcasts trivially.
        auto type = cast<ShapedType>(elems.getType());
        return type.getRank() == 1 && count == xShape[axis];
      };
      if (!isFoldablePerAxisParam(scaleElems, scaleElemCount) ||
          (hasZeroPoint && !isFoldablePerAxisParam(zpElems, zpElemCount)))
        return rewriter.notifyMatchFailure(
            qOp, "unsupported per-axis scale/zp");
    }

    FloatType f64Type = rewriter.getF64Type();
    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());

    // Reshape a per-axis scale/zp (1-D of length xShape[axis]) so it
    // broadcasts against x's shape; per-tensor (scalar) needs no reshape.
    auto broadcastToX = [&](ElementsAttr elems) -> ElementsAttr {
      auto type = cast<ShapedType>(elems.getType());
      int64_t numElems = type.getNumElements();
      if (numElems == 1)
        return elems; // Per-tensor: scalar broadcasts trivially.
      SmallVector<int64_t> bcastShape(xRank, 1);
      bcastShape[axis] = numElems;
      return elementsBuilder.reshape(elems, bcastShape);
    };

    ShapedType combinedType = cast<ShapedType>(
        elementsBuilder.castToFPElementType(xElems, f64Type).getType());

    // scaled = x / scale (both in f64).
    ElementsAttr xF64 = elementsBuilder.castToFPElementType(xElems, f64Type);
    ElementsAttr scaleF64 =
        broadcastToX(elementsBuilder.castToFPElementType(scaleElems, f64Type));
    ElementsAttr scaled = elementsBuilder.combine(xF64, scaleF64, combinedType,
        elementwiseBinaryOpCombiner<ONNXDivOp>(f64Type));

    // shifted = (x / scale) + zero_point
    ElementsAttr shifted = scaled;
    if (hasZeroPoint) {
      ElementsAttr zpF64 =
          broadcastToX(elementsBuilder.castToFPElementType(zpElems, f64Type));
      shifted = elementsBuilder.combine(
          scaled, zpF64, combinedType, addCombiner(f64Type));
    }

    ElementsAttr rounded =
        elementsBuilder.transform(shifted, f64Type, [](WideNum n) {
          int savedMode = fegetround();
          fesetround(FE_TONEAREST);
          double r = std::nearbyint(n.narrow<BType::DOUBLE>());
          fesetround(savedMode);
          return WideNum::widen<BType::DOUBLE>(r);
        });

    ElementsAttr quantized =
        elementsBuilder.castToIntElementType(rounded, intType, /*round=*/false);

    rewriter.replaceOp(
        qOp, createReplacingConstantOp(rewriter, qOp.getY(), quantized));
    return success();
  }
};

// True if `v` is a constant, or a DequantizeLinear of a constant.
static bool isConstantOrDequantizeOfConstant(Value v) {
  if (getDenseOrDisposableConstLikeElements(v))
    return true;
  auto dq = v.getDefiningOp<ONNXDequantizeLinearOp>();
  if (!dq)
    return false;
  if (!getDenseOrDisposableConstLikeElements(dq.getX()) ||
      !getDenseOrDisposableConstLikeElements(dq.getXScale()))
    return false;
  Value zp = dq.getXZeroPoint();
  return isNoneValue(zp) || getDenseOrDisposableConstLikeElements(zp);
}

// True if `value` is only consumed by constant computation ending at a
// QuantizeLinear or graph output -- never by an op with a non-constant operand.
// This keeps the fold off ordinary quantized weights (whose consumer mixes in a
// non-constant activation).
static bool onlyFeedsConstantIsland(Value value) {
  for (Operation *user : value.getUsers()) {
    if (isa<ONNXQuantizeLinearOp>(user))
      continue;
    if (user->hasTrait<OpTrait::IsTerminator>())
      continue;
    for (Value operand : user->getOperands()) {
      if (operand == value || isNoneValue(operand))
        continue;
      if (!isConstantOrDequantizeOfConstant(operand))
        return false;
    }
    for (Value result : user->getResults())
      if (!onlyFeedsConstantIsland(result))
        return false;
  }
  return true;
}

// Fold DequantizeLinear on constants to `(x - x_zero_point) * x_scale`, the
// inverse of ConstFoldQuantizeLinearOnConst, gated by
// enable-dequant-const-fold. Per-tensor and per-axis are handled; blocked
// quantization is out of scope. Confined to constant islands
// (onlyFeedsConstantIsland) so quantized weights are not dequantized.
class ConstFoldDequantizeLinearOnConst
    : public OpRewritePattern<ONNXDequantizeLinearOp> {
public:
  using OpRewritePattern<ONNXDequantizeLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDequantizeLinearOp dqOp, PatternRewriter &rewriter) const override {
    // Blocked quantization is out of scope.
    if (dqOp.getBlockSize() != 0)
      return rewriter.notifyMatchFailure(dqOp, "blocked quantization");

    // The input and quantization parameters must all be constants.
    ElementsAttr xElems = getDenseOrDisposableConstLikeElements(dqOp.getX());
    if (!xElems)
      return rewriter.notifyMatchFailure(dqOp, "x is not a constant");

    ElementsAttr scaleElems =
        getDenseOrDisposableConstLikeElements(dqOp.getXScale());
    if (!scaleElems)
      return rewriter.notifyMatchFailure(dqOp, "x_scale is not a constant");

    Value zpValue = dqOp.getXZeroPoint();
    bool hasZeroPoint = !isNoneValue(zpValue);
    ElementsAttr zpElems;
    if (hasZeroPoint) {
      zpElems = getDenseOrDisposableConstLikeElements(zpValue);
      if (!zpElems)
        return rewriter.notifyMatchFailure(
            dqOp, "x_zero_point is not a constant");
    }

    auto floatType = dyn_cast<FloatType>(getElementTypeOrSelf(dqOp.getY()));
    if (!floatType)
      return rewriter.notifyMatchFailure(dqOp, "non-floating-point output");

    // Only fold inside a constant island, not ordinary quantized weights.
    if (!onlyFeedsConstantIsland(dqOp.getY()))
      return rewriter.notifyMatchFailure(
          dqOp, "not confined to a constant island");

    auto xType = cast<ShapedType>(dqOp.getX().getType());
    if (!xType.hasStaticShape())
      return rewriter.notifyMatchFailure(dqOp, "x has dynamic shape");
    int64_t xRank = xType.getRank();
    ArrayRef<int64_t> xShape = xType.getShape();

    int64_t scaleElemCount =
        cast<ShapedType>(scaleElems.getType()).getNumElements();
    int64_t zpElemCount =
        hasZeroPoint ? cast<ShapedType>(zpElems.getType()).getNumElements() : 1;
    bool isPerAxis = scaleElemCount > 1 || zpElemCount > 1;

    int64_t axis = dqOp.getAxis();
    if (isPerAxis) {
      if (axis < 0)
        axis += xRank;
      if (axis < 0 || axis >= xRank)
        return rewriter.notifyMatchFailure(dqOp, "axis out of range");

      auto isFoldablePerAxisParam = [&](ElementsAttr elems,
                                        int64_t count) -> bool {
        if (count == 1)
          return true; // Scalar broadcasts trivially.
        auto type = cast<ShapedType>(elems.getType());
        return type.getRank() == 1 && count == xShape[axis];
      };
      if (!isFoldablePerAxisParam(scaleElems, scaleElemCount) ||
          (hasZeroPoint && !isFoldablePerAxisParam(zpElems, zpElemCount)))
        return rewriter.notifyMatchFailure(
            dqOp, "unsupported per-axis scale/zp");
    }

    FloatType f64Type = rewriter.getF64Type();
    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());

    // Reshape a per-axis scale/zp (1-D of length xShape[axis]) so it
    // broadcasts against x's shape; per-tensor (scalar) needs no reshape.
    auto broadcastToX = [&](ElementsAttr elems) -> ElementsAttr {
      auto type = cast<ShapedType>(elems.getType());
      int64_t numElems = type.getNumElements();
      if (numElems == 1)
        return elems; // Per-tensor: scalar broadcasts trivially.
      SmallVector<int64_t> bcastShape(xRank, 1);
      bcastShape[axis] = numElems;
      return elementsBuilder.reshape(elems, bcastShape);
    };

    ShapedType combinedType = cast<ShapedType>(
        elementsBuilder.castToFPElementType(xElems, f64Type).getType());

    // shifted = x - zero_point (both in f64).
    ElementsAttr xF64 = elementsBuilder.castToFPElementType(xElems, f64Type);
    ElementsAttr shifted = xF64;
    if (hasZeroPoint) {
      ElementsAttr zpF64 =
          broadcastToX(elementsBuilder.castToFPElementType(zpElems, f64Type));
      shifted = elementsBuilder.combine(
          xF64, zpF64, combinedType, subCombiner(f64Type));
    }

    // dequantized = (x - zero_point) * scale (still in f64).
    ElementsAttr scaleF64 =
        broadcastToX(elementsBuilder.castToFPElementType(scaleElems, f64Type));
    ElementsAttr dequantizedF64 = elementsBuilder.combine(shifted, scaleF64,
        combinedType, elementwiseBinaryOpCombiner<ONNXMulOp>(f64Type));

    // Cast down to the DequantizeLinear result's floating-point type.
    ElementsAttr dequantized =
        elementsBuilder.castToFPElementType(dequantizedF64, floatType);

    rewriter.replaceOp(
        dqOp, createReplacingConstantOp(rewriter, dqOp.getY(), dequantized));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstPropONNXToONNXPass)

  Option<bool> enableQDQ{*this, "enable-qdq", llvm::cl::init(true)};

  Option<bool> enableQuantConstFold{
      *this, "enable-quant-const-fold", llvm::cl::init(false)};

  Option<bool> enableDequantConstFold{
      *this, "enable-dequant-const-fold", llvm::cl::init(false)};

  Option<int64_t> maxLoopUnrollCount{*this, "max-loop-unroll-count",
      llvm::cl::desc("Maximum constant onnx.Loop trip count to unroll."),
      llvm::cl::init(64)};

  ConstPropONNXToONNXPass(bool enableQDQ, bool enableQuantConstFold,
      int64_t maxLoopUnrollCount, bool enableDequantConstFold) {
    this->enableQDQ = enableQDQ;
    this->enableQuantConstFold = enableQuantConstFold;
    this->maxLoopUnrollCount = maxLoopUnrollCount;
    this->enableDequantConstFold = enableDequantConstFold;
  }

  ConstPropONNXToONNXPass(const ConstPropONNXToONNXPass &other) {
    copyOptionValuesFrom(&other);
  }

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
  getConstPropONNXToONNXPatterns(patterns, enableQDQ, enableQuantConstFold,
      maxLoopUnrollCount, enableDequantConstFold);
  onnx_mlir::ResultNamesUpdater rnUpdater;
  if (failed(applyPatternsGreedily(function, std::move(patterns),
          GreedyRewriteConfig().setListener(&rnUpdater))))
    signalPassFailure();
}

} // end anonymous namespace.

void onnx_mlir::getConstPropONNXToONNXPatterns(RewritePatternSet &patterns,
    bool enableQDQ, bool enableQuantConstFold, int64_t maxLoopUnrollCount,
    bool enableDequantConstFold) {
  if (isConstantPropagationDisabled())
    return;
  populateWithGenerated(patterns);
  if (isNotDisabled("SplitOfConst"))
    patterns.insert<SplitOfConst>(patterns.getContext());
  patterns.insert<IfOfConst>(patterns.getContext());
  patterns.insert<LoopUnroll>(patterns.getContext(), maxLoopUnrollCount);
  patterns.insert<ConstPropConcatFromSequence>(patterns.getContext());
  if (enableQDQ)
    patterns.add<RemoveQDQForConst<ONNXSliceOp>,
        RemoveQDQForConst<ONNXTransposeOp>, RemoveQDQForConst<ONNXReshapeOp>,
        RemoveQDQForConst<ONNXSqueezeOp>, RemoveQDQForConst<ONNXUnsqueezeOp>,
        RemoveQDQForConst<ONNXGatherOp>>(patterns.getContext());
  if (enableQuantConstFold)
    patterns.add<ConstFoldQuantizeLinearOnConst>(patterns.getContext());
  if (enableDequantConstFold)
    patterns.add<ConstFoldDequantizeLinearOnConst>(patterns.getContext());
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

void onnx_mlir::configureConstPropMaxTileFoldSize(int64_t maxTileFoldSize) {
  ConstPropONNXToONNXPassConfiguration::maxTileFoldSize = maxTileFoldSize;
}

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass(
    bool enableQDQ, bool enableQuantConstFold, int64_t maxLoopUnrollCount,
    bool enableDequantConstFold) {
  return std::make_unique<ConstPropONNXToONNXPass>(enableQDQ,
      enableQuantConstFold, maxLoopUnrollCount, enableDequantConstFold);
}
