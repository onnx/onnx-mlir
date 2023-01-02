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
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the constpropd operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
// TODO: Edit the above statement. Seems inaccurate because some of the
//       const prop functions rely on static result shape.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/BType.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

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

/// A helper function to construct a RankedTensorType from a ShapedType.
ATTRIBUTE(unused) RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// A helper function to check whether a value is produced by a dense
/// ONNXConstantOp.
///
/// TODO: remove obsolete trueONNXConstant argument
bool isFromDenseONNXConstantOp(Value result, bool trueONNXConstant = false) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<::mlir::Attribute>("value"))) {
    return false;
  }
  // The other attributes must be null.
  if (op->getAttrOfType<::mlir::Attribute>("sparse_value"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_float"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_floats"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_int"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_ints"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_string"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_strings"))
    return false;

  return true;
}

/// A helper function to check whether a variadic value is produced by dense
/// ONNXConstantOps.
bool isVariadicOperandFromDenseONNXConstantOp(ValueRange operands) {
  return llvm::all_of(
      operands, [](Value v) { return isFromDenseONNXConstantOp(v); });
}

//===----------------------------------------------------------------------===//
// Helpers to support DisposableElementsAttr constant propagation.
//===----------------------------------------------------------------------===//

ElementsAttr getConstValueElements(Value constValue) {
  ONNXConstantOp constOp = getONNXConstantOp(constValue);
  return constOp.valueAttr().cast<ElementsAttr>();
}

DisposableElementsAttr getConstValueAsDisposableElements(
    ElementsAttrBuilder &elementsBuilder, Value constValue) {
  return elementsBuilder.toDisposableElementsAttr(
      getConstValueElements(constValue));
}

// Creates ONNXConstantOp with the location and result type from replacingValue.
ONNXConstantOp createReplacingConstantOp(
    PatternRewriter &rewriter, Value replacingValue, ElementsAttr elements) {
  return rewriter.create<ONNXConstantOp>(replacingValue.getLoc(),
      replacingValue.getType(), Attribute(), elements, FloatAttr(), ArrayAttr(),
      IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
}

template <typename T>
using EnableFloat = std::enable_if_t<CppTypeTrait<T>::isFloat>;

template <typename T>
using EnableNotBool = std::enable_if_t<!std::is_same_v<T, bool>>;

template <typename T>
SmallVector<T, 4> createIntVectorFromArrayAttr(ArrayAttr a) {
  SmallVector<T, 4> vec;
  for (auto val : a.getValue())
    vec.push_back(val.cast<IntegerAttr>().getInt());
  return vec;
}

ElementsAttr ConstPropReshapeImpl(PatternRewriter &rewriter,
    Value replacingValue, Value constValue, ArrayRef<int64_t> reshapedShape) {
  ElementsAttr constElements = getConstValueElements(constValue);
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
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
  static T impl(T lhs, T rhs); // Every template specialization implements this.
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAddOp, T, EnableNotBool<T>> {
  static T impl(T lhs, T rhs) { return lhs + rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSubOp, T, EnableNotBool<T>> {
  static T impl(T lhs, T rhs) { return lhs - rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMulOp, T, EnableNotBool<T>> {
  static T impl(T lhs, T rhs) { return lhs * rhs; }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXDivOp, T, EnableNotBool<T>> {
  static T impl(T lhs, T rhs) { return lhs / rhs; }
};

template <typename ElementwiseBinaryOp, typename T>
WideNum combine(WideNum lhs, WideNum rhs) {
  using W = WideBType<toBType<T>>;
  static_assert(std::is_same_v<T, typename W::type>, "T must be a wide type");
  using OpImpl = ElementWiseBinaryOpImpl<ElementwiseBinaryOp, T>;
  return W::pack(OpImpl::impl(W::unpack(lhs), W::unpack(rhs)));
}

template <typename ElementwiseBinaryOp>
auto combinerOfElementwiseBinaryOp(Type operandsElemType) {
  // TODO: change Combiner to plain function pointer
  typedef WideNum (*Combiner)(WideNum, WideNum);
  return dispatchByBType(
      btypeOfMlirType(operandsElemType), [](auto btype) -> Combiner {
        return combine<ElementwiseBinaryOp, WideType<btype>>;
      });
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
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements = elementsBuilder.combine(lhs, rhs, replacingType,
      combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsElemType));
  return createReplacingConstantOp(rewriter, replacingValue, resultElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
//// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP, typename T, class Enable = void>
struct ElementWiseUnaryOpImpl {
  static T impl(T val) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T, EnableNotBool<T>> {
  static T impl(T val) { return (-val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, T, EnableFloat<T>> {
  static T impl(T val) { return sqrt(val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXReluOp, T, EnableNotBool<T>> {
  static T impl(T val) {
    if (val < 0)
      return 0;
    return val;
  }
};

template <typename OP>
ElementsAttrBuilder::Transformer transformElementWiseUnaryOp(Type elemType) {
  return dispatchByBType(btypeOfMlirType(elemType),
      [](auto btype) -> ElementsAttrBuilder::Transformer {
        using W = WideBType<btype>;
        using OpImpl = ElementWiseUnaryOpImpl<OP, typename W::type>;
        return ElementsAttrBuilder::functionTransformer(
            [](WideNum n) -> WideNum {
              return W::pack(OpImpl::impl(W::unpack(n)));
            });
      });
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
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr transposedElements =
      elementsBuilder.transform(constElements, replacingElemType,
          transformElementWiseUnaryOp<ElementwiseUnaryOp>(replacingElemType));
  return createReplacingConstantOp(rewriter, replacingValue, transposedElements)
      .getResult();
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
  SmallVector<uint64_t, 4> perm =
      createIntVectorFromArrayAttr<uint64_t>(permAttr);

  ElementsAttr constElements = getConstValueElements(constValue);
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr transposedElements =
      elementsBuilder.transpose(constElements, perm);
  return createReplacingConstantOp(rewriter, replacingValue, transposedElements)
      .getResult();
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
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements)
      .getResult();
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
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

void SplitImpl(ArrayRef<WideNum> inputData, size_t start, size_t len,
    size_t stride, MutableArrayRef<WideNum> outputData) {
  auto in = inputData.begin();
  auto out = outputData.begin();
  for (size_t offset = start; offset < inputData.size(); offset += stride)
    out = std::copy_n(in + offset, len, out);
  assert(out == outputData.end() && "result num elements mismatch");
}

template <typename Op>
LogicalResult ConstPropSplitPatternCommon(Op splitOp, PatternRewriter &rewriter,
    llvm::Optional<ArrayAttr> splitAttr) {
  // Basic info.
  unsigned numResults = splitOp.getNumResults();
  Value input = splitOp.input();
  if (!isFromDenseONNXConstantOp(input))
    return failure();
  ConstPropCounters::count("Split", {input});
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  uint64_t splitAxis = splitOp.axis();
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

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr inputElements =
      getConstValueAsDisposableElements(elementsBuilder, input);
  ArrayBuffer<WideNum> inputData = inputElements.getWideNums();
  size_t stride = ShapedType::getNumElements(inputShape.drop_front(splitAxis));
  size_t substride = stride / splitAxisSize;
  size_t offset = 0;
  std::vector<ElementsAttr> resElements;
  resElements.reserve(numResults);
  for (unsigned int i = 0; i < numResults; ++i) {
    Type replacingType = splitOp.getResult(i).getType();
    size_t len = splitSizes[i] * substride;
    ElementsAttr splitElements = elementsBuilder.fromWideNums(
        replacingType, [&](MutableArrayRef<WideNum> outputData) {
          SplitImpl(inputData.get(), offset, len, stride, outputData);
        });
    resElements.push_back(splitElements);
    offset += len;
  }

  std::vector<Value> resValues;
  resValues.reserve(numResults);
  for (unsigned int i = 0; i < numResults; ++i) {
    Value replacingValue = splitOp.getResult(i);
    ElementsAttr splitElements = resElements[i];
    resValues.push_back(
        createReplacingConstantOp(rewriter, replacingValue, splitElements)
            .getResult());
  }
  rewriter.replaceOp(splitOp, resValues);
  return success();
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {

    auto split = splitOp.split();
    auto builder = mlir::Builder(splitOp.getContext());

    llvm::Optional<ArrayAttr> optionalAttr;
    if (auto splitConstOp = getONNXConstantOp(split)) {
      // Checking value of split parameter.
      auto splitAttribute =
          createArrayAttrFromConstantOp(builder, splitConstOp);
      optionalAttr.emplace(splitAttribute);
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
    return ConstPropSplitPatternCommon(splitOp, rewriter, splitOp.split());
  }
};

// https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-13
/*
 * output = np.copy(data)
 * update_indices = indices.shape[:-1]
 * for idx in np.ndindex(update_indices):
 *     output[indices[idx]] = updates[idx]
 */
void ScatterNDImpl(DisposableElementsAttr dataElements,
    DisposableElementsAttr indicesElements,
    DisposableElementsAttr updatesElements, MutableArrayRef<WideNum> output) {
  dataElements.readWideNums(output);
  ArrayBuffer<int64_t> indicesBuffer = indicesElements.getArray<int64_t>();
  ArrayRef<int64_t> indices = indicesBuffer.get();
  ArrayBuffer<WideNum> updatesBuffer = updatesElements.getWideNums();
  ArrayRef<WideNum> updates = updatesBuffer.get();

  auto dataShape = dataElements.getShape();
  auto indicesShape = indicesElements.getShape();
  auto updatesShape = updatesElements.getShape();

  int64_t indices_nd = indicesShape.back();
  auto outer = indicesShape.drop_back();
  int64_t n_slices = ShapedType::getNumElements(outer);
  int64_t slice_size =
      ShapedType::getNumElements(updatesShape.drop_front(outer.size()));
  auto dataStrides = getStrides(dataShape);
  auto sliceStrides = llvm::makeArrayRef(dataStrides).take_front(indices_nd);

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

    if (!isFromDenseONNXConstantOp(scatterNdOp.data()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.indices()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.updates()))
      return failure();

    ConstPropCounters::count("Scatter",
        {scatterNdOp.data(), scatterNdOp.indices(), scatterNdOp.updates()});

    ElementsAttrBuilder elementsBuilder(rewriter.getContext());
    DisposableElementsAttr dataElements =
        getConstValueAsDisposableElements(elementsBuilder, scatterNdOp.data());
    DisposableElementsAttr indicesElements = getConstValueAsDisposableElements(
        elementsBuilder, scatterNdOp.indices());
    DisposableElementsAttr updatesElements = getConstValueAsDisposableElements(
        elementsBuilder, scatterNdOp.updates());
    ElementsAttr scatteredElements = elementsBuilder.fromWideNums(
        dataElements.getType(), [&](MutableArrayRef<WideNum> dst) {
          ScatterNDImpl(dataElements, indicesElements, updatesElements, dst);
        });
    ONNXConstantOp constOp = createReplacingConstantOp(
        rewriter, scatterNdOp.data(), scatteredElements);

    rewriter.replaceOp(scatterNdOp, constOp.getResult());
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
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr castElements =
      elementsBuilder.castElementType(constElements, replacingElemType);
  return createReplacingConstantOp(rewriter, replacingValue, castElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//===----------------------------------------------------------------------===//

void ConstPropSliceImpl(ShapedType outputType,
    const ONNXSliceOpShapeHelper &shapeHelper,
    DisposableElementsAttr inputElements, MutableArrayRef<WideNum> outputData) {
  size_t rank = outputType.getRank();
  auto outputShape = outputType.getShape();
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  std::vector<int64_t> inputStrides = getStrides(inputElements.getShape());
  size_t start = 0;
  SmallVector<size_t, 4> steps(rank, 0);
  for (size_t axis = 0; axis < rank; ++axis) {
    start += shapeHelper.starts[axis].getLiteral() * inputStrides[axis];
    steps[axis] = shapeHelper.steps[axis].getLiteral() * inputStrides[axis];
  }
  ArrayBuffer<WideNum> inputBuffer = inputElements.getWideNums();
  ArrayRef<WideNum> inputData = inputBuffer.get();
  auto traverse = [&](size_t axis, size_t srcPos, size_t dstPos,
                      const auto &recurse) -> void {
    if (axis == rank) {
      outputData[dstPos] = inputData[srcPos];
    } else {
      size_t srcStep = steps[axis];
      size_t dstStride = outputStrides[axis];
      size_t dimSize = outputShape[axis];
      for (size_t i = 0; i < dimSize; ++i) {
        recurse(axis + 1, srcPos, dstPos, recurse);
        srcPos += srcStep;
        dstPos += dstStride;
      }
    }
  };
  traverse(0, start, 0, traverse);
}

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXSliceOp sliceOp = cast<ONNXSliceOp>(op);

  // Get starts, ends, axes and steps via ShapeHelper.
  ONNXSliceOpShapeHelper shapeHelper(op, {});
  if (failed(shapeHelper.computeShape())) {
    sliceOp.emitError("Failed to scan " + ONNXSliceOp::getOperationName() +
                      " parameters successfully");
    return nullptr;
  }

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr inputElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
  ShapedType outputType = replacingValue.getType().cast<ShapedType>();
  ElementsAttr slicedElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropSliceImpl(outputType, shapeHelper, inputElements, dst);
      });
  return createReplacingConstantOp(rewriter, replacingValue, slicedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConcatOp.
//===----------------------------------------------------------------------===//

void ConstPropConcatImpl(ShapedType outputType,
    ArrayRef<DisposableElementsAttr> inputElements, int64_t axis,
    MutableArrayRef<WideNum> outputData) {
  ArrayRef<int64_t> outputShape = outputType.getShape();
  size_t stride = ShapedType::getNumElements(outputShape.drop_front(axis));
  size_t start = 0;
  auto out = outputData.begin();
  for (DisposableElementsAttr input : inputElements) {
    ArrayRef<int64_t> inputShape = input.getShape();
    size_t len = ShapedType::getNumElements(inputShape.drop_front(axis));
    ArrayBuffer<WideNum> inputData = input.getWideNums();
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

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  SmallVector<DisposableElementsAttr, 4> inputElements;
  inputElements.reserve(operands.size());
  for (Value input : operands)
    inputElements.push_back(
        getConstValueAsDisposableElements(elementsBuilder, input));
  ElementsAttr concatenatedElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropConcatImpl(outputType, inputElements, axis, dst);
      });
  return createReplacingConstantOp(
      rewriter, replacingValue, concatenatedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ExpandOp.
//===----------------------------------------------------------------------===//

Value ConstPropExpand(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ConstPropCounters::count("Expand", {constValue});
  ArrayRef<int64_t> expandedShape = getShape(replacingValue.getType());

  ElementsAttr constElements = getConstValueElements(constValue);
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr expandedElements =
      elementsBuilder.expand(constElements, expandedShape);
  return createReplacingConstantOp(rewriter, replacingValue, expandedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for GatherOp.
//===----------------------------------------------------------------------===//

void ConstPropGatherImpl(ShapedType outputType,
    DisposableElementsAttr inputElements,
    DisposableElementsAttr indicesElements, int64_t axis,
    MutableArrayRef<WideNum> outputData) {
  ArrayBuffer<WideNum> inputData = inputElements.getWideNums();
  ArrayBuffer<int64_t> indicesData = indicesElements.getArray<int64_t>();
  auto inputShape = inputElements.getShape();
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
  int64_t axis = gatherOp.axis();
  if (axis < 0)
    axis += inputValue.getType().cast<ShapedType>().getRank();

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr inputElements =
      getConstValueAsDisposableElements(elementsBuilder, inputValue);
  DisposableElementsAttr indicesElements =
      getConstValueAsDisposableElements(elementsBuilder, indicesValue);
  ShapedType outputType = replacingValue.getType().cast<ShapedType>();
  ElementsAttr gatheredElements = elementsBuilder.fromWideNums(
      outputType, [&](MutableArrayRef<WideNum> dst) {
        ConstPropGatherImpl(
            outputType, inputElements, indicesElements, axis, dst);
      });
  return createReplacingConstantOp(rewriter, replacingValue, gatheredElements)
      .getResult();
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
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements)
      .getResult();
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
