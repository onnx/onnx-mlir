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

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

#include <math.h>

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
//
// TODO: Migrate all constant propagation to DisposableElementsAttr and remove
//       all "buffer_id" buffer pool helpers above and in ConstPropHelper.
//===----------------------------------------------------------------------===//

DisposableElementsAttr getConstValueAsDisposableElements(
    ElementsAttrBuilder &elementsBuilder, Value constValue) {
  ONNXConstantOp constOp = getONNXConstantOp(constValue);
  return elementsBuilder.fromElementsAttr(
      constOp.valueAttr().cast<ElementsAttr>());
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
  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr constElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
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
  static T impl(T lhs, T rhs) { llvm_unreachable("unknown operation"); }
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

template <typename ElementwiseBinaryOp>
auto combinerOfElementwiseBinaryOp(DType operandsDType) {
  using Combiner = std::function<WideNum(WideNum, WideNum)>;
  return dispatchByDType(operandsDType, [](auto dtype) -> Combiner {
    using W = WideDType<dtype>;
    using OpImpl =
        ElementWiseBinaryOpImpl<ElementwiseBinaryOp, typename W::type>;
    return [](WideNum lhs, WideNum rhs) -> WideNum {
      return W::pack(OpImpl::impl(W::unpack(lhs), W::unpack(rhs)));
    };
  });
}

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  Type replacingType = replacingValue.getType().cast<ShapedType>();

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr lhs =
      getConstValueAsDisposableElements(elementsBuilder, lhsValue);
  DisposableElementsAttr rhs =
      getConstValueAsDisposableElements(elementsBuilder, rhsValue);
  DType operandsDType = lhs.getDType();
  assert(operandsDType == rhs.getDType());
  ElementsAttr resultElements = elementsBuilder.combine(lhs, rhs, replacingType,
      combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsDType));
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
  return dispatchByMlirType(
      elemType, [](auto dtype) -> ElementsAttrBuilder::Transformer {
        using W = WideDType<dtype>;
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
  Type replacingElemType =
      replacingValue.getType().cast<ShapedType>().getElementType();

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr constElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
  assert(replacingElemType == constElements.getElementType() &&
         "all element wise unary ops preserve element type");
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
  // TODO: figure out if default may be omitted and what to do in that case
  ArrayAttr permAttr =
      replacingValue.getDefiningOp()->getAttr("perm").cast<ArrayAttr>();
  SmallVector<uint64_t, 4> perm =
      createIntVectorFromArrayAttr<uint64_t>(permAttr);

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr constElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
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
  ArrayRef<int64_t> reshapedShape = getShape(replacingValue.getType());
  ElementsAttr reshapedElements =
      ConstPropReshapeImpl(rewriter, replacingValue, input, reshapedShape);
  return createReplacingConstantOp(rewriter, replacingValue, reshapedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult ConstPropSplitPatternCommon(Op splitOp, PatternRewriter &rewriter,
    llvm::Optional<ArrayAttr> splitAttr) {
  // Basic info.
  unsigned numOfResults = splitOp.getNumResults();
  Value input = splitOp.input();
  if (!isFromDenseONNXConstantOp(input))
    return failure();
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Split axis.
  uint64_t splitAxis = splitOp.axis();
  // Compute split offsets.
  SmallVector<int64_t, 4> splitOffsets;
  {
    if (!splitAttr.has_value())
      // If split attribute is not specified, split size is equally divided.
      assert(inputShape[splitAxis] % numOfResults == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
    int64_t offset = 0;
    for (unsigned int i = 0; i < numOfResults; ++i) {
      splitOffsets.emplace_back(offset);
      if (splitAttr.has_value())
        offset += splitAttr.value()[i].cast<IntegerAttr>().getInt();
      else
        offset += inputShape[splitAxis] / numOfResults;
    }
  }

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr inputElements =
      getConstValueAsDisposableElements(elementsBuilder, input);
  ArrayBuffer<WideNum> inputNums = inputElements.getWideNums();
  size_t numElements = inputElements.getNumElements();
  auto strides = getDefaultStrides(inputShape);
  size_t iterationOffset =
      splitAxis == 0 ? numElements : strides[splitAxis - 1];
  size_t iterations = numElements / iterationOffset;
  std::vector<Value> resValues;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    Value replacingValue = splitOp.getResults()[i];
    ElementsAttr splitElements = elementsBuilder.fromWideNums(
        replacingValue.getType(), [&](MutableArrayRef<WideNum> dstNums) {
          auto dstIterator = dstNums.begin();
          size_t start = splitOffsets[i];
          size_t stop = i == numOfResults - 1 ? inputShape[splitAxis]
                                              : splitOffsets[i + 1];
          auto splitBegin =
              inputNums.get().begin() + (strides[splitAxis] * start);
          size_t splitSize = strides[splitAxis] * (stop - start);
          for (size_t j = 0; j < iterations; ++j) {
            dstIterator = std::copy_n(splitBegin, splitSize, dstIterator);
            splitBegin += iterationOffset;
          }
          assert(dstIterator == dstNums.end() &&
                 "result type num elements mismatch");
        });
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
    DisposableElementsAttr updatesElements,
    MutableArrayRef<WideNum> output_data) {
  dataElements.readWideNums(output_data);
  ArrayBuffer<char> indices_bytes = indicesElements.getRawBytes();
  assert(indicesElements.getDType() == DType::INT64);
  ArrayRef<int64_t> indices_data = castArrayRef<int64_t>(indices_bytes.get());
  ArrayBuffer<WideNum> updates_data = updatesElements.getWideNums();

  auto data_shape = dataElements.getShape();
  auto indices_shape = indicesElements.getShape();
  auto updates_shape = updatesElements.getShape();

  int64_t n_slices = 1;
  int64_t slice_size = 1;

  int64_t outer_dims = indices_shape.size() - 1;
  int64_t indices_nd = indices_shape[outer_dims];
  int64_t updates_dims = updates_shape.size();

  for (int64_t i = 0; i < outer_dims; i++) {
    n_slices *= indices_shape[i];
  }

  for (int64_t i = outer_dims; i < updates_dims; i++) {
    slice_size *= updates_shape[i];
  }

  int64_t output_flat_size = ShapedType::getNumElements(data_shape);
  int64_t remain_flat_size = output_flat_size;
  std::vector<int64_t> dims_to_count(indices_nd, 0);

  for (int64_t i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / data_shape[i];
    remain_flat_size = dims_to_count[i];
  }

  for (int64_t i = 0; i < n_slices; ++i) {
    int64_t to_pos = 0;
    for (int64_t j = 0; j < indices_nd; ++j) {
      int64_t idx = indices_data[i * indices_nd + j];
      // assert(0 <= idx && idx < data_shape[j]);
      to_pos += idx * dims_to_count[j];
    }
    for (int64_t j = 0; j < slice_size; j++) {
      output_data[to_pos + j] = updates_data.get()[i * slice_size + j];
    }
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
  Type replacingElemType =
      replacingValue.getType().cast<ShapedType>().getElementType();

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr constElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
  ElementsAttr castElements =
      elementsBuilder.castElementType(constElements, replacingElemType);
  return createReplacingConstantOp(rewriter, replacingValue, castElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for SliceOp.
//===----------------------------------------------------------------------===//

void ConstPropSliceImpl(ShapedType outputType,
    const NewONNXSliceOpShapeHelper &shapeHelper,
    DisposableElementsAttr inputElements, MutableArrayRef<WideNum> outputData) {
  std::vector<int64_t> outputStrides = getStrides(outputType.getShape());
  std::vector<int64_t> inputStrides = getStrides(inputElements.getShape());
  ArrayBuffer<WideNum> inputData = inputElements.getWideNums();
  // Iterate over the output index space.
  for (size_t i = 0; i < outputData.size(); ++i) {
    // Input index: "ii * step + start" for all dim.
    // Output index: "ii" for all dims.
    // where `ii` is a tensor index.
    std::vector<int64_t> outputIndices = getAccessIndex(i, outputStrides);
    SmallVector<int64_t, 4> inputIndices;
    for (unsigned k = 0; k < outputIndices.size(); ++k) {
      int64_t ii = outputIndices[k];
      inputIndices.emplace_back(ii * shapeHelper.steps[k].getLiteral() +
                                shapeHelper.starts[k].getLiteral());
    }
    int64_t inputOffset = getLinearAccessIndex(inputIndices, inputStrides);
    outputData[i] = inputData.get()[inputOffset];
  }
}

Value ConstPropSlice(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXSliceOp sliceOp = cast<ONNXSliceOp>(op);

  // Get starts, ends, axes and steps via ShapeHelper.
  NewONNXSliceOpShapeHelper shapeHelper(op, {});
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
  std::vector<int64_t> outputStrides = getStrides(outputShape);

  // If concatenation is on the outermost dimension, do memcpy for better
  // performance. Otherwise, copy elements one-by-one.
  if (axis == 0) {
    auto outputIterator = outputData.begin();
    for (uint64_t i = 0; i < inputElements.size(); ++i) {
      ArrayBuffer<WideNum> inputData = inputElements[i].getWideNums();
      outputIterator = std::copy(
          inputData.get().begin(), inputData.get().end(), outputIterator);
    }
  } else {
    int64_t dimAtAxis = 0;
    for (uint64_t i = 0; i < inputElements.size(); ++i) {
      ArrayBuffer<WideNum> inputData = inputElements[i].getWideNums();
      ArrayRef<int64_t> inputShape = inputElements[i].getShape();
      std::vector<int64_t> inputStrides = getStrides(inputShape);
      for (size_t k = 0; k < inputData.get().size(); ++k) {
        std::vector<int64_t> inputIndices = getAccessIndex(k, inputStrides);
        std::vector<int64_t> outputIndices(inputIndices);
        outputIndices[axis] += dimAtAxis;
        int64_t outputOffset =
            getLinearAccessIndex(outputIndices, outputStrides);
        outputData[outputOffset] = inputData.get()[k];
      }
      dimAtAxis += inputShape[axis];
    }
  }
}

Value ConstPropConcat(PatternRewriter &rewriter, Value replacingValue,
    ValueRange operands, IntegerAttr axisAttr) {
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
  ArrayRef<int64_t> expandedShape = getShape(replacingValue.getType());

  ElementsAttrBuilder elementsBuilder(rewriter.getContext());
  DisposableElementsAttr constElements =
      getConstValueAsDisposableElements(elementsBuilder, constValue);
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
  std::vector<int64_t> outputStrides = getStrides(outputType.getShape());
  ArrayRef<int64_t> inputShape = inputElements.getShape();
  ArrayRef<int64_t> indicesShape = indicesElements.getShape();
  std::vector<int64_t> inputStrides = getStrides(inputShape);
  std::vector<int64_t> indicesStrides = getStrides(indicesShape);
  int64_t inputRank = inputShape.size();
  int64_t indicesRank = indicesShape.size();
  int64_t axisDim = inputShape[axis];

  ArrayBuffer<WideNum> inputData = inputElements.getWideNums();
  ArrayBuffer<char> indicesBytes = indicesElements.getRawBytes();
  assert(indicesElements.getDType() == DType::INT64);
  ArrayRef<int64_t> indicesData = castArrayRef<int64_t>(indicesBytes.get());

  // Iterate over the output index space.
  for (size_t ii = 0; ii < outputData.size(); ++ii) {
    std::vector<int64_t> outputIndices = getAccessIndex(ii, outputStrides);
    SmallVector<int64_t, 4> inputIndices, indicesIndices;
    // Compute tensor access indices for indices: indices[jj].
    for (int j = 0; j < indicesRank; ++j)
      indicesIndices.emplace_back(outputIndices[axis + j]);
    int64_t indicesOffset =
        getLinearAccessIndex(indicesIndices, indicesStrides);
    // Get indices.
    int64_t axisIndex = indicesData[indicesOffset];
    if (axisIndex < 0)
      axisIndex += axisDim;

    // Compute tensor access indices for input: input[ii + (indices[jj],) + kk]
    // First add indices ii
    for (int i = 0; i < axis; ++i)
      inputIndices.emplace_back(outputIndices[i]);
    // Then add indices[jj] at axis.
    inputIndices.emplace_back(axisIndex);
    // Then add kk.
    for (int k = axis + 1; k < inputRank; ++k)
      inputIndices.emplace_back(outputIndices[indicesRank - 1 + k]);

    // Copy values.
    int64_t inputOffset = getLinearAccessIndex(inputIndices, inputStrides);
    outputData[ii] = inputData.get()[inputOffset];
  }
}

Value ConstPropGather(PatternRewriter &rewriter, Value replacingValue,
    Value inputValue, Value indicesValue) {
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

  void runOnOperation() final;
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

#ifdef SCRUB_DISPOSABLE_ATTR_AFTER_CONST_PROP
  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    if (auto disposable =
            constOp.valueAttr().dyn_cast<DisposableElementsAttr>()) {
      constOp.valueAttr(toDenseElementsAttr(disposable));
    }
  });

  // TODO: determine if we should call DisposablePool::garbageCollectUnreachable
  //       (what's the relationship between function and the ModuleOp?)
#endif
}

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
