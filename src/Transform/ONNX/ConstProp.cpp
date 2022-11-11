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
// that there is no knowledge about tensor shape at this point
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

const StringRef BUFFER_ID_ATTR = "buffer_id";

/// Buffers will be allocated to store intermediate constants during the const
/// propagation. The use of buffers is to avoid creating dense attributes which
/// are immortal by design in MLIR, leading to small memory footprint.
///
/// There are three helper functions to use when working with buffers:
/// 1) getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op)
///    - create a buffer from a dense attribute at the first time we reach the
///      const 'op' and add the buffer to the buffer pool, or
///    - get the buffer from the buffer pool if it was created.
/// 2) createConstantOpAndStoreBufferPtr(..., char *buffer)
///    - create a new ONNXConstantOp using the given buffer, and
///    - add the buffer to the buffer pool.
/// 3) allocateBufferFor(Value value, bool useMaxSize = false)
///    - create a new buffer whose size is obtained from the type of 'value'.
///
/// Note that:
///   - The buffers in the buffer pool will be automatically freed. Users don't
///     need to take care about that.
///   - If we create a buffer and do not put it on the buffer pool, please
///     make sure that it is correctly freed.
///
/// Buffer pool to store buffer pointers.
SmallVector<char *, 4> bufferPtrs;

/// A helper function to get a value of a given type from an attribute.
template <typename T>
T getAttrValue(Attribute attr) {
  llvm_unreachable("unknown operation");
}

template <>
ATTRIBUTE(unused)
double getAttrValue(Attribute attr) {
  return attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
ATTRIBUTE(unused)
float getAttrValue(Attribute attr) {
  return (float)attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
ATTRIBUTE(unused)
int64_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

template <>
ATTRIBUTE(unused)
int32_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

/// Get a data array from a given ONNXConstantOp. If data were stored in memory,
/// get from memory. Otherwise, get from the dense attribute.
char *getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");
  char *res = nullptr;

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    res = bufferPtrs[bufferId];
  } else {
    ElementsAttr dataAttr = constOp.valueAttr().cast<mlir::ElementsAttr>();
    res = createArrayFromDenseElementsAttr(dataAttr);
    bufferPtrs.emplace_back(res);
    unsigned bufferId = bufferPtrs.size() - 1;
    // Add an attribute to store the buffer id.
    op->setAttr(BUFFER_ID_ATTR,
        IntegerAttr::get(
            rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false),
            bufferId));
  }
  return res;
}

/// Get array with the exact data type for the final ONNXConstantOp.
void getArrayForFinalOutput(Operation *op, char *res) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    char *resArr = bufferPtrs[bufferId];
    convertDoubleInt64ToExactType(constOp.getResult().getType(), resArr, res);
  } else {
    llvm_unreachable("Could not find the input buffer");
  }
}

/// A helper function to construct a RankedTensorType from a ShapedType.
ATTRIBUTE(unused) RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// A helper function to check whether a value is produced by a dense
/// ONNXConstantOp.
bool isFromDenseONNXConstantOp(Value result, bool trueONNXConstant = false) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<::mlir::Attribute>("value"))) {
    if (trueONNXConstant)
      return false;
    if (!(op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)))
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

/// A helper function to create an ONNXConstantOp for a given data array.
/// This ONNXConstantOp is only used internally.
ONNXConstantOp createConstantOpAndStoreBufferPtr(
    PatternRewriter &rewriter, Value replacingValue, char *vt) {
  Location loc = replacingValue.getLoc();
  // int64_t maxSizeInBytes = getMaxSizeInBytes(replacingValue.getType());

  ONNXConstantOp constOp = rewriter.create<ONNXConstantOp>(loc,
      replacingValue.getType(), Attribute(), Attribute(), FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());

  // Store the buffer pointer.
  unsigned bufferId = (unsigned)-1;
  for (unsigned i = 0; i < bufferPtrs.size(); ++i) {
    if (bufferPtrs[i] == vt) {
      bufferId = i;
      break;
    }
  }

  if (bufferId == (unsigned)-1) {
    bufferPtrs.emplace_back(vt);
    bufferId = bufferPtrs.size() - 1;
  }
  // Store the buffer id.
  constOp.getOperation()->setAttr(BUFFER_ID_ATTR,
      IntegerAttr::get(
          rewriter.getIntegerType(/*width=*/64, /*isSigned=*/false), bufferId));

  return constOp;
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

  // -------------------------------------------------------------- //
  // Adapter code to make DisposableElements logic interoperate with the
  // "buffer_id" buffer pool logic. TODO: Remove after migration.
  Attribute bufferIDAttr =
      constOp->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    ShapedType type = constValue.getType().cast<ShapedType>();
    int64_t maxSize = getMaxSizeInBytes(type);
    DType dtype = dtypeOfMlirType(type.getElementType());
    DType bufferDType = wideDTypeOfDType(dtype);
    ArrayRef<char> buffer(bufferPtrs[bufferId], maxSize);
    return elementsBuilder.fromRawBytes(
        type, bufferDType, buffer, /*mustCopy=*/true);
  }
  // -------------------------------------------------------------- //

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
  DType bufferDType = wideDTypeOfDType(inputElements.getDType());
  std::vector<Value> resValues;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    Value replacingValue = splitOp.getResults()[i];
    ElementsAttr splitElements = elementsBuilder.fromRawBytes(
        replacingValue.getType(), bufferDType, [&](MutableArrayRef<char> dst) {
          MutableArrayRef<WideNum> dstNums = castMutableArrayRef<WideNum>(dst);
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
  dataElements.readElements(output_data);
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
    DType bufferDType = wideDTypeOfDType(dataElements.getDType());
    ElementsAttr scatteredElements = elementsBuilder.fromRawBytes(
        dataElements.getType(), bufferDType, [&](MutableArrayRef<char> dst) {
          ScatterNDImpl(dataElements, indicesElements, updatesElements,
              castMutableArrayRef<WideNum>(dst));
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
  DType bufferDType = wideDTypeOfDType(inputElements.getDType());
  ElementsAttr slicedElements = elementsBuilder.fromRawBytes(
      outputType, bufferDType, [&](MutableArrayRef<char> dst) {
        ConstPropSliceImpl(outputType, shapeHelper, inputElements,
            castMutableArrayRef<WideNum>(dst));
      });
  return createReplacingConstantOp(rewriter, replacingValue, slicedElements)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for ConcatOp.
//===----------------------------------------------------------------------===//

Value ConstPropConcat(PatternRewriter &rewriter, Value replacingValue,
    ValueRange operands, IntegerAttr axisAttr) {
  // Get the const values using the maximum precision e.g. double, int64_t.
  SmallVector<char *, 4> inputArrays;
  for (uint64_t i = 0; i < operands.size(); ++i) {
    char *array =
        getArrayFromAttributeOrBuffer(rewriter, operands[i].getDefiningOp());
    inputArrays.emplace_back(array);
  }
  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  int64_t axis = axisAttr.getValue().getSExtValue();
  if (axis < 0)
    axis += outputShape.size();

  // If concatenation is on the outermost dimension, do memcpy for better
  // performance. Otherwise, copy elements one-by-one.
  if (axis == 0) {
    int64_t offset = 0;
    for (uint64_t i = 0; i < operands.size(); ++i) {
      int64_t sizeInBytes = getMaxSizeInBytes(operands[i].getType());
      memcpy(resArray + offset, inputArrays[i], sizeInBytes);
      offset += sizeInBytes;
    }
  } else {
    int64_t dimAtAxis = 0;
    for (uint64_t i = 0; i < operands.size(); ++i) {
      ArrayRef<int64_t> inputShape = getShape(operands[i].getType());
      std::vector<int64_t> inputStrides = getStrides(inputShape);
      for (int64_t k = 0; k < ShapedType::getNumElements(inputShape); ++k) {
        std::vector<int64_t> inputIndices = getAccessIndex(k, inputStrides);
        std::vector<int64_t> outputIndices(inputIndices);
        outputIndices[axis] += dimAtAxis;
        int64_t outputOffset =
            getLinearAccessIndex(outputIndices, outputStrides);
        int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
        memcpy(resArray + outputOffset * typeSize,
            inputArrays[i] + k * typeSize, typeSize);
      }
      dimAtAxis += inputShape[axis];
    }
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
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

Value ConstPropGather(PatternRewriter &rewriter, Value replacingValue,
    Value inputValue, Value indicesValue) {
  Operation *op = replacingValue.getDefiningOp();
  ONNXGatherOp gatherOp = cast<ONNXGatherOp>(op);

  ArrayRef<int64_t> inputShape = getShape(inputValue.getType());
  ArrayRef<int64_t> indicesShape = getShape(indicesValue.getType());
  ArrayRef<int64_t> outputShape = getShape(replacingValue.getType());
  std::vector<int64_t> inputStrides = getStrides(inputShape);
  std::vector<int64_t> indicesStrides = getStrides(indicesShape);
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  int64_t inputRank = inputShape.size();
  int64_t indicesRank = indicesShape.size();

  int64_t axis = gatherOp.axis();
  if (axis < 0)
    axis += inputRank;
  int64_t axisDim = inputShape[axis];

  // Get the input value using the maximum precision e.g. double, int64_t.
  char *inputArray =
      getArrayFromAttributeOrBuffer(rewriter, inputValue.getDefiningOp());

  // Get the indices value using the maximum precision. Index is integer.
  int64_t *indicesArray = (int64_t *)getArrayFromAttributeOrBuffer(
      rewriter, indicesValue.getDefiningOp());

  // Create the result buffer using the maximum precision e.g. double, int64_t.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);

  // Iterate over the output index space.
  for (int64_t ii = 0; ii < ShapedType::getNumElements(outputShape); ++ii) {
    std::vector<int64_t> outputIndices = getAccessIndex(ii, outputStrides);
    SmallVector<int64_t, 4> inputIndices, indicesIndices;
    // Compute tensor access indices for indices: indices[jj].
    for (int j = 0; j < indicesRank; ++j)
      indicesIndices.emplace_back(outputIndices[axis + j]);
    int64_t indicesOffset =
        getLinearAccessIndex(indicesIndices, indicesStrides);
    // Get indices.
    int64_t axisIndex = *(indicesArray + indicesOffset);
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
    int64_t typeSize = 8; // both double and int64_t have size of 8 bytes.
    memcpy(resArray + ii * typeSize, inputArray + inputOffset * typeSize,
        typeSize);
  }

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res.getResult();
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

  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    Operation *op = constOp.getOperation();
    if (op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)) {
      ShapedType type = constOp.getResult().getType().cast<ShapedType>();
      char *arr = allocateBufferFor(type, /*useMaxSize=*/false);
      getArrayForFinalOutput(op, arr);
      DenseElementsAttr denseAttr =
          createDenseElementsAttrFromRawBuffer(type, arr);
      op->setAttr("value", denseAttr);
      op->removeAttr(BUFFER_ID_ATTR);
      free(arr);
    } else if (auto elements =
                   constOp.valueAttr().dyn_cast<DisposableElementsAttr>()) {
      constOp.valueAttr(toDenseElementsAttr(elements));
    }
  });

  // Remove temporary buffers.
  for (char *ptr : bufferPtrs) {
    free(ptr);
  }
  bufferPtrs.clear();

  // TODO: determine if we should call DisposablePool::garbageCollectUnreachable
  //       (what's the relationship between function and the ModuleOp?)
}

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
