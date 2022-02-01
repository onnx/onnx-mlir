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

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

#include <math.h>

using namespace mlir;

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
    DenseElementsAttr dataAttr =
        op->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>();
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
bool isFromDenseONNXConstantOp(Value result) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<::mlir::Attribute>("value")))
    if (!(op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)))
      return false;
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
// Code to perform constant propagation for binary in presence of broadcast.
//===----------------------------------------------------------------------===//

// Template to generate binary operation results. It takes as input the element
// type as well as the two element attributes for the operation, and return the
// result of the operation.

template <typename OP, typename T>
struct ElementWiseBinaryOpImpl {
  static T impl(T lhs, T rhs) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXAddOp, T> {
  static T impl(T lhs, T rhs) { return (lhs + rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXSubOp, T> {
  static T impl(T lhs, T rhs) { return (lhs - rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXMulOp, T> {
  static T impl(T lhs, T rhs) { return (lhs * rhs); }
};

template <typename T>
struct ElementWiseBinaryOpImpl<ONNXDivOp, T> {
  static T impl(T lhs, T rhs) { return (lhs / rhs); }
};

template <typename OP, typename T>
T ComputeConstPropElementwiseBinary(T lhs, T rhs) {
  return ElementWiseBinaryOpImpl<OP, T>::impl(lhs, rhs);
}

template <typename ElementwiseBinaryOp, typename T>
void IterateConstPropElementwiseBinary(char *lhs, char *rhs,
    ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape, char *res,
    ArrayRef<int64_t> outputShape) {
  // Rank info.
  int lhsRank = lhsShape.size();
  int rhsRank = rhsShape.size();
  int outputRank = outputShape.size();
  // Strides info.
  std::vector<int64_t> outputStrides = getStrides(outputShape);
  std::vector<int64_t> lhsStrides = getStrides(lhsShape);
  std::vector<int64_t> rhsStrides = getStrides(rhsShape);
  // Data pointers.
  T *lhsArray = reinterpret_cast<T *>(lhs);
  T *rhsArray = reinterpret_cast<T *>(rhs);
  T *resArray = reinterpret_cast<T *>(res);

  // Check broadcasting.
  bool broadcasting = false;
  if (lhsRank != rhsRank)
    broadcasting = true;
  else
    for (int i = 0; i < outputRank; ++i)
      if (lhsShape[i] != rhsShape[i]) {
        broadcasting = true;
        break;
      }

  // Do computation.
  for (int64_t i = 0; i < getNumberOfElements(outputShape); ++i) {
    // Compute indices to access the output.
    std::vector<int64_t> outputIndices = getAccessIndex(i, outputStrides);

    // Compute indices to access inputs.
    SmallVector<int64_t, 4> lhsIndices(lhsRank, 0);
    SmallVector<int64_t, 4> rhsIndices(rhsRank, 0);
    if (!broadcasting) {
      for (int k = 0; k < outputRank; ++k) {
        lhsIndices[k] = outputIndices[k];
        rhsIndices[k] = outputIndices[k];
      }
    } else {
      for (int k = 0; k < outputRank; ++k) {
        // in the lhs index range.
        if (k >= outputRank - lhsRank) {
          int lhsIndex = k - outputRank + lhsRank;
          if (lhsShape[lhsIndex] == 1)
            // broadcast
            lhsIndices[lhsIndex] = 0;
          else
            lhsIndices[lhsIndex] = outputIndices[k];
        }
        // in the rhs index range.
        if (k >= outputRank - rhsRank) {
          int rhsIndex = k - outputRank + rhsRank;
          if (rhsShape[rhsIndex] == 1)
            // broadcast
            rhsIndices[rhsIndex] = 0;
          else
            rhsIndices[rhsIndex] = outputIndices[k];
        }
      }
    }

    // Calculate element-wise binary result.
    int64_t lhsOffset = getLinearAccessIndex(lhsIndices, lhsStrides);
    int64_t rhsOffset = getLinearAccessIndex(rhsIndices, rhsStrides);

    T lhsValue = *(lhsArray + lhsOffset);
    T rhsValue = *(rhsArray + rhsOffset);
    *(resArray + i) = ComputeConstPropElementwiseBinary<ElementwiseBinaryOp, T>(
        lhsValue, rhsValue);
  }
}

/// Do element-wise binary calculation of 'lhs' and 'rhs' values and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseBinaryOp>
ONNXConstantOp ConstPropElementwiseBinary(
    PatternRewriter &rewriter, Value replacingValue, Value lhs, Value rhs) {
  Type elementType =
      replacingValue.getType().cast<ShapedType>().getElementType();
  ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> outputShape =
      replacingValue.getType().cast<ShapedType>().getShape();

  // Get lhs and rhs values.
  char *lhsArray = getArrayFromAttributeOrBuffer(rewriter, lhs.getDefiningOp());
  char *rhsArray = getArrayFromAttributeOrBuffer(rewriter, rhs.getDefiningOp());

  // Do calculation.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    IterateConstPropElementwiseBinary<ElementwiseBinaryOp, double>(
        lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int64_t>(
        lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res;
}

//===----------------------------------------------------------------------===//
//// Code to perform constant propagation for unary operation.
//===----------------------------------------------------------------------===//

template <typename OP, typename T>
struct ElementWiseUnaryOpImpl {
  static T impl(T val) { llvm_unreachable("unknown operation"); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXNegOp, T> {
  static T impl(T val) { return (-val); }
};

template <typename T>
struct ElementWiseUnaryOpImpl<ONNXSqrtOp, T> {
  static T impl(T val) { return sqrt(val); }
};
template <typename OP, typename T>
T ComputeConstPropElementwiseUnary(T val) {
  return ElementWiseUnaryOpImpl<OP, T>::impl(val);
}

template <typename ElementwiseUnaryOp, typename T>
void IterateConstPropElementwiseUnary(
    char *input, char *res, ArrayRef<int64_t> outputShape) {
  // Data pointers.
  T *inputArray = reinterpret_cast<T *>(input);
  T *resArray = reinterpret_cast<T *>(res);

  // Calculate element-wise unary result.
  for (int64_t i = 0; i < getNumberOfElements(outputShape); ++i) {
    *(resArray + i) = ComputeConstPropElementwiseUnary<ElementwiseUnaryOp, T>(
        *(inputArray + i));
  }
}

/// Do element-wise unary calculation of 'input' value and create an
/// ONNXConstantOp for the result.
template <typename ElementwiseUnaryOp>
ONNXConstantOp ConstPropElementwiseUnary(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ShapedType replacingType = replacingValue.getType().cast<ShapedType>();
  ArrayRef<int64_t> replacingShape = replacingType.getShape();
  Type elementType = replacingType.getElementType();

  // Get the const value.
  char *constArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());

  // Do calculation.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    IterateConstPropElementwiseUnary<ElementwiseUnaryOp, double>(
        constArray, resArray, replacingShape);
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    IterateConstPropElementwiseUnary<ElementwiseUnaryOp, int64_t>(
        constArray, resArray, replacingShape);
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for transpose.
//===----------------------------------------------------------------------===//

ONNXConstantOp ConstPropTranspose(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ArrayRef<int64_t> replacingShape =
      replacingValue.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> constShape =
      constValue.getType().cast<ShapedType>().getShape();
  Type elementType =
      replacingValue.getType().cast<ShapedType>().getElementType();

  // Get perm attribute.
  SmallVector<uint64_t, 4> perm;
  Attribute permAttr =
      replacingValue.getDefiningOp()->getAttrOfType<::mlir::Attribute>("perm");
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.cast<ArrayAttr>().getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

  // Get the const value.
  char *constArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());

  // Do calculation.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray =
      allocateBufferFor(replacingValue.getType(), /*useMaxSize=*/true);
  ConstPropTransposeImpl(
      elementType, constArray, constShape, perm, replacingShape, resArray);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for unsqueeze.
//===----------------------------------------------------------------------===//

ONNXConstantOp ConstPropUnsqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Operation *inputOp = input.getDefiningOp();

  char *resArray = getArrayFromAttributeOrBuffer(rewriter, inputOp);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for Squeeze.
//===----------------------------------------------------------------------===//

ONNXConstantOp ConstPropSqueeze(
    PatternRewriter &rewriter, Value replacingValue, Value input) {
  Operation *inputOp = input.getDefiningOp();

  char *resArray = getArrayFromAttributeOrBuffer(rewriter, inputOp);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      createConstantOpAndStoreBufferPtr(rewriter, replacingValue, resArray);

  return res;
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
  Type elementType = inputType.getElementType();

  // Split axis.
  uint64_t splitAxis = splitOp.axis();
  // Compute split offsets.
  SmallVector<int64_t, 4> splitOffsets;
  {
    if (!splitAttr.hasValue())
      // If split attribute is not specified, split size is equally divided.
      assert(inputShape[splitAxis] % numOfResults == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
    int64_t offset = 0;
    for (unsigned int i = 0; i < numOfResults; ++i) {
      splitOffsets.emplace_back(offset);
      if (splitAttr.hasValue())
        offset += splitAttr.getValue()[i].cast<IntegerAttr>().getInt();
      else
        offset += inputShape[splitAxis] / numOfResults;
    }
  }

  // Get the constant input value.
  char *inputArray =
      getArrayFromAttributeOrBuffer(rewriter, input.getDefiningOp());

  SmallVector<Value, 4> replacingValues;
  SmallVector<Type, 4> replacingTypes;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    replacingValues.emplace_back(splitOp.getResults()[i]);
    replacingTypes.emplace_back(splitOp.getResults()[i].getType());
  }

  // Do splitting.
  std::vector<char *> resBuffers;
  ConstPropSplitImpl(elementType, inputArray, inputShape, splitAxis,
      splitOffsets, replacingTypes, resBuffers);

  // Construct result values.
  std::vector<Value> resValues;
  for (unsigned int i = 0; i < numOfResults; ++i) {
    ONNXConstantOp res = createConstantOpAndStoreBufferPtr(
        rewriter, replacingValues[i], resBuffers[i]);
    resValues.emplace_back(res.getResult());
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
template <typename T>
LogicalResult ScatterNDImpl(
    PatternRewriter &rewriter, ONNXScatterNDOp scatterNdOp, char *raw_buffer) {

  char *data_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.data().getDefiningOp());
  char *indices_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.indices().getDefiningOp());
  char *updates_value = getArrayFromAttributeOrBuffer(
      rewriter, scatterNdOp.updates().getDefiningOp());

  auto data_shape = scatterNdOp.data().getType().cast<ShapedType>().getShape();
  auto indices_shape =
      scatterNdOp.indices().getType().cast<ShapedType>().getShape();
  auto updates_shape =
      scatterNdOp.updates().getType().cast<ShapedType>().getShape();

  // the output shape keep same with data, so fill with input data temporarily
  T *output_data = reinterpret_cast<T *>(data_value);
  int64_t *indices_data = reinterpret_cast<int64_t *>(indices_value);
  T *updates_data = reinterpret_cast<T *>(updates_value);

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

  int64_t output_flat_size = getNumberOfElements(data_shape);
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
      output_data[to_pos + j] = updates_data[i * slice_size + j];
    }
  }

  std::memcpy(raw_buffer, data_value, output_flat_size * 8);
  return success();
}

class ConstPropScatterNDPattern : public OpRewritePattern<ONNXScatterNDOp> {
public:
  using OpRewritePattern<ONNXScatterNDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXScatterNDOp scatterNdOp, PatternRewriter &rewriter) const override {
    // Match
    if (!scatterNdOp.getResult()
             .getType()
             .template dyn_cast_or_null<RankedTensorType>())
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.data()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.indices()))
      return failure();

    if (!isFromDenseONNXConstantOp(scatterNdOp.updates()))
      return failure();

    char *result_raw_data =
        allocateBufferFor(scatterNdOp.data().getType(), /*useMaxSize=*/true);

    mlir::ShapedType shaped_type =
        scatterNdOp.data().getType().cast<ShapedType>();

    if (shaped_type.getElementType().isa<FloatType>()) {
      if (mlir::failed(
              ScatterNDImpl<double>(rewriter, scatterNdOp, result_raw_data)))
        return failure();
    } else if (shaped_type.getElementType().isa<IntegerType>()) {
      if (mlir::failed(
              ScatterNDImpl<int64_t>(rewriter, scatterNdOp, result_raw_data)))
        return failure();
    } else {
      llvm_unreachable("type not yet supported");
    }

    // Construct result values.
    ONNXConstantOp gen_const_op = createConstantOpAndStoreBufferPtr(
        rewriter, scatterNdOp.data(), result_raw_data);

    SmallVector<Value, 1> op_repl_values(1, gen_const_op.getResult());
    rewriter.replaceOp(scatterNdOp, op_repl_values);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern definition.
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/ONNXConstProp.inc"

//===----------------------------------------------------------------------===//
// Code to manage the pass.
//===----------------------------------------------------------------------===//

struct ConstPropONNXToONNXPass
    : public PassWrapper<ConstPropONNXToONNXPass, OperationPass<FuncOp>> {

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

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

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
      char *arr = allocateBufferFor(constOp.getResult().getType());
      getArrayForFinalOutput(op, arr);
      ShapedType type = constOp.getResult().getType().cast<ShapedType>();
      DenseElementsAttr denseAttr = createDenseElementsAttrFromArray(arr, type);
      op->setAttr("value", denseAttr);
      op->removeAttr(BUFFER_ID_ATTR);
      free(arr);
    }
  });

  // Remove temporary buffers.
  for (char *ptr : bufferPtrs) {
    free(ptr);
  }
  bufferPtrs.clear();

} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
