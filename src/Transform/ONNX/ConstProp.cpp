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
#include "src/Pass/Passes.hpp"

#include <math.h>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Instructions to add a constant operation.
//===----------------------------------------------------------------------===//
// There is currently support for adding constant propagation for unary and
// binary athythmetic ops (binary ops support broadcast). To add an operation,
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
double getAttrValue(Attribute attr) {
  return attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
float getAttrValue(Attribute attr) {
  return (float)attr.cast<FloatAttr>().getValueAsDouble();
}

template <>
int64_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

template <>
int32_t getAttrValue(Attribute attr) {
  return attr.cast<IntegerAttr>().getInt();
}

/// Get the element size in bytes. Use the biggest size to avoid loss in
/// casting.
int64_t getEltSizeInBytes(Type ty) {
  auto elementType = ty.cast<ShapedType>().getElementType();

  int64_t sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the number of elements.
int64_t getNumberOfElements(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(Type ty) {
  ShapedType shapedType = ty.dyn_cast<ShapedType>();
  auto shape = shapedType.getShape();
  return getNumberOfElements(shape) * getEltSizeInBytes(shapedType);
}

/// Get the size of a tensor from its ranked type in bytes, using the largest
/// precision.
int64_t getMaxSizeInBytes(Type ty) {
  auto shape = ty.dyn_cast<ShapedType>().getShape();
  return getNumberOfElements(shape) * 8;
}

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
  for (int i = 0; i < strides.size(); ++i)
    index += indices[i] * strides[i];
  return index;
}

// Compute the tensor access index from a linear index.
std::vector<int64_t> getAccessIndex(
    int64_t linearIndex, ArrayRef<int64_t> strides) {
  std::vector<int64_t> res;
  for (int i = 0; i < strides.size(); ++i) {
    int64_t s = strides[i];
    if (linearIndex < s) {
      res.emplace_back(0);
    } else {
      res.emplace_back(floor(linearIndex / s));
      linearIndex = linearIndex % s;
    }
  }
  return res;
}

/// Allocate a buffer whose size is getting from a given Value's type.
char *allocateBufferFor(Value value, bool useMaxSize = false) {
  int64_t sizeInBytes;
  if (useMaxSize)
    sizeInBytes = getMaxSizeInBytes(value.getType().cast<ShapedType>());
  else
    sizeInBytes = getSizeInBytes(value.getType().cast<ShapedType>());
  char *res = (char *)malloc(sizeInBytes);
  return res;
}

/// Get a data array from a given ONNXConstantOp. If data were stored in memory,
/// get from memory. Otherwise, get from the dense attribute.
char *getArrayFromAttributeOrBuffer(PatternRewriter &rewriter, Operation *op) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");
  char *res;

  ShapedType shapedType = constOp.getResult().getType().cast<ShapedType>();
  int64_t maxSizeInBytes = getMaxSizeInBytes(shapedType);
  int64_t numElements = getNumberOfElements(shapedType.getShape());
  Type elementType = shapedType.getElementType();

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    res = bufferPtrs[bufferId];
  } else {
    // Use maximum size (double or int64_t) to avoid the precision loss.
    res = allocateBufferFor(constOp.getResult(), /*useMaxSize=*/true);
    DenseElementsAttr dataAttr =
        op->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (elementType.isa<FloatType>()) {
      // Use double to avoid the precision loss during computation.
      double *resArr = (double *)res;
      auto valueIt = dataAttr.getFloatValues().begin();
      for (int64_t i = 0; i < numElements; ++i) {
        double val = (double)(*valueIt++).convertToFloat();
        *(resArr + i) = val;
      }
    } else if (elementType.isa<IntegerType>()) {
      // Use int64_t to avoid the precision loss during computation.
      int64_t *resArr = (int64_t *)res;
      auto valueIt = dataAttr.getIntValues().begin();
      for (int64_t i = 0; i < numElements; ++i) {
        int64_t val = (*valueIt++).getSExtValue();
        *(resArr + i) = val;
      }
    } else
      llvm_unreachable("Unknown data type");

    // Store the buffer pointer.
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

  ShapedType shapedType = constOp.getResult().getType().cast<ShapedType>();
  int64_t maxSizeInBytes = getMaxSizeInBytes(shapedType);
  int64_t numElements = getNumberOfElements(shapedType.getShape());
  Type elementType = shapedType.getElementType();

  Attribute bufferIDAttr = op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR);
  if (bufferIDAttr) {
    unsigned bufferId = bufferIDAttr.cast<IntegerAttr>().getUInt();
    char *resArr = bufferPtrs[bufferId];
    if (elementType.isa<FloatType>()) {
      FloatType floatTy = elementType.cast<FloatType>();
      if (floatTy.getWidth() == 32) {
        double *resArrDouble = (double *)resArr;
        float *resArrFloat = (float *)res;
        for (int64_t i = 0; i < numElements; ++i)
          *(resArrFloat + i) = (float)*(resArrDouble + i);
      } else if (floatTy.getWidth() == 64) {
        std::copy(resArr, resArr + maxSizeInBytes, res);
      } else
        llvm_unreachable("Unknown data type");
    } else if (elementType.isa<IntegerType>()) {
      IntegerType intTy = elementType.cast<IntegerType>();
      if (intTy.getWidth() == 32) {
        int64_t *resArrInt64 = (int64_t *)resArr;
        int32_t *resArrInt32 = (int32_t *)res;
        for (int64_t i = 0; i < numElements; ++i)
          *(resArrInt32 + i) = (int32_t)(*(resArrInt64 + i));
      } else if (intTy.getWidth() == 64) {
        std::copy(resArr, resArr + maxSizeInBytes, res);
      } else
        llvm_unreachable("Unknown data type");
    } else
      llvm_unreachable("Unknown data type");
  } else {
    llvm_unreachable("Could not find the input buffer");
  }
}

/// A helper function to contruct a RankedTensorType from a ShapedType.
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// A helper function to construct a DenseElementsAttr from an array.
static DenseElementsAttr createDenseElementsAttr(char *arr, Type outputType) {
  int64_t sizeInBytes = getSizeInBytes(outputType);
  RankedTensorType resType =
      constructRankedTensorType(outputType.cast<ShapedType>());
  return DenseElementsAttr::getFromRawBuffer(
      resType, ArrayRef<char>(arr, sizeInBytes), /*isSplat=*/false);
}

/// A helper fucntion to check whether a value is produced by a dense
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
  int64_t maxSizeInBytes = getMaxSizeInBytes(replacingValue.getType());

  ONNXConstantOp constOp = rewriter.create<ONNXConstantOp>(loc,
      replacingValue.getType(), Attribute(), Attribute(), FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());

  // Store the buffer pointer.
  unsigned bufferId = -1;
  for (unsigned i = 0; i < bufferPtrs.size(); ++i)
    if (bufferPtrs[i] == vt) {
      bufferId = i;
      break;
    }
  if (bufferId == -1) {
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

// Template to generate binary operation results. It takes as inupt the element
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
  char *resArray = allocateBufferFor(replacingValue, /*useMaxSize=*/true);
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
  int64_t maxSizeInBytes = getMaxSizeInBytes(constValue.getType());

  // Get the const value.
  char *constArray =
      getArrayFromAttributeOrBuffer(rewriter, constValue.getDefiningOp());

  // Do calculation.
  // Use maximum size (double or int64_t) to avoid the precision loss.
  char *resArray = allocateBufferFor(replacingValue, /*useMaxSize=*/true);
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

template <typename T>
void IterateConstPropTranspose(char *constArray, ArrayRef<int64_t> constShape,
    ArrayRef<uint64_t> perm, char *resArray, ArrayRef<int64_t> resShape) {
  // Data pointers.
  T *constArrayT = reinterpret_cast<T *>(constArray);
  T *resArrayT = reinterpret_cast<T *>(resArray);

  // Get a reversed perm.
  SmallVector<uint64_t, 4> reversedPerm(perm.size(), 0);
  for (int i = 0; i < perm.size(); ++i)
    reversedPerm[perm[i]] = i;

  // Strides info.
  std::vector<int64_t> constStrides = getStrides(constShape);
  std::vector<int64_t> resStrides = getStrides(resShape);

  // Calculate transpose result.
  for (int64_t i = 0; i < getNumberOfElements(resShape); ++i) {
    // Indices.
    std::vector<int64_t> resIndices = getAccessIndex(i, resStrides);
    SmallVector<int64_t, 4> constIndices(perm.size(), 0);
    for (int j = 0; j < constIndices.size(); ++j)
      constIndices[j] = resIndices[reversedPerm[j]];
    // Transpose.
    int64_t constOffset = getLinearAccessIndex(constIndices, constStrides);
    int64_t resOffset = getLinearAccessIndex(resIndices, resStrides);
    *(resArrayT + resOffset) = *(constArrayT + constOffset);
  }
}

ONNXConstantOp ConstPropTranspose(
    PatternRewriter &rewriter, Value replacingValue, Value constValue) {
  ArrayRef<int64_t> replacingShape =
      replacingValue.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> constShape =
      constValue.getType().cast<ShapedType>().getShape();
  Type elementType =
      replacingValue.getType().cast<ShapedType>().getElementType();
  int64_t maxSizeInBytes = getMaxSizeInBytes(replacingValue.getType());

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
  char *resArray = allocateBufferFor(replacingValue, /*useMaxSize=*/true);
  if (elementType.isa<FloatType>()) {
    // Use double to avoid the precision loss during computation.
    IterateConstPropTranspose<double>(
        constArray, constShape, perm, resArray, replacingShape);
  } else if (elementType.isa<IntegerType>()) {
    // Use int64_t to avoid the precision loss during computation.
    IterateConstPropTranspose<int64_t>(
        constArray, constShape, perm, resArray, replacingShape);
  } else
    llvm_unreachable("Unknown data type");

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
  Type replacingType = replacingValue.getType();
  Type elementType = replacingType.cast<ShapedType>().getElementType();
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

template <typename T>
void IterateConstPropSplit(PatternRewriter &rewriter,
    std::vector<Value> &resValues, char *constArray,
    ArrayRef<int64_t> constShape, ArrayRef<Value> replacingValues,
    uint64_t splitAxis, ArrayRef<int64_t> splitOffsets) {
  // Basic info.
  int rank = constShape.size();
  int numOfResults = replacingValues.size();

  // Data pointers.
  T *constArrayT = reinterpret_cast<T *>(constArray);
  // Strides info.
  std::vector<int64_t> constStrides = getStrides(constShape);

  // Allocate temporary buffers.
  std::vector<char *> resBuffers;
  for (int i = 0; i < numOfResults; ++i) {
    // Use maximum size (double or int64_t) to avoid the precision loss.
    char *resArray = allocateBufferFor(replacingValues[i], /*useMaxSize=*/true);
    resBuffers.emplace_back(resArray);
  }

  // Do splitting
  for (int64_t i = 0; i < getNumberOfElements(constShape); ++i) {
    // Input indices.
    std::vector<int64_t> constIndices = getAccessIndex(i, constStrides);

    // Find the corresponding output and compute access indices.
    int toResult = numOfResults - 1;
    SmallVector<int64_t, 4> resIndices(rank, 0);
    for (int r = 0; r < rank; ++r) {
      if (r == splitAxis) {
        for (int k = 0; k < numOfResults - 1; ++k)
          if (constIndices[r] >= splitOffsets[k] &&
              constIndices[r] < splitOffsets[k + 1]) {
            toResult = k;
            break;
          }
        resIndices[r] = constIndices[r] - splitOffsets[toResult];
      } else {
        resIndices[r] = constIndices[r];
      }
    }

    // Get linear access indices.
    std::vector<int64_t> resStrides = getStrides(
        replacingValues[toResult].getType().cast<ShapedType>().getShape());
    int64_t resOffset = getLinearAccessIndex(resIndices, resStrides);

    // Copy data.
    T *resArrayT = reinterpret_cast<T *>(resBuffers[toResult]);
    *(resArrayT + resOffset) = *(constArrayT + i);
  }

  // Construct result values.
  for (int i = 0; i < numOfResults; ++i) {
    ONNXConstantOp res = createConstantOpAndStoreBufferPtr(
        rewriter, replacingValues[i], resBuffers[i]);
    resValues.emplace_back(res.getResult());
  }
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp splitOp, PatternRewriter &rewriter) const override {
    // Basic info.
    unsigned numOfResults = splitOp.getNumResults();
    Value input = splitOp.input();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elementType = inputType.getElementType();

    if (!isFromDenseONNXConstantOp(input))
      return failure();

    // Split axis.
    uint64_t splitAxis = splitOp.axisAttr().getValue().getSExtValue();
    // Compute split offsets.
    SmallVector<int64_t, 4> splitOffsets;
    {
      ArrayAttr splitAttr = splitOp.splitAttr();
      if (!splitAttr)
        // If split attribute is not specified, split size is equally divided.
        assert(inputShape[splitAxis] % numOfResults == 0 &&
               "The dimension at the split axis is expected to be divisible by "
               "the number of results");
      int64_t offset = 0;
      for (int i = 0; i < numOfResults; ++i) {
        splitOffsets.emplace_back(offset);
        if (splitAttr)
          offset += splitAttr.getValue()[i].cast<IntegerAttr>().getInt();
        else
          offset += inputShape[splitAxis] / numOfResults;
      }
    }

    // Get the constant input value.
    char *inputArray =
        getArrayFromAttributeOrBuffer(rewriter, input.getDefiningOp());

    SmallVector<Value, 4> replacingValues;
    for (int i = 0; i < numOfResults; ++i)
      replacingValues.emplace_back(splitOp.getResults()[i]);

    // Do splitting.
    std::vector<Value> resValues;
    if (elementType.isa<FloatType>()) {
      IterateConstPropSplit<double>(rewriter, resValues, inputArray, inputShape,
          replacingValues, splitAxis, splitOffsets);
    } else if (elementType.isa<IntegerType>()) {
      IterateConstPropSplit<int64_t>(rewriter, resValues, inputArray,
          inputShape, replacingValues, splitAxis, splitOffsets);
    } else
      llvm_unreachable("Unknown data type");

    rewriter.replaceOp(splitOp, resValues);
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
    : public PassWrapper<ConstPropONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void ConstPropONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  OwningRewritePatternList patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());

  applyPatternsAndFoldGreedily(function, std::move(patterns));

  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    Operation *op = constOp.getOperation();
    if (op->getAttrOfType<::mlir::Attribute>(BUFFER_ID_ATTR)) {
      char *arr = allocateBufferFor(constOp.getResult());
      getArrayForFinalOutput(op, arr);
      ShapedType type = constOp.getResult().getType().cast<ShapedType>();
      DenseElementsAttr denseAttr = createDenseElementsAttr(arr, type);
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
