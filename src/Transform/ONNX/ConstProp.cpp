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

#include <fstream>
#include <malloc.h>
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

const StringRef FILE_NAME_ATTR = "file_name";

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

/// Get the element size in bytes.
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

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(Type ty) {
  ShapedType shapedType = ty.dyn_cast<ShapedType>();
  auto shape = shapedType.getShape();
  int64_t size = 1;
  for (int i = 0; i < shape.size(); i++)
    size *= shape[i];
  size *= getEltSizeInBytes(shapedType);
  return size;
}

/// Get the number of elements.
int64_t getNumberOfElements(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
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

/// Get a data array from a given ONNXConstantOp. If data were stored to a file,
/// get from the file. Otherwise, get from the dense attribute.
void getArrayFromAttributeOrFile(Operation *op, char *res) {
  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  assert(constOp && "Not a constant operation");

  int64_t size = getSizeInBytes(constOp.getResult().getType());

  Attribute fileNameAttr = op->getAttrOfType<::mlir::Attribute>(FILE_NAME_ATTR);
  if (fileNameAttr) {
    StringRef fileName = fileNameAttr.cast<StringAttr>().getValue();
    std::string pathStr = std::string(fileName.begin(), fileName.end());
    std::ifstream file(pathStr, std::ios::binary);
    file.read(res, size);
  } else {
    DenseElementsAttr dataAttr =
        op->getAttrOfType<::mlir::Attribute>("value")
            .dyn_cast_or_null<mlir::DenseElementsAttr>();
    ArrayRef<char> rawData = dataAttr.getRawData();
    std::copy(rawData.data(), rawData.data() + size, res);
  }
}

/// A helper function to contruct a RankedTensorType from a ShapedType.
RankedTensorType constructRankedTensorType(ShapedType type) {
  assert(type.hasRank() && "Not a ranked type");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

/// A helper function to construct a DenseElementsAttr from an array.
static DenseElementsAttr createDenseElementsAttr(char *arr, Type outputType) {
  int64_t size = getSizeInBytes(outputType);
  RankedTensorType resType =
      constructRankedTensorType(outputType.cast<ShapedType>());
  return DenseElementsAttr::getFromRawBuffer(
      resType, ArrayRef<char>(arr, size), /*isSplat=*/false);
}

/// A helper function to create an ONNXConstantOp for a given data array.
/// This ONNXConstantOp is only used internally.
ONNXConstantOp CreateDenseONNXConstantOp(
    PatternRewriter &rewriter, Value replacingValue, char *vt) {
  Location loc = replacingValue.getLoc();
  int64_t size = getSizeInBytes(replacingValue.getType());

  ONNXConstantOp constOp = rewriter.create<ONNXConstantOp>(
      loc, replacingValue.getType(), Attribute(), Attribute());

  // Write to file.
  llvm::SmallVector<char, 10> path;
  llvm::sys::fs::createTemporaryFile("constprop", "tmp", path);
  std::string pathStr = std::string(path.begin(), path.end());

  std::ofstream outfile(pathStr, std::ofstream::binary);
  outfile.write(vt, size);

  // Store the file name.
  constOp.getOperation()->setAttr(
      FILE_NAME_ATTR, rewriter.getStringAttr(pathStr));
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
  char *lhsArray = (char *)malloc(getSizeInBytes(lhs.getType()));
  getArrayFromAttributeOrFile(lhs.getDefiningOp(), lhsArray);
  char *rhsArray = (char *)malloc(getSizeInBytes(rhs.getType()));
  getArrayFromAttributeOrFile(rhs.getDefiningOp(), rhsArray);

  // Do calculation.
  char *resArray = (char *)malloc(getSizeInBytes(replacingValue.getType()));
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      IterateConstPropElementwiseBinary<ElementwiseBinaryOp, float>(
          lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
    } else if (floatTy.getWidth() == 64) {
      IterateConstPropElementwiseBinary<ElementwiseBinaryOp, double>(
          lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
    } else
      llvm_unreachable("Unknown data type");
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int32_t>(
          lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
    } else if (intTy.getWidth() == 64) {
      IterateConstPropElementwiseBinary<ElementwiseBinaryOp, int64_t>(
          lhsArray, rhsArray, lhsShape, rhsShape, resArray, outputShape);
    } else
      llvm_unreachable("Unknown data type");
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, resArray);

  // Clean up.
  free(lhsArray);
  free(rhsArray);
  free(resArray);

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
  int64_t size = getSizeInBytes(constValue.getType());

  // Get the const value.
  char *constArray = (char *)malloc(size);
  getArrayFromAttributeOrFile(constValue.getDefiningOp(), constArray);

  // Do calculation.
  char *resArray = (char *)malloc(size);
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      IterateConstPropElementwiseUnary<ElementwiseUnaryOp, float>(
          constArray, resArray, replacingShape);
    } else if (floatTy.getWidth() == 64) {
      IterateConstPropElementwiseUnary<ElementwiseUnaryOp, double>(
          constArray, resArray, replacingShape);
    } else
      llvm_unreachable("Unknown data type");
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      IterateConstPropElementwiseUnary<ElementwiseUnaryOp, int32_t>(
          constArray, resArray, replacingShape);
    } else if (intTy.getWidth() == 64) {
      IterateConstPropElementwiseUnary<ElementwiseUnaryOp, int64_t>(
          constArray, resArray, replacingShape);
    } else
      llvm_unreachable("Unknown data type");
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, resArray);

  // Clean up.
  free(constArray);
  free(resArray);
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
  int64_t size = getSizeInBytes(replacingValue.getType());

  // Get perm attribute.
  SmallVector<uint64_t, 4> perm;
  Attribute permAttr =
      replacingValue.getDefiningOp()->getAttrOfType<::mlir::Attribute>("perm");
  assert(permAttr && "permute attribute expected to be defined here");
  for (auto permVal : permAttr.cast<ArrayAttr>().getValue())
    perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

  // Get the const value.
  char *constArray = (char *)malloc(size);
  getArrayFromAttributeOrFile(constValue.getDefiningOp(), constArray);

  // Do calculation.
  char *resArray = (char *)malloc(size);
  if (elementType.isa<FloatType>()) {
    // FloatType
    FloatType floatTy = elementType.cast<FloatType>();
    if (floatTy.getWidth() == 32) {
      IterateConstPropTranspose<float>(
          constArray, constShape, perm, resArray, replacingShape);
    } else if (floatTy.getWidth() == 64) {
      IterateConstPropTranspose<double>(
          constArray, constShape, perm, resArray, replacingShape);
    } else
      llvm_unreachable("Unknown data type");
  } else if (elementType.isa<IntegerType>()) {
    // IntegerType
    IntegerType intTy = elementType.cast<IntegerType>();
    if (intTy.getWidth() == 32) {
      IterateConstPropTranspose<int32_t>(
          constArray, constShape, perm, resArray, replacingShape);
    } else if (intTy.getWidth() == 64) {
      IterateConstPropTranspose<int64_t>(
          constArray, constShape, perm, resArray, replacingShape);
    } else
      llvm_unreachable("Unknown data type");
  } else
    llvm_unreachable("Unknown data type");

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, resArray);

  free(constArray);
  free(resArray);
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

  char *resArray = (char *)malloc(getSizeInBytes(replacingValue.getType()));
  getArrayFromAttributeOrFile(inputOp, resArray);

  // Construct a new ONNXConstantOp.
  ONNXConstantOp res =
      CreateDenseONNXConstantOp(rewriter, replacingValue, resArray);

  free(resArray);
  return res;
}

//===----------------------------------------------------------------------===//
// Code to perform constant propagation for split.
//===----------------------------------------------------------------------===//

void RecurseConstPropSplit(PatternRewriter &rewriter,
    std::vector<Attribute> &resVector, DenseElementsAttr attr,
    SmallVector<uint64_t, 4> &indices, uint64_t splitAxis, uint64_t axisOffset,
    uint64_t axisSize, int freeRank) {
  if (freeRank == 0) {
    // Fully defined ranks.
    Attribute res = attr.getValue(ArrayRef<uint64_t>(indices));
    resVector.emplace_back(res);
  } else {
    // Recurse.
    ArrayRef<int64_t> shape = attr.getType().getShape();
    int rank = shape.size();
    int index = rank - freeRank;
    int start, size;
    if (index == splitAxis) {
      start = axisOffset;
      size = axisSize;
    } else {
      start = 0;
      size = attr.getType().getShape()[index];
    }
    for (int i = start; i < start + size; ++i) {
      indices[index] = i;
      RecurseConstPropSplit(rewriter, resVector, attr, indices, splitAxis,
          axisOffset, axisSize, freeRank - 1);
    }
  }
}

DenseElementsAttr ConstPropSplit(PatternRewriter &rewriter, Value resOperand,
    Attribute attr, IntegerAttr axisAttr, ArrayAttr splitAttr,
    unsigned resIndex) {
  // Read dense attribute, the constant tensor we are transforming.
  DenseElementsAttr denseAttr =
      attr.dyn_cast_or_null<mlir::DenseElementsAttr>();
  assert(denseAttr && "expected dense attribute");
  RankedTensorType resType =
      constructRankedTensorType(resOperand.getType().cast<ShapedType>());
  unsigned rank = denseAttr.getType().getShape().size();
  // Read split axis.
  uint64_t splitAxis = axisAttr.getValue().getSExtValue();
  // Read split vector.
  SmallVector<uint64_t, 4> splits;
  assert(splitAttr && "split attribute expected to be defined here");
  for (Attribute splitVal : splitAttr.getValue())
    splits.emplace_back(splitVal.cast<IntegerAttr>().getInt());
  // Compute the range of elements of interest in the given axis.
  uint64_t axisOffset = 0, axisSize = splits[resIndex];
  for (int i = 0; i < resIndex; ++i)
    axisOffset += splits[i];
  // Init indice vector.
  SmallVector<uint64_t, 4> indices(rank, -1);
  std::vector<Attribute> resVector;
  // Copy.
  RecurseConstPropSplit(rewriter, resVector, denseAttr, indices, splitAxis,
      axisOffset, axisSize, rank);
  ArrayRef<Attribute> resRef(resVector);
  return DenseElementsAttr::get(resType, resRef);
}

class ConstPropSplitPattern : public OpRewritePattern<ONNXSplitOp> {
public:
  using OpRewritePattern<ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSplitOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // A dense attribute that contains constant values of the split op's
    // input.
    Attribute denseAttr;

    // Match
    ONNXSplitOp *splitOp = ::llvm::dyn_cast_or_null<::mlir::ONNXSplitOp>(&op);
    {
      Operation *producerOp = splitOp->input().getDefiningOp();
      ONNXConstantOp castedProducerOp =
          ::llvm::dyn_cast_or_null<::mlir::ONNXConstantOp>(producerOp);
      if (!castedProducerOp)
        return failure();
      // Check whether the constant op is using a dense value or not.
      Attribute sparseAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("sparse_value");
      if (sparseAttr)
        return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
          diag << "entities '' failed to satisfy constraint: Attribute "
                  "is null";
        });
      Attribute dataAttr =
          producerOp->getAttrOfType<::mlir::Attribute>("value");
      denseAttr = dataAttr;
    }

    // Rewrite
    unsigned outputNum = splitOp->getNumResults();
    Value splitInput = splitOp->input();
    int64_t rank = splitInput.getType().cast<ShapedType>().getRank();
    IntegerAttr axisAttr = splitOp->axisAttr();
    ArrayAttr splitAttr = splitOp->splitAttr();
    if (!splitAttr) {
      // If split attribute is not specified, it is constructed from input.
      ArrayRef<int64_t> shape =
          splitInput.getType().cast<ShapedType>().getShape();
      uint64_t splitAxis = axisAttr.getValue().getSExtValue();
      assert(shape[splitAxis] % outputNum == 0 &&
             "The dimension at the split axis is expected to be divisible by "
             "the number of results");
      Attribute splitSize = rewriter.getIntegerAttr(
          rewriter.getIntegerType(64), shape[splitAxis] / outputNum);
      SmallVector<Attribute, 4> splits(outputNum, splitSize);
      splitAttr = rewriter.getArrayAttr(splits);
    }

    SmallVector<::mlir::Value, 4> returnValues;
    for (int i = 0; i < outputNum; ++i) {
      Value splitOutput = splitOp->getResults()[i];
      Value constOp =
          rewriter.create<ONNXConstantOp>(loc, splitOutput.getType(),
              /*sparse_value=*/Attribute(),
              /*dense_value=*/
              ConstPropSplit(
                  rewriter, splitOutput, denseAttr, axisAttr, splitAttr, i));
      returnValues.emplace_back(constOp);
    }

    rewriter.replaceOp(op, returnValues);
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

  OwningRewritePatternList patterns;
  populateWithGenerated(context, patterns);
  patterns.insert<ConstPropSplitPattern>(&getContext());

  applyPatternsAndFoldGreedily(function, std::move(patterns));

  // Create DenseElementsAttr and clean up helper attributes.
  function.walk([&](ONNXConstantOp constOp) {
    Operation *op = constOp.getOperation();
    if (op->getAttrOfType<::mlir::Attribute>(FILE_NAME_ATTR)) {
      int64_t size = getSizeInBytes(constOp.getResult().getType());
      ShapedType type = constOp.getResult().getType().cast<ShapedType>();
      char *arr = (char *)malloc(size);
      getArrayFromAttributeOrFile(op, arr);
      DenseElementsAttr denseAttr = createDenseElementsAttr(arr, type);
      op->setAttr("value", denseAttr);
      op->removeAttr(FILE_NAME_ATTR);
      free(arr);
    }
  });
} // end anonymous namespace

/*!
 * Create a ConstPropONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createConstPropONNXToONNXPass() {
  return std::make_unique<ConstPropONNXToONNXPass>();
}
