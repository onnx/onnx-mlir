/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- TypeUtilities.cpp - functions related to MLIR Type -------===//
//
// Copyright 2022-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code related to MLIR Type, e.g. RankedTensorType,
// MemRefType, etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

/// Get element type.
Type getElementType(Type ty) { return getElementTypeOrSelf(ty); }

/// Check if a type is ShapedType and has rank.
bool isRankedShapedType(Type ty) {
  return (mlir::isa<ShapedType>(ty) && mlir::cast<ShapedType>(ty).hasRank());
}

/// Check if a type has static shape.
bool hasStaticShape(Type ty) {
  if (!isRankedShapedType(ty))
    return false;
  return mlir::cast<ShapedType>(ty).hasStaticShape();
}

/// Get shape.
ArrayRef<int64_t> getShape(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return mlir::cast<ShapedType>(ty).getShape();
}

/// Get specific shape value. If is a scalar, return 1.
int64_t getShape(mlir::Type ty, int64_t index) {
  int64_t rank = getRank(ty);
  if (rank == 0) {
    // We have a scalar, return size of 1.
    assert((index == 0 || index == -1) && "bad index for scalar");
    return 1;
  }
  if (index < 0)
    index += rank;
  assert(index >= 0 && index < rank && "out of range index [-rank...rank-1]");
  return getShape(ty)[index];
}

/// Get rank.
int64_t getRank(Type ty) {
  assert(isRankedShapedType(ty) && "Type must be ranked");
  return mlir::cast<ShapedType>(ty).getRank();
}

/// Get the number of elements.
int64_t getNumberOfElements(ShapedType ty) {
  assert(ty.hasStaticShape() && "Has unknown dimensions");
  ArrayRef<int64_t> shape = getShape(ty);
  return ShapedType::getNumElements(shape);
}

/// Get the element size in bytes.
int64_t getEltSizeInBytes(Type ty) {
  Type elementType = getElementTypeOrSelf(ty);
  int64_t sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = mlir::cast<VectorType>(elementType);
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the size of a tensor from its ranked type in bytes.
int64_t getSizeInBytes(ShapedType ty) {
  ArrayRef<int64_t> shape = getShape(ty);
  assert(ty.hasStaticShape() && "Has unknown dimensions");
  return ShapedType::getNumElements(shape) * getEltSizeInBytes(ty);
}

/// Check if two RankedTensorTypes have the same encoding attribute or not.
bool sameEncodingAttr(Type t1, Type t2) {
  if (auto rtp1 = llvm::dyn_cast<RankedTensorType>(t1))
    if (auto rtp2 = llvm::dyn_cast<RankedTensorType>(t2)) {
      return rtp1.getEncoding() == rtp2.getEncoding();
    }
  return false;
}

/// Get the byte width of an int or float type.
unsigned getIntOrFloatByteWidth(Type ty) {
  return llvm::divideCeil(ty.getIntOrFloatBitWidth(), 8);
}

std::map<int64_t, std::vector<int64_t>> parseShapeInformation(
    const std::string &shapeInformation) {
  // For users of onnx-mlir.
  // -1 is used for dynamic/unknown dimension.
  static constexpr int64_t kUserDynamic = -1;
  // -1 is used to indicate all input indices.
  static constexpr int64_t kUserAllInputs = -1;

  // Separator between inputs.
  static constexpr char INPUT_SEP = ',';
  // Separator between dimensions.
  static constexpr char DIM_SEP = 'x';
  // Separator between one input and its dimensions.
  static constexpr char INPUT_DIM_SEP = ':';
  // Separator to define a range of input indices, e.g. 2-5.
  static constexpr char INPUT_RANGE_SEP = '-';
  std::map<int64_t, std::vector<int64_t>> inputs_shape_information;
  if (!shapeInformation.empty()) {
    std::stringstream shapeInfoString(shapeInformation);
    std::string shapeString;
    while (std::getline(shapeInfoString, shapeString, INPUT_SEP)) {
      size_t pos = shapeString.find(INPUT_DIM_SEP);
      std::string inputString = shapeString.substr(0, pos);
      std::string dimString = shapeString.substr(pos + 1);

      // Parse dimString.
      std::stringstream dimSizes(dimString);
      std::string dimStr;
      std::vector<int64_t> dims;
      while (std::getline(dimSizes, dimStr, DIM_SEP)) {
        int64_t dimSize = std::stoi(dimStr);
        assert((dimSize == kUserDynamic || dimSize > 0) &&
               "dim must be -1 or > 0");
        if (dimSize == kUserDynamic)
          dimSize = ShapedType::kDynamic;
        dims.emplace_back(dimSize);
      }

      // Parse inputString.
      assert(std::count(inputString.begin(), inputString.end(),
                 INPUT_RANGE_SEP) <= 1 &&
             "input_id is invalid");
      // Check if users input a range or not.
      size_t rangePos = inputString.find(INPUT_RANGE_SEP);
      std::string startString = inputString.substr(0, rangePos);
      std::string endString = inputString.substr(rangePos + 1);
      assert(endString != "" && "input_id has _ at the end");
      bool isRangeInput = (startString != "");
      // Insert (input_id, dim_value) to the shape info.
      SmallVector<int64_t> inputIDs;
      if (isRangeInput) {
        int64_t startID = std::stoi(startString);
        int64_t endID = std::stoi(endString);
        assert((startID >= 0) && "start_id must be >= 0");
        assert((endID >= 0) && "end_id must be >= 0");
        for (int64_t i = startID; i <= endID; ++i)
          inputIDs.emplace_back(i);
      } else {
        int64_t inputID = std::stoi(inputString);
        assert((inputID >= 0 || inputID == kUserAllInputs) &&
               "input_id must be -1 or >= 0");
        inputIDs.emplace_back(inputID);
      }
      for (int64_t inputID : inputIDs) {
        // The semantics of c++ map.insert() makes sure that only the first
        // setting of inputID is inserted.
        inputs_shape_information.insert(std::make_pair(inputID, dims));
      }
    }
  }
  return inputs_shape_information;
}

} // namespace onnx_mlir
