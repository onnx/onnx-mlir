/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- FrontendDialectHelper.cpp ----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for handling input ONNX models.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include <llvm/Support/Endian.h>
#include <llvm/Support/SwapByteOrder.h>

#include "src/Builder/FrontendDialectHelper.hpp"

namespace onnx_mlir {

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<T> data(
      onnx::TensorProto initializer) {
    return google::protobuf::RepeatedField<T>();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> data(
      onnx::TensorProto initializer) {
    return initializer.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> data(
      onnx::TensorProto initializer) {
    return initializer.float_data();
  }
};

template <>
struct TransformValueToONNXData<int16_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int32_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> data(
      onnx::TensorProto initializer) {
    return initializer.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint8_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int8_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

template <>
struct TransformValueToONNXData<bool> {
  static const google::protobuf::RepeatedField<int32_t> data(
      onnx::TensorProto initializer) {
    return initializer.int32_data();
  }
};

// Helper method for constructing an array attribute from a model input.
template <typename T>
std::vector<T> CreateArrayAttribute(onnx::TensorProto initializer) {
  size_t size;
  if (initializer.raw_data().size()) {
    // Copy & take care of endianness.
    std::vector<char> byteInitializer;
    std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
        back_inserter(byteInitializer));
    size = initializer.raw_data().size() / sizeof(T);
    T *arrayPtr = reinterpret_cast<T *>(&byteInitializer[0]);
    auto array = std::vector<T>(arrayPtr, arrayPtr + size);
    // Perform byte swap if system endianness is BE.
    // ONNX tensor content raw data is always in LE.
    if (llvm::support::endian::system_endianness() !=
        llvm::support::endianness::little)
      for (size_t i = 0; i < array.size(); i++)
        llvm::sys::swapByteOrder<T>(array[i]);

    return array;
  }

  // Copy, no need to take care of endianness.
  auto data = TransformValueToONNXData<T>::data(initializer);
  size = data.size();
  return std::vector<T>(&data[0], &data[0] + size);
}

template <>
std::vector<bool> CreateArrayAttribute<bool>(onnx::TensorProto initializer) {
  // Copy, no need to take care of endianness.
  if (initializer.raw_data().size()) {
    std::vector<bool> bitInitializer;
    std::copy(initializer.raw_data().begin(), initializer.raw_data().end(),
        back_inserter(bitInitializer));
    return bitInitializer;
  }

  auto data = TransformValueToONNXData<bool>::data(initializer);
  return std::vector<bool>(&data[0], &data[0] + data.size());
}

mlir::Value InitializedTensorMapping::EmitInitializerForInputTensor(
    mlir::Location loc, mlir::OpBuilder &builder, const std::string &name) {
  // Initializer for input.
  onnx::TensorProto initializer = GetInitializedTensor(name);

  // Return none if the initializer is an empty tensor, e.g tensor<0xf32>.
  llvm::ArrayRef<int64_t> tensorDims(
      initializer.dims().data(), initializer.dims().size());
  if (tensorDims.size() == 1 && tensorDims[0] == 0)
    return builder.create<mlir::ONNXNoneOp>(
        loc, builder.getNoneType(), builder.getUnitAttr());

  // Emit ConstantOp and record the mapping between the input and
  // the constant value.
  // Create value attribute.
  mlir::DenseElementsAttr denseElmAttr =
      onnxTensorProtoToDenseElmAttr(builder, initializer);

  // Create ConstantOp for dense array.
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(
    mlir::OpBuilder &builder, const onnx::TensorProto &initializer) {
  // Tensor dimensions.
  llvm::ArrayRef<int64_t> tensorDims(
      initializer.dims().data(), initializer.dims().size());
  mlir::DenseElementsAttr denseElmAttr;
  switch (initializer.data_type()) {
  case (onnx::TensorProto::FLOAT): {
    const auto &arrayAttrInitializer = CreateArrayAttribute<float>(initializer);
    auto elmType = builder.getF32Type();
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::DOUBLE): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<double>(initializer);
    auto elmType = builder.getF64Type();
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::INT8): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<int8_t>(initializer);
    auto elmType = builder.getIntegerType(8);
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::UINT8): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<uint8_t>(initializer);
    auto elmType = builder.getIntegerType(8, false);
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::INT16): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<int16_t>(initializer);
    auto elmType = builder.getIntegerType(16);
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::INT32): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<int32_t>(initializer);
    auto elmType = builder.getIntegerType(32);
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::INT64): {
    const auto &arrayAttrInitializer =
        CreateArrayAttribute<int64_t>(initializer);
    auto elmType = builder.getIntegerType(64);
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::makeArrayRef(arrayAttrInitializer));
    break;
  }
  case (onnx::TensorProto::BOOL): {
    const auto &data = CreateArrayAttribute<bool>(initializer);
    auto elmType = builder.getI1Type();
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    denseElmAttr = mlir::DenseElementsAttr::get(
        tensorType, llvm::SmallVector<bool, 64>(data.begin(), data.end()));
    break;
  }
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
  return denseElmAttr;
}
} // namespace onnx_mlir
