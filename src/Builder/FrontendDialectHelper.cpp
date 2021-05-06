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

// Helper method for constructing an array attribute from a model input.
template <typename T>
static std::vector<T> CreateArrayAttribute(onnx::TensorProto initializer) {
  size_t size;
  if (initializer.raw_data().size()) {
    // copy & take care of endianness
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
      for (int i = 0; i < array.size(); i++)
        llvm::sys::swapByteOrder<T>(array[i]);

    return array;
  }

  // copy, no need to take care of endianness
  auto data = TransformValueToONNXData<T>::data(initializer);
  size = data.size();
  return std::vector<T>(&data[0], &data[0] + size);
}

mlir::Value InitializedTensorMapping::EmitInitializerForInputTensor(
    mlir::Location loc, mlir::OpBuilder &builder, const std::string &name) {
  // Initializer for input.
  onnx::TensorProto initializer = GetInitializedTensor(name);

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
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
  return denseElmAttr;
}

// Convert type to MLIR type.
// A complete list of types can be found in:
// <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
// TODO: Update Int*/Uint* to emit signed/unsigned MLIR types
mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType) {
  switch (onnxType) {
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    return builder_.getF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
    return builder_.getF32Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    return builder_.getF64Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
    return builder_.getIntegerType(/*width=*/8);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
    return builder_.getIntegerType(/*width=*/8, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
    return builder_.getIntegerType(/*width=*/16);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
    return builder_.getIntegerType(/*width=*/16, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
    return builder_.getIntegerType(/*width=*/32);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
    return builder_.getIntegerType(/*width=*/32, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
    return builder_.getIntegerType(/*width=*/64);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
    return builder_.getIntegerType(/*width=*/64, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
    return builder_.getI1Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
    return mlir::onnxmlir::StringType::get(builder_.getContext());

  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
  default:
    assert(false && "Unsupported data type encountered.");
    return nullptr;
  }
}

} // namespace onnx_mlir
