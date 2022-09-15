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
#include <llvm/Support/Path.h>
#include <llvm/Support/SwapByteOrder.h>

#include "src/Builder/FrontendDialectHelper.hpp"

namespace onnx_mlir {

ExternalDataReader::ExternalDataReader(const std::string &externalDataDir)
    : externalDataDir(externalDataDir) {}

ExternalDataReader::~ExternalDataReader() {}

llvm::StringRef ExternalDataReader::read(
    const std::string &fileName, size_t offset, llvm::Optional<size_t> length) {
  if (externalDataDir.empty()) {
    llvm::errs() << "external data read from " << fileName << " rejected\n";
    llvm_unreachable(
        "attempted to read external data without externalDataDir set");
  }
  llvm::StringRef buffer;
  auto it = files.find(fileName);
  if (it != files.end()) {
    buffer = it->second->getBuffer();
  } else {
    llvm::SmallVector<char> path(
        externalDataDir.begin(), externalDataDir.end());
    llvm::sys::path::append(path, fileName);
    auto bufferOrError = llvm::MemoryBuffer::getFile(
        path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (std::error_code ec = bufferOrError.getError()) {
      std::string pathStr(path.data(), path.size());
      llvm::errs() << "Error " << ec.message() << " reading from file "
                   << pathStr << "\n";
      llvm_unreachable("getFile failed");
    }
    buffer = bufferOrError.get()->getBuffer();
    files.emplace(fileName, std::move(bufferOrError.get()));
  }
  assert(offset <= buffer.size() && "read past end of external data file");
  if (length.has_value()) {
    assert(offset + length.value() <= buffer.size() &&
           "read past end of external data file");
    return buffer.substr(offset, length.value());
  } else {
    return buffer.substr(offset);
  }
}

size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

llvm::Optional<llvm::StringRef> dataBytes(
    ExternalDataReader &dataReader, const onnx::TensorProto &tp) {
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    std::string location;
    size_t offset = 0;
    llvm::Optional<size_t> length;
    for (const onnx::StringStringEntryProto &entry : tp.external_data()) {
      assert(entry.has_key() && "external_data entry must have key");
      assert(entry.has_value() && "external_data entry must have value");
      if (entry.key() == "location") {
        location = entry.value();
      } else if (entry.key() == "offset") {
        offset = parseOffsetOrLength(entry.value());
      } else if (entry.key() == "length") {
        length = parseOffsetOrLength(entry.value());
      }
    }
    assert(!location.empty() && "external data has no location");
    return dataReader.read(location, offset, length);
  }
  if (tp.has_raw_data()) {
    return llvm::StringRef(tp.raw_data());
  }
  return llvm::None;
}

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<T> data(
      const onnx::TensorProto &tp) {
    return google::protobuf::RepeatedField<T>();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> data(
      const onnx::TensorProto &tp) {
    return tp.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> data(
      const onnx::TensorProto &tp) {
    return tp.float_data();
  }
};

template <>
struct TransformValueToONNXData<int16_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int32_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> data(
      const onnx::TensorProto &tp) {
    return tp.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint8_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<int8_t> {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<bool> {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    return tp.int32_data();
  }
};

template <typename T>
mlir::DenseElementsAttr createDenseElmAttr(onnx::TensorProto tp,
    llvm::Optional<llvm::StringRef> bytes, mlir::RankedTensorType tensorType) {
  if (bytes.has_value()) {
    llvm::ArrayRef<T> arrayRef(
        reinterpret_cast<T const *>(bytes.value().data()),
        bytes.value().size());
    // Perform byte swap if system endianness is BE.
    // ONNX tensor content raw data is always in LE.
    if (sizeof(T) > 1 && llvm::support::endian::system_endianness() !=
                             llvm::support::endianness::little) {
      size_t size = arrayRef.size() / sizeof(T);
      llvm::SmallVector<T> vector;
      vector.reserve(size);
      for (T x : arrayRef) {
        vector.push_back(llvm::sys::getSwappedBytes(x));
      }
      return mlir::DenseElementsAttr::get(
          tensorType, llvm::makeArrayRef(vector));
    } else {
      // No need to take care of endianness.
      return mlir::DenseElementsAttr::get(tensorType, arrayRef);
    }
  } else {
    // Copy, no need to take care of endianness.
    auto data = TransformValueToONNXData<T>::data(tp);
    llvm::SmallVector<T> vector(data.begin(), data.end());
    return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(vector));
  }
}

mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
    mlir::OpBuilder &builder, ExternalDataReader &dataReader,
    const onnx::TensorProto &initializer) {
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
      onnxTensorProtoToDenseElmAttr(builder, dataReader, initializer);

  // Create ConstantOp for dense array.
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    ExternalDataReader &dataReader, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  mlir::Type elmType = convertONNXTypeToMLIRType(
      builder, (onnx::TensorProto_DataType)tp.data_type());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  auto bytes = dataBytes(dataReader, tp);
  switch (tp.data_type()) {
  case (onnx::TensorProto::FLOAT):
    return createDenseElmAttr<float>(tp, bytes, tensorType);
  case (onnx::TensorProto::DOUBLE):
    return createDenseElmAttr<double>(tp, bytes, tensorType);
  case (onnx::TensorProto::INT8):
    return createDenseElmAttr<int8_t>(tp, bytes, tensorType);
  case (onnx::TensorProto::UINT8):
    return createDenseElmAttr<uint8_t>(tp, bytes, tensorType);
  case (onnx::TensorProto::INT16):
    return createDenseElmAttr<int16_t>(tp, bytes, tensorType);
  case (onnx::TensorProto::INT32):
    return createDenseElmAttr<int32_t>(tp, bytes, tensorType);
  case (onnx::TensorProto::INT64):
    return createDenseElmAttr<int64_t>(tp, bytes, tensorType);
  case (onnx::TensorProto::BOOL):
    return createDenseElmAttr<bool>(tp, bytes, tensorType);
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
}
} // namespace onnx_mlir
