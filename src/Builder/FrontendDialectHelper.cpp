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
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SwapByteOrder.h>

#include "src/Builder/FrontendDialectHelper.hpp"

namespace onnx_mlir {

std::unique_ptr<llvm::MemoryBuffer> readExternalData(
    const std::string &externalDataDir, const std::string &location,
    size_t offset, llvm::Optional<size_t> length) {
  if (externalDataDir.empty()) {
    llvm::errs() << "external data read from " << location << " rejected\n";
    llvm_unreachable(
        "attempted to read external data without externalDataDir set");
  }
  llvm::SmallVector<char> path(externalDataDir.begin(), externalDataDir.end());
  llvm::sys::path::append(path, location);
  // MemoryBuffer implementation uses -1 to mean uint64_t infinity
  uint64_t mapSize = length.has_value() ? length.value() : -1;
  auto bufferOrError = llvm::MemoryBuffer::getFileSlice(
      path, mapSize, offset, /*IsVolatile=*/false);
  if (std::error_code ec = bufferOrError.getError()) {
    std::string pathStr(path.data(), path.size());
    llvm::errs() << "Error " << ec.message() << " reading from file " << pathStr
                 << " from " << offset << " to " << mapSize << "\n";
    llvm_unreachable("getFile failed");
  }
  return std::move(bufferOrError.get());
}

size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

std::unique_ptr<llvm::MemoryBuffer> dataBytes(
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
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
    return readExternalData(externalDataDir, location, offset, length);
  } else if (tp.has_raw_data()) {
    return llvm::MemoryBuffer::getMemBuffer(
        tp.raw_data(), /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  } else {
    return nullptr;
  }
}

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<int32_t> data(
      const onnx::TensorProto &tp) {
    // int32_data is used for int32, uint8, int8, uint16, int16, bool
    return tp.int32_data();
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
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> data(
      const onnx::TensorProto &tp) {
    return tp.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint32_t> {
  static const google::protobuf::RepeatedField<uint64_t> data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <>
struct TransformValueToONNXData<uint64_t> {
  static const google::protobuf::RepeatedField<uint64_t> data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <typename T>
mlir::DenseElementsAttr createDenseElmAttr(onnx::TensorProto tp,
    llvm::MemoryBuffer *bytes, mlir::RankedTensorType tensorType) {
  if (bytes) {
    llvm::StringRef buffer = bytes->getBuffer();
    size_t size = buffer.size() / sizeof(T);
    llvm::ArrayRef<T> arrayRef(
        reinterpret_cast<T const *>(buffer.data()), size);
    // Perform byte swap if system endianness is BE.
    // ONNX tensor content raw data is always in LE.
    if (sizeof(T) > 1 && llvm::support::endian::system_endianness() !=
                             llvm::support::endianness::little) {
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
    // Not raw, no need to take care of endianness.
    auto data = TransformValueToONNXData<T>::data(tp);
    // Access data directly via ArrayRef if same size as T,
    // or copy into correctly typed SmallVector otherwise
    // because DenseElementsAttr needs argument type of the correct bitwidth.
    typedef typename std::conditional<sizeof(T) == sizeof(data[0]),
        llvm::ArrayRef<T>, llvm::SmallVector<T>>::type ArrayRefOrSmallVector;
    ArrayRefOrSmallVector array(data.begin(), data.end());
    return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(array));
  }
}

mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
    mlir::OpBuilder &builder, const std::string &externalDataDir,
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
      onnxTensorProtoToDenseElmAttr(builder, externalDataDir, initializer);

  // Create ConstantOp for dense array.
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  mlir::Type elmType = convertONNXTypeToMLIRType(
      builder, (onnx::TensorProto_DataType)tp.data_type());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  std::unique_ptr<llvm::MemoryBuffer> bytes = dataBytes(externalDataDir, tp);
  switch (tp.data_type()) {
  case (onnx::TensorProto::FLOAT):
    return createDenseElmAttr<float>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::DOUBLE):
    return createDenseElmAttr<double>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::INT8):
    return createDenseElmAttr<int8_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::UINT8):
    return createDenseElmAttr<uint8_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::INT16):
    return createDenseElmAttr<int16_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::UINT16):
    return createDenseElmAttr<uint16_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::INT32):
    return createDenseElmAttr<int32_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::UINT32):
    return createDenseElmAttr<uint32_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::INT64):
    return createDenseElmAttr<int64_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::UINT64):
    return createDenseElmAttr<uint64_t>(tp, bytes.get(), tensorType);
  case (onnx::TensorProto::BOOL):
    return createDenseElmAttr<bool>(tp, bytes.get(), tensorType);
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
}
} // namespace onnx_mlir
