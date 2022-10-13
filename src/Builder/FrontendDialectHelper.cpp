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
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SwapByteOrder.h"

#include "src/Builder/FrontendDialectHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

namespace {

// Parses unsigned number.
size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

// Reads external data from file location specified in tensor proto.
// See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md
std::unique_ptr<llvm::MemoryBuffer> readExternalData(
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  std::string location;
  uint64_t offset = 0;
  uint64_t length = -1; // MemoryBuffer uses -1 to mean infinity
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
  assert(!location.empty() && "missing external data location");
  llvm::SmallVector<char> path(externalDataDir.begin(), externalDataDir.end());
  llvm::sys::path::append(path, location);
  auto bufferOrError = llvm::MemoryBuffer::getFileSlice(
      path, length, offset, /*IsVolatile=*/false);
  if (std::error_code ec = bufferOrError.getError()) {
    std::string pathStr(path.data(), path.size());
    llvm::errs() << "Error " << ec.message() << " reading from file " << pathStr
                 << ", offset=" << offset << ", length=" << length << "\n";
    llvm_unreachable("llvm::MemoryBuffer::getFileSlice failed");
  }
  return std::move(bufferOrError.get());
}

template <typename T>
struct TransformValueToONNXData {
  static const google::protobuf::RepeatedField<int32_t> &data(
      const onnx::TensorProto &tp) {
    // int32_data is used for int32, uint8, int8, uint16, int16, bool
    return tp.int32_data();
  }
};

template <>
struct TransformValueToONNXData<double> {
  static const google::protobuf::RepeatedField<double> &data(
      const onnx::TensorProto &tp) {
    return tp.double_data();
  }
};

template <>
struct TransformValueToONNXData<float> {
  static const google::protobuf::RepeatedField<float> &data(
      const onnx::TensorProto &tp) {
    return tp.float_data();
  }
};

template <>
struct TransformValueToONNXData<int64_t> {
  static const google::protobuf::RepeatedField<int64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.int64_data();
  }
};

template <>
struct TransformValueToONNXData<uint32_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <>
struct TransformValueToONNXData<uint64_t> {
  static const google::protobuf::RepeatedField<uint64_t> &data(
      const onnx::TensorProto &tp) {
    return tp.uint64_data();
  }
};

template <typename T>
using DenseElementsAttrBuilder =
    llvm::function_ref<mlir::DenseElementsAttr(llvm::ArrayRef<T>)>;

// When the protobuf repeated field has a type of the same size as T,
// access the data directly via ArrayRef.
template <typename T, typename U>
std::enable_if_t<std::is_same_v<T, U>, mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(const google::protobuf::RepeatedField<U> &data,
    DenseElementsAttrBuilder<T> denseBuilder) {
  return denseBuilder(llvm::makeArrayRef(data.data(), data.size()));
}

// When the protobuf repeated field has a type larger than T,
// copy the data into correctly typed SmallVector because
// DenseElementsAttr needs argument type of the correct bitwidth.
template <typename T, typename U>
std::enable_if_t<!std::is_same_v<T, U>, mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(const google::protobuf::RepeatedField<U> &data,
    DenseElementsAttrBuilder<T> denseBuilder) {
  llvm::SmallVector<T> copy(data.begin(), data.end());
  return denseBuilder(llvm::makeArrayRef(copy));
}

// Returns DenseElementsAttr with tp's data.
template <typename T>
mlir::DenseElementsAttr createDenseElmAttr(const std::string &externalDataDir,
    const onnx::TensorProto &tp, DenseElementsAttrBuilder<T> denseBuilder) {
  std::unique_ptr<llvm::MemoryBuffer> externalData =
      (tp.has_data_location() &&
          tp.data_location() == onnx::TensorProto::EXTERNAL)
          ? readExternalData(externalDataDir, tp)
          : nullptr;
  if (externalData || tp.has_raw_data()) {
    llvm::StringRef buffer = externalData ? externalData->getBuffer()
                                          : llvm::StringRef(tp.raw_data());
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
      return denseBuilder(llvm::makeArrayRef(vector));
    } else {
      // No need to take care of endianness.
      return denseBuilder(arrayRef);
    }
  } else {
    // Not raw, no need to take care of endianness.
    const auto &data = TransformValueToONNXData<T>::data(tp);
    return createDenseElmAttrFromProtoData(data, denseBuilder);
  }
}

llvm::SmallVector<llvm::APFloat> U16ToF16Array(
    llvm::ArrayRef<uint16_t> u16arrayRef) {
  llvm::SmallVector<llvm::APFloat> f16Array;
  f16Array.reserve(u16arrayRef.size());
  for (uint16_t u : u16arrayRef)
    f16Array.emplace_back(llvm::APFloat::IEEEhalf(), llvm::APInt(16, u));
  return f16Array;
}

} // namespace

namespace onnx_mlir {

mlir::Value EmitInitializerForInputTensor(mlir::Location loc,
    mlir::OpBuilder &builder, const std::string &externalDataDir,
    const onnx::TensorProto &initializer) {
  // Return none if the initializer is an empty tensor, e.g tensor<0xf32>.
  llvm::ArrayRef<int64_t> tensorDims(
      initializer.dims().data(), initializer.dims().size());
  if (tensorDims.size() == 1 && tensorDims[0] == 0)
    return builder.create<mlir::ONNXNoneOp>(
        loc, builder.getNoneType(), builder.getUnitAttr());

  mlir::DenseElementsAttr denseElmAttr =
      onnxTensorProtoToDenseElmAttr(builder, externalDataDir, initializer);
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::DenseElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  mlir::Type elmType = convertONNXTypeToMLIRType(
      builder, (onnx::TensorProto_DataType)tp.data_type());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  auto denseBuilder = [tensorType](auto arrayRef) {
    return mlir::DenseElementsAttr::get(tensorType, arrayRef);
  };
  switch (tp.data_type()) {
  case (onnx::TensorProto::FLOAT16): {
    // F16s are converted bit-wise to U16s when written to protobufs.
    // So we read U16 from the protobuf and then convert to F16.
    auto denseBuilderU16ToF16 = [denseBuilder](auto arrayRef) {
      return denseBuilder(llvm::makeArrayRef(U16ToF16Array(arrayRef)));
    };
    return createDenseElmAttr<uint16_t>(
        externalDataDir, tp, denseBuilderU16ToF16);
  }
  case (onnx::TensorProto::FLOAT):
    return createDenseElmAttr<float>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::DOUBLE):
    return createDenseElmAttr<double>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::INT8):
    return createDenseElmAttr<int8_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::UINT8):
    return createDenseElmAttr<uint8_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::INT16):
    return createDenseElmAttr<int16_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::UINT16):
    return createDenseElmAttr<uint16_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::INT32):
    return createDenseElmAttr<int32_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::UINT32):
    return createDenseElmAttr<uint32_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::INT64):
    return createDenseElmAttr<int64_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::UINT64):
    return createDenseElmAttr<uint64_t>(externalDataDir, tp, denseBuilder);
  case (onnx::TensorProto::BOOL):
    return createDenseElmAttr<bool>(externalDataDir, tp, denseBuilder);
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
}

} // namespace onnx_mlir
