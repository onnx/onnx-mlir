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
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/FloatingPoint16.hpp"

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
    // int32_data is used for:
    // int32, uint8, int8, uint16, int16, bool, float_16, bfloat_16
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

// Converts to the cpp type 'To' that correspond's to the tensor element type
// (bool, int8, float_16, uint32, etc) from the the proto data field type
// which may be a wider type (int32, uint64). In most cases the conversion is
// just standard C implicit conversion. The exception is float_16 and bfloat_16
// which must be bit-wise converted from uint16_t.
template <typename To, typename From>
To deserializeDatum(const From &from) {
  if constexpr (onnx_mlir::isFP16Type<To>)
    return To::bitcastFromU16(from);
  else
    return from;
}

// When the protobuf repeated field has type T,
// access the data directly via ArrayRef.
template <typename T, typename Repeated>
std::enable_if_t<std::is_same_v<T, typename Repeated::value_type>,
    mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(
    mlir::ShapedType tensorType, const Repeated &data) {
  return mlir::DenseElementsAttr::get(
      tensorType, llvm::makeArrayRef(data.data(), data.size()));
}

// When the protobuf repeated field has a type different from T,
// copy the data into correctly typed SmallVector because
// DenseElementsAttr needs argument type of the correct bitwidth.
template <typename T, typename Repeated>
std::enable_if_t<!std::is_same_v<T, typename Repeated::value_type>,
    mlir::DenseElementsAttr>
createDenseElmAttrFromProtoData(
    mlir::ShapedType tensorType, const Repeated &data) {
  llvm::SmallVector<T> copy;
  copy.resize_for_overwrite(data.size());
  std::transform(data.begin(), data.end(), copy.data(),
      deserializeDatum<T, typename Repeated::value_type>);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(copy));
}

// Perform byte swap if system endianness is BE.
// ONNX tensor content raw data is always in LE.
// Don't byte swap single byte types, because that's unnecessary
// and llvm::sys::getSwappedBytes(bool) also happens to be broken.
template <typename T>
constexpr bool shouldSwapLEBytes =
    sizeof(T) > 1 && llvm::support::endian::system_endianness() !=
                         llvm::support::endianness::little;

// Extension of llvm::sys::getSwappedBytes to also handle float_16, bfloat_16.
template <typename T>
T swappedBytes(T x) {
  if constexpr (onnx_mlir::isFP16Type<T>)
    return T::bitcastFromU16(llvm::sys::getSwappedBytes(x.bitcastToU16()));
  else
    return llvm::sys::getSwappedBytes(x);
}

// Returns DenseElementsAttr with tp's data.
template <typename T>
mlir::DenseElementsAttr createDenseElmAttr(mlir::ShapedType tensorType,
    const onnx::TensorProto &tp, const std::string &externalDataDir) {
  std::unique_ptr<llvm::MemoryBuffer> externalData =
      (tp.has_data_location() &&
          tp.data_location() == onnx::TensorProto::EXTERNAL)
          ? readExternalData(externalDataDir, tp)
          : nullptr;
  if (externalData || tp.has_raw_data()) {
    llvm::StringRef buffer = externalData ? externalData->getBuffer()
                                          : llvm::StringRef(tp.raw_data());
    size_t size = buffer.size() / sizeof(T);
    llvm::ArrayRef<T> array(reinterpret_cast<T const *>(buffer.data()), size);
    if (shouldSwapLEBytes<T>) {
      llvm::SmallVector<T> copy;
      copy.resize_for_overwrite(size);
      std::transform(array.begin(), array.end(), copy.data(), swappedBytes<T>);
      return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(copy));
    } else {
      return mlir::DenseElementsAttr::get(tensorType, array);
    }
  } else {
    // Not raw, no need to take care of endianness.
    const auto &data = TransformValueToONNXData<T>::data(tp);
    return createDenseElmAttrFromProtoData<T>(tensorType, data);
  }
}

mlir::DenseElementsAttr createDenseStringElmAttr(
    mlir::ShapedType tensorType, const onnx::TensorProto &tp) {
  // The string type is different from other data types in that it cannot be
  // raw or external data and it needs to be converted to StringRef
  // (or StringAttr) to construct a DenseElementsAttr.
  assert(!(tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) &&
         "string TensorProto cannot be external data");
  assert(!tp.has_raw_data() && "string TensorProto cannot be raw data");
  return createDenseElmAttrFromProtoData<llvm::StringRef>(
      tensorType, tp.string_data());
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
  switch (tp.data_type()) {
  case (onnx::TensorProto::FLOAT16):
    return createDenseElmAttr<float_16>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::BFLOAT16):
    return createDenseElmAttr<bfloat_16>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::FLOAT):
    return createDenseElmAttr<float>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::DOUBLE):
    return createDenseElmAttr<double>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::INT8):
    return createDenseElmAttr<int8_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::UINT8):
    return createDenseElmAttr<uint8_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::INT16):
    return createDenseElmAttr<int16_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::UINT16):
    return createDenseElmAttr<uint16_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::INT32):
    return createDenseElmAttr<int32_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::UINT32):
    return createDenseElmAttr<uint32_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::INT64):
    return createDenseElmAttr<int64_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::UINT64):
    return createDenseElmAttr<uint64_t>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::BOOL):
    return createDenseElmAttr<bool>(tensorType, tp, externalDataDir);
  case (onnx::TensorProto::STRING):
    return createDenseStringElmAttr(tensorType, tp);
  default:
    llvm_unreachable(
        "Failed to import ONNX TensorProto due to unsupported data types.");
  }
}

} // namespace onnx_mlir
