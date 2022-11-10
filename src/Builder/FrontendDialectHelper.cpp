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
#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"

// TODO: put everything in namespace onnx_mlir, and add 'using namespace mlir'

namespace {

// Perform byte swap if system endianness is BE.
// ONNX tensor content raw data is always in LE.
// Don't byte swap single byte types, because that's unnecessary
// and llvm::sys::getSwappedBytes(bool) also happens to be broken.
template <typename T>
constexpr bool shouldSwapLEBytes =
    sizeof(T) > 1 && llvm::support::endian::system_endianness() !=
                         llvm::support::endianness::little;

#ifndef DISABLE_DISPOSABLE_POOL
// TODO: make this work...
struct ElementsAttrFactory {
  template <typename T>
  static mlir::ElementsAttr get(
      mlir::RankedTensorType type, llvm::ArrayRef<T> data) {
    mlir::MLIRContext *ctx = type.getContext();
    assert(type.getElementType() == onnx_mlir::toMlirType<T>(ctx));
    return onnx_mlir::ElementsAttrBuilder(ctx).fromArray(
        type, data, /*mustCopy=*/true);
  }
};
template <typename T>
mlir::ElementsAttr createElementsAttrFromMemoryBuffer(
    mlir::RankedTensorType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  mlir::MLIRContext *ctx = type.getContext();
  assert(type.getElementType() == onnx_mlir::toMlirType<T>(ctx));
  if (shouldSwapLEBytes<T>) {
    // TODO: swap bytes into MemoryBuffer and create ElementsAttr from that
  } else {
    return onnx_mlir::ElementsAttrBuilder(ctx).create(type, std::move(membuf));
  }
}
#else
using ElementsAttrFactory = mlir::DenseElementsAttr;
template <typename T>
mlir::ElementsAttr createDenseElmAttrFromRawData(
    llvm::ArrayRef<char> buffer, mlir::RankedTensorType tensorType);
template <typename T>
mlir::ElementsAttr createElementsAttrFromMemoryBuffer(
    mlir::RankedTensorType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  assert(type.getElementType() == onnx_mlir::toMlirType<T>(type.getContext()));
  llvm::ArrayRef<char> bytes = onnx_mlir::asArrayRef(membuf->getBuffer());
  return shouldSwapLEBytes<T> ? createDenseElmAttrFromRawData<T>(bytes, type)
                              : ElementsAttrFactory::get(
                                    type, onnx_mlir::castArrayRef<T>(bytes));
}
#endif

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

// Converts to the cpp type 'To' that correspond's to the tensor element type
// (bool, int8, float_16, uint32, etc) from the the proto data field type
// which may be a wider type (int32, uint64). In most cases the conversion is
// just standard C implicit conversion. The exception is float_16 and bfloat_16
// which must be bit-wise converted from uint16_t.
template <typename To, typename From>
To deserializeDatum(From from) {
  if constexpr (onnx_mlir::isFP16Type<To>)
    return To::bitcastFromU16(from);
  else
    return from;
}

template <typename T, typename U>
mlir::ElementsAttr createDenseElmAttrFromProtoData(
    const google::protobuf::RepeatedField<U> &data,
    mlir::RankedTensorType tensorType) {
  if constexpr (std::is_same_v<T, U>) {
    // When the protobuf repeated field has a type of the same size as T,
    // access the data directly via ArrayRef.
    return ElementsAttrFactory::get(
        tensorType, llvm::makeArrayRef(data.data(), data.size()));
  }
  // Copy the data into correctly typed SmallVector because
  // DenseElementsAttr needs argument type of the correct bitwidth.
  llvm::SmallVector<T> copy;
  copy.resize_for_overwrite(data.size());
  std::transform(data.begin(), data.end(), copy.data(), deserializeDatum<T, U>);
  return ElementsAttrFactory::get(tensorType, llvm::makeArrayRef(copy));
}

// Extension of llvm::sys::getSwappedBytes to also handle float_16, bfloat_16.
template <typename T>
T swappedBytes(T x) {
  if constexpr (onnx_mlir::isFP16Type<T>)
    return T::bitcastFromU16(llvm::sys::getSwappedBytes(x.bitcastToU16()));
  else
    return llvm::sys::getSwappedBytes(x);
}

template <typename T>
mlir::ElementsAttr createDenseElmAttrFromRawData(
    llvm::ArrayRef<char> buffer, mlir::RankedTensorType tensorType) {
  llvm::ArrayRef<T> array = onnx_mlir::castArrayRef<T>(buffer);
  if (shouldSwapLEBytes<T>) {
    llvm::SmallVector<T> copy;
    copy.resize_for_overwrite(array.size());
    std::transform(array.begin(), array.end(), copy.data(), swappedBytes<T>);
    return ElementsAttrFactory::get(tensorType, llvm::makeArrayRef(copy));
  } else {
    // No need to take care of endianness.
    return ElementsAttrFactory::get(tensorType, array);
  }
}

// Returns ElementsAttr with tp's data.
template <typename T>
mlir::ElementsAttr createDenseElmAttr(const std::string &externalDataDir,
    const onnx::TensorProto &tp, mlir::RankedTensorType tensorType) {
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    if (std::unique_ptr<llvm::MemoryBuffer> externalData =
            readExternalData(externalDataDir, tp))
      return createElementsAttrFromMemoryBuffer<T>(
          tensorType, std::move(externalData));
    // TODO: llvm_unreachable ?
  }
  if (tp.has_raw_data()) {
    return createDenseElmAttrFromRawData<T>(
        onnx_mlir::asArrayRef(tp.raw_data()), tensorType);
  }
  // Not raw, no need to take care of endianness.
  const auto &data = TransformValueToONNXData<T>::data(tp);
  return createDenseElmAttrFromProtoData<T>(data, tensorType);
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

  mlir::ElementsAttr denseElmAttr =
      onnxTensorProtoToDenseElmAttr(builder, externalDataDir, initializer);
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, denseElmAttr);
}

mlir::ElementsAttr onnxTensorProtoToDenseElmAttr(mlir::OpBuilder &builder,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  DType dtype = dtypeOfOnnxDataType(tp.data_type());
  mlir::Type elmType = mlirTypeOfDType(dtype, builder.getContext());
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  return dispatchByDType(dtype, [&](auto dtype) {
    using cpptype = CppType<dtype>;
    return createDenseElmAttr<cpptype>(externalDataDir, tp, tensorType);
  });
}

} // namespace onnx_mlir
