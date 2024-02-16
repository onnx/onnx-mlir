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

#include "src/Builder/FrontendDialectHelper.hpp"

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SwapByteOrder.h"

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/SmallFP.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Parses unsigned number.
size_t parseOffsetOrLength(const std::string &value) {
  char *end = nullptr;
  size_t offsetOrLength = strtoull(value.c_str(), &end, 0);
  assert(end != value.c_str() && "failed to parse offset or length");
  return offsetOrLength;
}

// Reads external data from file location specified in tensor proto.
// The data is little endian encoded.
// See https://github.com/onnx/onnx/blob/main/docs/ExternalData.md
std::unique_ptr<llvm::MemoryBuffer> readExternalData_LE(
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
  SmallVector<char> path(externalDataDir.begin(), externalDataDir.end());
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
    // int32, uint8, int8, uint16, int16, bool, float_16, bfloat_16,
    // float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
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

template <typename T, typename Range, typename Transformation>
ElementsAttr createElmAttrFromArray(RankedTensorType tensorType,
    const Range &array, const Transformation &transformation) {
  MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == toMlirType<T>(ctx));
  return OnnxElementsAttrBuilder(ctx).fromArray<T>(
      tensorType, [array, &transformation](MutableArrayRef<T> copy) {
        std::transform(array.begin(), array.end(), copy.data(), transformation);
      });
}

// Perform byte swap if system endianness is BE.
// ONNX tensor content raw data is always in LE.
// Don't byte swap single byte types, because that's unnecessary
// and llvm::sys::getSwappedBytes(bool) also happens to be broken.
template <typename T>
constexpr bool
    shouldSwapLEBytes = sizeof(T) > 1 && llvm::endianness::native
                                             != llvm::endianness::little;
// Extension of llvm::sys::getSwappedBytes to also handle float_16, bfloat_16.
template <typename T>
T swappedBytes(T x) {
  if constexpr (isSmallFPType<T>)
    return T::bitcastFromUInt(llvm::sys::getSwappedBytes(x.bitcastToUInt()));
  else
    return llvm::sys::getSwappedBytes(x);
}

template <typename T>
ElementsAttr createElementsAttrFromMemoryBuffer_LE(
    RankedTensorType tensorType, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == toMlirType<T>(ctx));
  if constexpr (shouldSwapLEBytes<T>) {
    ArrayRef<T> array = asArrayRef<T>(membuf->getBuffer());
    return createElmAttrFromArray<T>(tensorType, array, swappedBytes<T>);
  } else {
    return OnnxElementsAttrBuilder(ctx).fromMemoryBuffer(
        tensorType, std::move(membuf));
  }
}

template <typename T>
ElementsAttr createElmAttrFromRawBytes_LE(
    RankedTensorType tensorType, ArrayRef<char> bytes) {
  ArrayRef<T> array = castArrayRef<T>(bytes);
  return createElmAttrFromArray<T>(tensorType, array, [](T x) {
    if constexpr (shouldSwapLEBytes<T>)
      return swappedBytes<T>(x);
    else
      return x;
  });
}

// Converts to the cpp type 'To' that correspond's to the tensor element type
// (bool, int8, float_16, uint32, etc) from the the proto data field type
// which may be a wider type (int32, uint64). In most cases the conversion is
// just standard C implicit conversion. The exception is float_16 and bfloat_16
// which must be bit-wise converted from uint16_t.
template <typename To, typename From>
To deserializeDatum(const From &from) {
  if constexpr (isSmallFPType<To>)
    return To::bitcastFromUInt(from);
  else
    return from;
}

template <typename T, typename U>
ElementsAttr createElmAttrFromProtoData(RankedTensorType tensorType,
    const google::protobuf::RepeatedField<U> &data) {
  // "Deserialize" the data to the correct bitwidth.
  return createElmAttrFromArray<T>(tensorType, data, deserializeDatum<T, U>);
}

// Returns ElementsAttr with tp's data.
template <typename T>
ElementsAttr createElmAttr(RankedTensorType tensorType,
    const onnx::TensorProto &tp, const std::string &externalDataDir) {
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    return createElementsAttrFromMemoryBuffer_LE<T>(
        tensorType, readExternalData_LE(externalDataDir, tp));
  }
  if (tp.has_raw_data()) {
    return createElmAttrFromRawBytes_LE<T>(
        tensorType, asArrayRef(tp.raw_data()));
  }
  // Not raw, no need to take care of endianness.
  const auto &data = TransformValueToONNXData<T>::data(tp);
  return createElmAttrFromProtoData<T>(tensorType, data);
}

ElementsAttr createStringElmAttr(
    RankedTensorType tensorType, const onnx::TensorProto &tp) {
  // The string type is different from other data types in that it cannot be
  // raw or external data, it cannot be represented as a DisposableElementsAttr,
  // and it needs to be converted to StringRef (or StringAttr) to construct a
  // DenseElementsAttr.
  assert(!(tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) &&
         "string TensorProto cannot be external data");
  assert(!tp.has_raw_data() && "string TensorProto cannot be raw data");
  auto data = tp.string_data();
  SmallVector<StringRef> copy(data.begin(), data.end());
  return DenseElementsAttr::get(tensorType, ArrayRef(copy));
}

} // namespace

ElementsAttr onnxTensorProtoToElmAttr(MLIRContext *ctx,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  if (tp.data_type() == onnx::TensorProto::STRING) {
    Type elmType = ONNXStringType::get(ctx);
    auto tensorType = RankedTensorType::get(tensorDims, elmType);
    return createStringElmAttr(tensorType, tp);
  }
  BType btype = btypeOfOnnxDataType(tp.data_type());
  Type elmType = mlirTypeOfBType(btype, ctx);
  auto tensorType = RankedTensorType::get(tensorDims, elmType);
  return dispatchByBType(btype, [&](auto btype) {
    using cpptype = CppType<btype>;
    return createElmAttr<cpptype>(tensorType, tp, externalDataDir);
  });
}

} // namespace onnx_mlir
