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
#include "src/Dialect/ONNX/ElementsAttr/Arrays.hpp"
#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/FloatingPoint16.hpp"

// TODO: put everything in namespace onnx_mlir, and be using namespace mlir

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

template <typename T, typename Range, typename Transformation>
mlir::ElementsAttr createElmAttrFromArray(mlir::RankedTensorType tensorType,
    const Range &array, const Transformation &transformation) {
  mlir::MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == onnx_mlir::toMlirType<T>(ctx));
  return onnx_mlir::OnnxElementsAttrBuilder(ctx).fromArray<T>(
      tensorType, [array, &transformation](llvm::MutableArrayRef<T> copy) {
        std::transform(array.begin(), array.end(), copy.data(), transformation);
      });
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

template <typename T>
mlir::ElementsAttr createElementsAttrFromMemoryBuffer_LE(
    mlir::RankedTensorType tensorType,
    std::unique_ptr<llvm::MemoryBuffer> membuf) {
  mlir::MLIRContext *ctx = tensorType.getContext();
  assert(tensorType.getElementType() == onnx_mlir::toMlirType<T>(ctx));
  if (shouldSwapLEBytes<T>) {
    llvm::ArrayRef<T> array = onnx_mlir::asArrayRef<T>(membuf->getBuffer());
    return createElmAttrFromArray<T>(tensorType, array, swappedBytes<T>);
  } else {
    return onnx_mlir::OnnxElementsAttrBuilder(ctx).fromMemoryBuffer(
        tensorType, std::move(membuf));
  }
}

template <typename T>
mlir::ElementsAttr createElmAttrFromRawBytes_LE(
    mlir::RankedTensorType tensorType, llvm::ArrayRef<char> bytes) {
  llvm::ArrayRef<T> array = onnx_mlir::castArrayRef<T>(bytes);
  return createElmAttrFromArray<T>(tensorType, array,
      [](T x) { return shouldSwapLEBytes<T> ? swappedBytes<T>(x) : x; });
}

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

template <typename T, typename U>
mlir::ElementsAttr createElmAttrFromProtoData(mlir::RankedTensorType tensorType,
    const google::protobuf::RepeatedField<U> &data) {
  // "Deserialize" the data to the correct bitwidth.
  return createElmAttrFromArray<T>(tensorType, data, deserializeDatum<T, U>);
}

// Returns ElementsAttr with tp's data.
template <typename T>
mlir::ElementsAttr createElmAttr(mlir::RankedTensorType tensorType,
    const onnx::TensorProto &tp, const std::string &externalDataDir) {
  if (tp.has_data_location() &&
      tp.data_location() == onnx::TensorProto::EXTERNAL) {
    return createElementsAttrFromMemoryBuffer_LE<T>(
        tensorType, readExternalData_LE(externalDataDir, tp));
  }
  if (tp.has_raw_data()) {
    return createElmAttrFromRawBytes_LE<T>(
        tensorType, onnx_mlir::asArrayRef(tp.raw_data()));
  }
  // Not raw, no need to take care of endianness.
  const auto &data = TransformValueToONNXData<T>::data(tp);
  return createElmAttrFromProtoData<T>(tensorType, data);
}

mlir::ElementsAttr createStringElmAttr(
    mlir::RankedTensorType tensorType, const onnx::TensorProto &tp) {
  // The string type is different from other data types in that it cannot be
  // raw or external data, it cannot be represented as a DisposableElementsAttr,
  // and it needs to be converted to StringRef (or StringAttr) to construct a
  // DenseElementsAttr.
  assert(!(tp.has_data_location() &&
             tp.data_location() == onnx::TensorProto::EXTERNAL) &&
         "string TensorProto cannot be external data");
  assert(!tp.has_raw_data() && "string TensorProto cannot be raw data");
  auto data = tp.string_data();
  llvm::SmallVector<llvm::StringRef> copy(data.begin(), data.end());
  return mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(copy));
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
    return builder.create<mlir::ONNXNoneOp>(loc);

  mlir::ElementsAttr elmAttr =
      onnxTensorProtoToElmAttr(builder, externalDataDir, initializer);
  return builder.create<mlir::ONNXConstantOp>(loc, nullptr, elmAttr);
}

mlir::ElementsAttr onnxTensorProtoToElmAttr(mlir::OpBuilder &builder,
    const std::string &externalDataDir, const onnx::TensorProto &tp) {
  // Tensor dimensions.
  llvm::ArrayRef<int64_t> tensorDims(tp.dims().data(), tp.dims().size());
  if (tp.data_type() == onnx::TensorProto::STRING) {
    mlir::Type elmType = mlir::ONNXStringType::get(builder.getContext());
    auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
    return createStringElmAttr(tensorType, tp);
  }
  BType btype = btypeOfOnnxDataType(tp.data_type());
  mlir::Type elmType = mlirTypeOfBType(btype, builder.getContext());
  auto tensorType = mlir::RankedTensorType::get(tensorDims, elmType);
  return dispatchByBType(btype, [&](auto btype) {
    using cpptype = CppType<btype>;
    return createElmAttr<cpptype>(tensorType, tp, externalDataDir);
  });
}

} // namespace onnx_mlir
