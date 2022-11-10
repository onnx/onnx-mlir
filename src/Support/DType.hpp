/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.hpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/FloatingPoint16.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

namespace onnx_mlir {

// Numerical representation of basic data types.
//
// DType faithfully copies onnx::TensorProto_DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and DType and onnx::TensorProto_DataType can be used interchangeably.
// In some places it is convenient to use DType to avoid compile time
// dependencies on third_party/onnx.
//
// TODO: rename DType because of a name clash in uses of onnx-mlir
enum class DType : int8_t {
  // clang-format off
  UNDEFINED = 0,
  // Basic types.
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16 = 10,

  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,     // complex with float32 real and imaginary components
  COMPLEX128 = 15,    // complex with float64 real and imaginary components

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16 = 16,
  // clang-format on

  MAX_DTYPE = 16 // TODO: update this if more types are added to the enum
};

// DType and enum onnx::TensorProto_DataType convert to each other with
// static_cast because DType faithfully copies onnx::TensorProto_DataType.
// The conversion functions onnxDataTypeOfDType and dtypeOfOnnxDataType pass
// onnx::TensorProto_DataType values as int in line with the C++ protobuf API
// in #include "onnx/onnx_pb.h".

// Returns a value from enum onnx::TensorProto_DataType.
constexpr int onnxDataTypeOfDType(DType dtype) {
  return static_cast<int>(dtype);
}
// Precondition: onnxDataType must be from enum onnx::TensorProto_DataType.
constexpr DType dtypeOfOnnxDataType(int onnxDataType) {
  return static_cast<DType>(onnxDataType);
}

namespace detail {
template <DType DTYPE, typename CPPTY>
struct DTypeTraitBase {
  static constexpr DType dtype = DTYPE;
  static constexpr bool isFloat =
      std::is_floating_point_v<CPPTY> || isFP16Type<CPPTY>;
  static constexpr bool isIntOrFloat = std::is_integral_v<CPPTY> || isFloat;
  static constexpr bool isSignedInt =
      std::is_integral_v<CPPTY> && std::is_signed_v<CPPTY>;
  static constexpr bool isUnsignedInt =
      std::is_integral_v<CPPTY> && !std::is_signed_v<CPPTY>;
  static constexpr unsigned bitwidth =
      std::is_same_v<CPPTY, bool> ? 1 : (8 * sizeof(CPPTY));
  static constexpr unsigned bytewidth = (bitwidth + 7) / 8;
  using cpptype = CPPTY;
  using widetype = std::conditional_t<isFloat, double,
      std::conditional_t<isSignedInt, int64_t, uint64_t>>;
};
} // namespace detail

template <DType DTYPE>
struct DTypeTrait : public detail::DTypeTraitBase<DTYPE, void> {};

template <typename CPPTY>
struct CppTypeTrait : public detail::DTypeTraitBase<DType::UNDEFINED, CPPTY> {};

#define DEFINE_DTypeCppTypeTraits(DTYPE, CPPTY)                                \
  template <>                                                                  \
  struct DTypeTrait<DTYPE> : public detail::DTypeTraitBase<DTYPE, CPPTY> {};   \
  template <>                                                                  \
  struct CppTypeTrait<CPPTY> : public DTypeTrait<DTYPE> {};

DEFINE_DTypeCppTypeTraits(DType::BOOL, bool);
DEFINE_DTypeCppTypeTraits(DType::INT8, int8_t);
DEFINE_DTypeCppTypeTraits(DType::UINT8, uint8_t);
DEFINE_DTypeCppTypeTraits(DType::INT16, int16_t);
DEFINE_DTypeCppTypeTraits(DType::UINT16, uint16_t);
DEFINE_DTypeCppTypeTraits(DType::INT32, int32_t);
DEFINE_DTypeCppTypeTraits(DType::UINT32, uint32_t);
DEFINE_DTypeCppTypeTraits(DType::INT64, int64_t);
DEFINE_DTypeCppTypeTraits(DType::UINT64, uint64_t);
DEFINE_DTypeCppTypeTraits(DType::DOUBLE, double);
DEFINE_DTypeCppTypeTraits(DType::FLOAT, float);
DEFINE_DTypeCppTypeTraits(DType::FLOAT16, float_16);
DEFINE_DTypeCppTypeTraits(DType::BFLOAT16, bfloat_16);

#undef DEFINE_DTypeCppTypeTraits

// Compile time mapping from DType to cpp type.
template <DType DTYPE>
using CppType = typename DTypeTrait<DTYPE>::cpptype;

// Compile time mapping from cpp type to DType. It is "compile time" because
// it's a constexpr which can be used in template arguments like
//
//   using T = DTypeTrait<toDType<cpptype>>::widetype;
//
// and in constexpr expressions like
//
//   constexpr WideNum n = WideNum::from(toDType<cpptype>, true);
//
// Note: decay_t strips reference, const, and volatile qualifiers,
// otherwise e.g. toDType<decltype(x)> easily fails because decltype
// picks up these qualifiers in ways that are easy to overlook.
template <typename CPPTY>
constexpr DType toDType = CppTypeTrait<std::decay_t<CPPTY>>::dtype;

// Runtime mapping from mlir type to DType.
DType dtypeOfMlirType(mlir::Type type);

// Runtime mapping from DType to mlir type.
mlir::Type mlirTypeOfDType(DType dtype, mlir::MLIRContext *ctx);

// Runtime mapping from cpp type to mlir type.
template <typename CPPTY>
mlir::Type toMlirType(mlir::MLIRContext *ctx) {
  return mlirTypeOfDType(toDType<CPPTY>, ctx);
}

// The following functions isFloatDType(dtype), bitwidthOfDType(dtype), etc are
// helpful alternatives to DTypeTrait<dtype>::isFloat/bitwidth/etc
// when dtype isn't known at compile.
//
// TODO: Fix them up so they don't crash with
//       llvm_unreachable("not a supported datatype")
//       when called with DType::STRING or DType::COMPLEX64/128.

// == mlirTypeOfDType(dtype, ctx).isa<FloatType>()
bool isFloatDType(DType);

// == mlirTypeOfDType(dtype, ctx).isIntOrFloat()
bool isIntOrFloatDType(DType);

// == mlirTypeOfDType(dtype, ctx).isSignlessInteger()
bool isSignedIntDType(DType);

// == mlirTypeOfDType(dtype, ctx).isUnsignedInteger()
bool isUnsignedIntDType(DType);

// == mlirTypeOfDType(dtype, ctx).getIntOrFloatBitWidth()
unsigned bitwidthOfDType(DType);

// == getIntOrFloatByteWidth(mlirTypeOfDType(dtype, ctx))
unsigned bytewidthOfDType(DType);

// == toDType<DTypeTrait<dtype>::widetype> if dtype is constexpr
DType wideDTypeOfDType(DType dtype);

template <DType DTYPE>
struct DTypeToken {
  constexpr DTypeToken() {}
  constexpr operator DType() const { return DTYPE; }
};

template <typename Action>
auto dispatchByDType(DType dtype, Action &&act) {
#define ACT(DTYPE) act(DTypeToken<DTYPE>{})
  // clang-format off
  switch (dtype) {
  case DType::BOOL     : return ACT(DType::BOOL);
  case DType::INT8     : return ACT(DType::INT8);
  case DType::UINT8    : return ACT(DType::UINT8);
  case DType::INT16    : return ACT(DType::INT16);
  case DType::UINT16   : return ACT(DType::UINT16);
  case DType::INT32    : return ACT(DType::INT32);
  case DType::UINT32   : return ACT(DType::UINT32);
  case DType::INT64    : return ACT(DType::INT64);
  case DType::UINT64   : return ACT(DType::UINT64);
  case DType::DOUBLE   : return ACT(DType::DOUBLE);
  case DType::FLOAT    : return ACT(DType::FLOAT);
  case DType::FLOAT16  : return ACT(DType::FLOAT16);
  case DType::BFLOAT16 : return ACT(DType::BFLOAT16);
  default: llvm_unreachable("not a supported datatype");
  }
  // clang-format on
#undef ACT
}

template <typename Action>
auto dispatchByMlirType(mlir::Type type, Action &&act) {
  return dispatchByDType(dtypeOfMlirType(type), std::forward<Action>(act));
}

} // namespace onnx_mlir