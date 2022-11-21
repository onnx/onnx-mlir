/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- BType.hpp ------------------------------===//
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
// BType faithfully copies onnx::TensorProto_DataType from
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// and BType and onnx::TensorProto_DataType can be used interchangeably.
// In some places it is convenient to use BType to avoid compile time
// dependencies on third_party/onnx.
enum class BType : int8_t {
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

// BType and enum onnx::TensorProto_DataType convert to each other with
// static_cast because BType faithfully copies onnx::TensorProto_DataType.
// The conversion functions onnxDataTypeOfBType and btypeOfOnnxDataType pass
// onnx::TensorProto_DataType values as int in line with the C++ protobuf API
// in #include "onnx/onnx_pb.h".

// Returns a value from enum onnx::TensorProto_DataType.
constexpr int onnxDataTypeOfBType(BType btype) {
  return static_cast<int>(btype);
}
// Precondition: onnxDataType must be from enum onnx::TensorProto_DataType.
constexpr BType btypeOfOnnxDataType(int onnxDataType) {
  return static_cast<BType>(onnxDataType);
}

namespace detail {
template <BType DTYPE, typename CPPTY>
struct BTypeTraitBase {
  static constexpr BType btype = DTYPE;
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

template <BType DTYPE>
struct BTypeTrait : public detail::BTypeTraitBase<DTYPE, void> {};

template <typename CPPTY>
struct CppTypeTrait : public detail::BTypeTraitBase<BType::UNDEFINED, CPPTY> {};

#define DEFINE_BTypeCppTypeTraits(DTYPE, CPPTY)                                \
  template <>                                                                  \
  struct BTypeTrait<DTYPE> : public detail::BTypeTraitBase<DTYPE, CPPTY> {};   \
  template <>                                                                  \
  struct CppTypeTrait<CPPTY> : public BTypeTrait<DTYPE> {};

DEFINE_BTypeCppTypeTraits(BType::BOOL, bool);
DEFINE_BTypeCppTypeTraits(BType::INT8, int8_t);
DEFINE_BTypeCppTypeTraits(BType::UINT8, uint8_t);
DEFINE_BTypeCppTypeTraits(BType::INT16, int16_t);
DEFINE_BTypeCppTypeTraits(BType::UINT16, uint16_t);
DEFINE_BTypeCppTypeTraits(BType::INT32, int32_t);
DEFINE_BTypeCppTypeTraits(BType::UINT32, uint32_t);
DEFINE_BTypeCppTypeTraits(BType::INT64, int64_t);
DEFINE_BTypeCppTypeTraits(BType::UINT64, uint64_t);
DEFINE_BTypeCppTypeTraits(BType::DOUBLE, double);
DEFINE_BTypeCppTypeTraits(BType::FLOAT, float);
DEFINE_BTypeCppTypeTraits(BType::FLOAT16, float_16);
DEFINE_BTypeCppTypeTraits(BType::BFLOAT16, bfloat_16);

#undef DEFINE_BTypeCppTypeTraits

// Compile time mapping from BType to cpp type.
template <BType DTYPE>
using CppType = typename BTypeTrait<DTYPE>::cpptype;

// Compile time mapping from cpp type to BType. It is "compile time" because
// it's a constexpr which can be used in template arguments like
//
//   using T = BTypeTrait<toBType<cpptype>>::widetype;
//
// and in constexpr expressions like
//
//   constexpr WideNum n = WideNum::from(toBType<cpptype>, true);
//
// Note: decay_t strips reference, const, and volatile qualifiers,
// otherwise e.g. toBType<decltype(x)> easily fails because decltype
// picks up these qualifiers in ways that are easy to overlook.
template <typename CPPTY>
constexpr BType toBType = CppTypeTrait<std::decay_t<CPPTY>>::btype;

// Runtime mapping from mlir type to BType.
BType btypeOfMlirType(mlir::Type type);

// Runtime mapping from BType to mlir type.
mlir::Type mlirTypeOfBType(BType btype, mlir::MLIRContext *ctx);

// Runtime mapping from cpp type to mlir type.
template <typename CPPTY>
mlir::Type toMlirType(mlir::MLIRContext *ctx) {
  return mlirTypeOfBType(toBType<CPPTY>, ctx);
}

// The following functions isFloatBType(btype), bitwidthOfBType(btype), etc are
// helpful alternatives to BTypeTrait<btype>::isFloat/bitwidth/etc
// when btype isn't known at compile.
//
// TODO: Fix them up so they don't crash with
//       llvm_unreachable("not a supported datatype")
//       when called with BType::STRING or BType::COMPLEX64/128.

// == mlirTypeOfBType(btype, ctx).isa<FloatType>()
bool isFloatBType(BType);

// == mlirTypeOfBType(btype, ctx).isIntOrFloat()
bool isIntOrFloatBType(BType);

// == mlirTypeOfBType(btype, ctx).isSignlessInteger()
bool isSignedIntBType(BType);

// == mlirTypeOfBType(btype, ctx).isUnsignedInteger()
bool isUnsignedIntBType(BType);

// == mlirTypeOfBType(btype, ctx).getIntOrFloatBitWidth()
unsigned bitwidthOfBType(BType);

// == getIntOrFloatByteWidth(mlirTypeOfBType(btype, ctx))
unsigned bytewidthOfBType(BType);

// == toBType<BTypeTrait<btype>::widetype> if btype is constexpr
BType wideBTypeOfBType(BType btype);

template <BType DTYPE>
struct BTypeToken {
  constexpr BTypeToken() {}
  constexpr operator BType() const { return DTYPE; }
};

template <typename Action>
auto dispatchByBType(BType btype, Action &&act) {
#define ACT(DTYPE) act(BTypeToken<DTYPE>{})
  // clang-format off
  switch (btype) {
  case BType::BOOL     : return ACT(BType::BOOL);
  case BType::INT8     : return ACT(BType::INT8);
  case BType::UINT8    : return ACT(BType::UINT8);
  case BType::INT16    : return ACT(BType::INT16);
  case BType::UINT16   : return ACT(BType::UINT16);
  case BType::INT32    : return ACT(BType::INT32);
  case BType::UINT32   : return ACT(BType::UINT32);
  case BType::INT64    : return ACT(BType::INT64);
  case BType::UINT64   : return ACT(BType::UINT64);
  case BType::DOUBLE   : return ACT(BType::DOUBLE);
  case BType::FLOAT    : return ACT(BType::FLOAT);
  case BType::FLOAT16  : return ACT(BType::FLOAT16);
  case BType::BFLOAT16 : return ACT(BType::BFLOAT16);
  default: llvm_unreachable("not a supported datatype");
  }
  // clang-format on
#undef ACT
}

template <typename Action>
auto dispatchByMlirType(mlir::Type type, Action &&act) {
  return dispatchByBType(btypeOfMlirType(type), std::forward<Action>(act));
}

} // namespace onnx_mlir