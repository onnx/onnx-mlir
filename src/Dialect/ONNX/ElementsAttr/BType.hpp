/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- BType.hpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_B_TYPE_H
#define ONNX_MLIR_B_TYPE_H

#include "src/Support/SmallFP.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

#include <type_traits>

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

  // Non-IEEE floating-point format based on papers
  // FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
  // 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
  // Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
  // The computation usually happens inside a block quantize / dequantize
  // fused by the runtime.
  FLOAT8E4M3FN = 17,    // float 8, mostly used for coefficients, supports nan, not inf
  FLOAT8E4M3FNUZ = 18,  // float 8, mostly used for coefficients, supports nan, not inf, no negative zero
  FLOAT8E5M2 = 19,      // follows IEEE 754, supports nan, inf, mostly used for gradients
  FLOAT8E5M2FNUZ = 20,  // follows IEEE 754, supports nan, inf, mostly used for gradients, no negative zero
  // clang-format on

  MAX_BTYPE = 20 // TODO: update this if more types are added to the enum
};

constexpr int kNumBTypes = static_cast<int8_t>(BType::MAX_BTYPE) + 1;

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
template <typename CPPTY>
constexpr unsigned bytewidthOf() {
  return sizeof(CPPTY);
}
template <>
constexpr unsigned bytewidthOf<void>() {
  return 0; // sizeof(void) is illegal with MSVC
}

template <BType BTYPE, typename CPPTY>
struct BTypeTraitBase {
  static constexpr BType btype = BTYPE;
  static constexpr bool isFloat =
      std::is_floating_point_v<CPPTY> || isSmallFPType<CPPTY>;
  static constexpr bool isInt = std::is_integral_v<CPPTY>;
  static constexpr bool isIntOrFloat = isInt || isFloat;
  static constexpr bool isSignedInt = isInt && std::is_signed_v<CPPTY>;
  static constexpr bool isUnsignedInt = isInt && !std::is_signed_v<CPPTY>;
  static constexpr unsigned bytewidth = bytewidthOf<CPPTY>();
  static constexpr unsigned bitwidth =
      std::is_same_v<CPPTY, bool> ? 1 : (CHAR_BIT * bytewidth);
  using cpptype = CPPTY;
  using widetype = std::conditional_t<isFloat, double,
      std::conditional_t<isSignedInt, int64_t, uint64_t>>;
};
} // namespace detail

template <BType BTYPE>
struct BTypeTrait : public detail::BTypeTraitBase<BTYPE, void> {};

template <typename CPPTY>
struct CppTypeTrait : public detail::BTypeTraitBase<BType::UNDEFINED, CPPTY> {};

#define DEFINE_BTypeCppTypeTraits(BTYPE, CPPTY)                                \
  template <>                                                                  \
  struct BTypeTrait<BTYPE> : public detail::BTypeTraitBase<BTYPE, CPPTY> {};   \
  template <>                                                                  \
  struct CppTypeTrait<CPPTY> : public BTypeTrait<BTYPE> {}

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
DEFINE_BTypeCppTypeTraits(BType::FLOAT8E4M3FN, float_8e4m3fn);
DEFINE_BTypeCppTypeTraits(BType::FLOAT8E4M3FNUZ, float_8e4m3fnuz);
DEFINE_BTypeCppTypeTraits(BType::FLOAT8E5M2, float_8e5m2);
DEFINE_BTypeCppTypeTraits(BType::FLOAT8E5M2FNUZ, float_8e5m2fnuz);

#undef DEFINE_BTypeCppTypeTraits

// Compile time mapping from BType to cpp type.
template <BType BTYPE>
using CppType = typename BTypeTrait<BTYPE>::cpptype;

// Compile time mapping from BType to wide type.
template <BType BTYPE>
using WideType = typename BTypeTrait<BTYPE>::widetype;

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

// == mlir::isa<FloatType>(mlirTypeOfBType(btype, ctx))
bool isFloatBType(BType);

// == mlir::isa<IntegerType>(mlirTypeOfBType(btype, ctx))
bool isIntBType(BType);

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

template <BType BTYPE>
using BTypeConstant = std::integral_constant<BType, BTYPE>;

// If a is a BType runtime value and expr(btype) is an expression that
// uses btype as a constexpr, e.g. expr(btype) = sizeof(CppType<btype>), then
//
//   r = dispatchByBType(a, [&](auto btype) { return expr(btype); })
//
// is shorthand for:
//
//   switch (a) {
//   case BType::BOOL: { constexpr BType btype = BType::BOOL; r = expr(btype); }
//   case BType::INT8: { constexpr BType btype = BType::INT8; r = expr(btype); }
//   // etc for all the other int and float BType values
//   default: llvm_unreachable("not a supported datatype");
//   }
//
// So we can write:
//
//   unsigned sizeofBType(BType a) {
//     return dispatchByBType(
//         a, [](auto btype) { return sizeof(CppType<btype>); });
//   }
//
template <typename Action>
auto dispatchByBType(BType btype, Action &&act) {
#define ACT(BTYPE) act(BTypeConstant<BTYPE>{})
  // clang-format off
  switch (btype) {
  case BType::BOOL           : return ACT(BType::BOOL);
  case BType::INT8           : return ACT(BType::INT8);
  case BType::UINT8          : return ACT(BType::UINT8);
  case BType::INT16          : return ACT(BType::INT16);
  case BType::UINT16         : return ACT(BType::UINT16);
  case BType::INT32          : return ACT(BType::INT32);
  case BType::UINT32         : return ACT(BType::UINT32);
  case BType::INT64          : return ACT(BType::INT64);
  case BType::UINT64         : return ACT(BType::UINT64);
  case BType::DOUBLE         : return ACT(BType::DOUBLE);
  case BType::FLOAT          : return ACT(BType::FLOAT);
  case BType::FLOAT16        : return ACT(BType::FLOAT16);
  case BType::BFLOAT16       : return ACT(BType::BFLOAT16);
  case BType::FLOAT8E4M3FN   : return ACT(BType::FLOAT8E4M3FN);
  case BType::FLOAT8E4M3FNUZ : return ACT(BType::FLOAT8E4M3FNUZ);
  case BType::FLOAT8E5M2     : return ACT(BType::FLOAT8E5M2);
  case BType::FLOAT8E5M2FNUZ : return ACT(BType::FLOAT8E5M2FNUZ);
  default: llvm_unreachable("not a supported datatype");
  }
  // clang-format on
#undef ACT
}

} // namespace onnx_mlir
#endif