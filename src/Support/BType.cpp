/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- BType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/BType.hpp"

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace onnx_mlir {

BType btypeOfMlirType(Type type) {
  // clang-format off
  if (type.isa<mlir::Float64Type>())  return BType::DOUBLE;
  if (type.isa<mlir::Float32Type>())  return BType::FLOAT;
  if (type.isa<mlir::Float16Type>())  return BType::FLOAT16;
  if (type.isa<mlir::BFloat16Type>()) return BType::BFLOAT16;
  auto itype = type.cast<mlir::IntegerType>();
  switch (itype.getWidth()) {
    case  1: return BType::BOOL;
    case  8: return itype.isUnsigned() ? BType::UINT8  : BType::INT8;
    case 16: return itype.isUnsigned() ? BType::UINT16 : BType::INT16;
    case 32: return itype.isUnsigned() ? BType::UINT32 : BType::INT32;
    case 64: return itype.isUnsigned() ? BType::UINT64 : BType::INT64;
  }
  llvm_unreachable("unsupported int or float type");
  // clang-format on
}

Type mlirTypeOfBType(BType btype, MLIRContext *ctx) {
  constexpr bool isUnsigned = false;
  Builder b(ctx);
  // clang-format off
  switch (btype) {
    case BType::BOOL     : return b.getI1Type();
    case BType::INT8     : return b.getIntegerType(8);
    case BType::UINT8    : return b.getIntegerType(8, isUnsigned);
    case BType::INT16    : return b.getIntegerType(16);
    case BType::UINT16   : return b.getIntegerType(16, isUnsigned);
    case BType::INT32    : return b.getIntegerType(32);
    case BType::UINT32   : return b.getIntegerType(32, isUnsigned);
    case BType::INT64    : return b.getIntegerType(64);
    case BType::UINT64   : return b.getIntegerType(64, isUnsigned);
    case BType::DOUBLE   : return b.getF64Type();
    case BType::FLOAT    : return b.getF32Type();
    case BType::FLOAT16  : return b.getF16Type();
    case BType::BFLOAT16 : return b.getBF16Type();
    default: llvm_unreachable("unsupported data type");
  }
  // clang-format on
}

bool isFloatBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isFloat; });
}

bool isIntOrFloatBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isIntOrFloat; });
}

bool isSignedIntBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isSignedInt; });
}

bool isUnsignedIntBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::isUnsignedInt; });
}

unsigned bitwidthOfBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::bitwidth; });
}

unsigned bytewidthOfBType(BType d) {
  return dispatchByBType(
      d, [](auto btype) { return BTypeTrait<btype>::bytewidth; });
}

BType wideBTypeOfBType(BType d) {
  return dispatchByBType(d,
      [](auto btype) { return toBType<typename BTypeTrait<btype>::widetype>; });
}

} // namespace onnx_mlir