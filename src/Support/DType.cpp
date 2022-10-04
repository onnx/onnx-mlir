/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/DType.hpp"

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace onnx_mlir {

DType dtypeOfMlirType(Type type) {
  // clang-format off
  if (type.isa<mlir::Float64Type>())  return DType::DOUBLE;
  if (type.isa<mlir::Float32Type>())  return DType::FLOAT;
  if (type.isa<mlir::Float16Type>())  return DType::FLOAT16;
  if (type.isa<mlir::BFloat16Type>()) return DType::BFLOAT16;
  auto itype = type.cast<mlir::IntegerType>();
  switch (itype.getWidth()) {
    case  1: return DType::BOOL;
    case  8: return itype.isUnsigned() ? DType::UINT8  : DType::INT8;
    case 16: return itype.isUnsigned() ? DType::UINT16 : DType::INT16;
    case 32: return itype.isUnsigned() ? DType::UINT32 : DType::INT32;
    case 64: return itype.isUnsigned() ? DType::UINT64 : DType::INT64;
  }
  llvm_unreachable("unsupported int or float type");
  // clang-format on
}

Type mlirTypeOfDType(DType dtype, MLIRContext *ctx) {
  constexpr bool isUnsigned = false;
  Builder b(ctx);
  // clang-format off
  switch (dtype) {
    case DType::BOOL     : return b.getI1Type();
    case DType::INT8     : return b.getIntegerType(8);
    case DType::UINT8    : return b.getIntegerType(8, isUnsigned);
    case DType::INT16    : return b.getIntegerType(16);
    case DType::UINT16   : return b.getIntegerType(16, isUnsigned);
    case DType::INT32    : return b.getIntegerType(32);
    case DType::UINT32   : return b.getIntegerType(32, isUnsigned);
    case DType::INT64    : return b.getIntegerType(64);
    case DType::UINT64   : return b.getIntegerType(64, isUnsigned);
    case DType::DOUBLE   : return b.getF64Type();
    case DType::FLOAT    : return b.getF32Type();
    case DType::FLOAT16  : return b.getF16Type();
    case DType::BFLOAT16 : return b.getBF16Type();
    default: llvm_unreachable("unsupported data type");
  }
  // clang-format on
}

bool isFloatDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isFloat; });
}

bool isIntOrFloatDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isIntOrFloat; });
}

bool isSignedIntDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isSignedInt; });
}

bool isUnsignedIntDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::isUnsignedInt; });
}

unsigned bitwidthOfDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::bitwidth; });
}

unsigned bytewidthOfDType(DType d) {
  return dispatchByDType(
      d, [](auto dtype) { return DTypeTrait<dtype>::bytewidth; });
}

DType wideDTypeOfDType(DType d) {
  return dispatchByDType(d,
      [](auto dtype) { return toDType<typename DTypeTrait<dtype>::widetype>; });
}

} // namespace onnx_mlir