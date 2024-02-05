/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- WideNum.cpp -----------------------------===//
//
// WideNum data type.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

APFloat WideNum::toAPFloat(BType tag) const {
  switch (tag) {
  case BType::DOUBLE:
    return APFloat(dbl);
  case BType::FLOAT:
    return APFloat(static_cast<float>(dbl));
  case BType::FLOAT16:
    return float_16(dbl).toAPFloat();
  case BType::BFLOAT16:
    return bfloat_16(dbl).toAPFloat();
  case BType::FLOAT8E4M3FN:
    return float_8e4m3fn(dbl).toAPFloat();
  case BType::FLOAT8E4M3FNUZ:
    return float_8e4m3fnuz(dbl).toAPFloat();
  case BType::FLOAT8E5M2:
    return float_8e5m2(dbl).toAPFloat();
  case BType::FLOAT8E5M2FNUZ:
    return float_8e5m2fnuz(dbl).toAPFloat();
  default:
    llvm_unreachable("BType must be a float");
  }
}

APInt WideNum::toAPInt(BType tag) const {
  unsigned bitwidth = bitwidthOfBType(tag);
  if (isSignedIntBType(tag))
    // Actually, isSigned flag is ignored because bitwidth <= 64.
    return APInt(bitwidth, i64, /*isSigned=*/true);
  if (isUnsignedIntBType(tag))
    return APInt(bitwidth, u64);
  llvm_unreachable("BType must be an integer");
}

/*static*/
WideNum WideNum::fromAPFloat(APFloat x) {
  return WideNum(x.convertToDouble()); // .dbl
}

/*static*/
WideNum WideNum::fromAPInt(APInt x, bool isSigned) {
  if (isSigned)
    return WideNum(x.getSExtValue()); // .i64
  else
    return WideNum(x.getZExtValue()); // .u64
}

} // namespace onnx_mlir
