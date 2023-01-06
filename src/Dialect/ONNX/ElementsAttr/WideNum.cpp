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
WideNum WideNum::fromAPFloat(BType tag, APFloat x) {
  assert(isFloatBType(tag) && "BType must be an integer");
  return WideNum(x.convertToDouble()); // .dbl
}

/*static*/
WideNum WideNum::fromAPInt(BType tag, APInt x) {
  if (isSignedIntBType(tag))
    return WideNum(x.getSExtValue()); // .i64
  if (isUnsignedIntBType(tag))
    return WideNum(x.getZExtValue()); // .u64
  llvm_unreachable("BType must be an integer");
}

} // namespace onnx_mlir
