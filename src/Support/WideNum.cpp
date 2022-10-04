/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- WideNum.cpp -----------------------------===//
//
// WideNum data type.
//
//===----------------------------------------------------------------------===//

#include "src/Support/WideNum.hpp"

#include "src/Support/Arrays.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

using llvm::APFloat;
using llvm::APInt;
using llvm::ArrayRef;
using llvm::MutableArrayRef;

namespace onnx_mlir {

APFloat WideNum::toAPFloat(DType tag) const {
  switch (tag) {
  case DType::DOUBLE:
    return APFloat(dbl);
  case DType::FLOAT:
    return APFloat(static_cast<float>(dbl));
  case DType::FLOAT16:
    return float_16(dbl).toAPFloat();
  case DType::BFLOAT16:
    return bfloat_16(dbl).toAPFloat();
  default:
    llvm_unreachable("DType must be a float");
  }
}

APInt WideNum::toAPInt(DType tag) const {
  unsigned bitwidth = bitwidthOfDType(tag);
  if (isSignedIntDType(tag))
    // Actually, isSigned flag is ignored because bitwidth <= 64.
    return APInt(bitwidth, i64, /*isSigned=*/true);
  if (isUnsignedIntDType(tag))
    return APInt(bitwidth, u64);
  llvm_unreachable("DType must be an integer");
}

/*static*/
WideNum WideNum::fromAPFloat(DType tag, APFloat x) {
  assert(isFloatDType(tag) && "DType must be an integer");
  return {.dbl = x.convertToDouble()};
}

/*static*/
WideNum WideNum::fromAPInt(DType tag, APInt x) {
  if (isSignedIntDType(tag))
    return {.i64 = x.getSExtValue()};
  if (isUnsignedIntDType(tag))
    return {.u64 = x.getZExtValue()};
  llvm_unreachable("DType must be an integer");
}

void WideNum::store(DType dtag, MutableArrayRef<char> memory) const {
  dispatchByDType(dtag, [memory, this](auto dtype) {
    using X = CppType<dtype>;
    assert(memory.size() == sizeof(X));
    *castMutableArrayRef<X>(memory).begin() = this->to<X>(dtype);
  });
}

/*static*/
WideNum WideNum::load(DType dtag, ArrayRef<char> memory) {
  return dispatchByDType(dtag, [memory](auto dtype) {
    using X = CppType<dtype>;
    assert(memory.size() == sizeof(X));
    return from<X>(dtype, *castArrayRef<X>(memory).begin());
  });
}

} // namespace onnx_mlir