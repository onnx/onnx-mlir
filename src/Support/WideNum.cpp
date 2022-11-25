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
  // C++20: return {.dbl = x.convertToDouble()};
  WideNum w;
  w.dbl = x.convertToDouble();
  return w;
}

/*static*/
WideNum WideNum::fromAPInt(BType tag, APInt x) {
  WideNum w;
  if (isSignedIntBType(tag)) {
    // C++20: return {.i64 = x.getSExtValue()};
    w.i64 = x.getSExtValue();
  } else if (isUnsignedIntBType(tag)) {
    // C++20: return {.u64 = x.getZExtValue()};
    w.u64 = x.getZExtValue();
  } else {
    llvm_unreachable("BType must be an integer");
  }
  return w;
}

void WideNum::store(BType dtag, MutableArrayRef<char> memory) const {
  dispatchByBType(dtag, [memory, this](auto btype) {
    using X = CppType<btype>;
    assert(memory.size() == sizeof(X));
    *castMutableArrayRef<X>(memory).begin() = this->to<X>(btype);
  });
}

/*static*/
WideNum WideNum::load(BType dtag, ArrayRef<char> memory) {
  return dispatchByBType(dtag, [memory](auto btype) {
    using X = CppType<btype>;
    assert(memory.size() == sizeof(X));
    return from<X>(btype, *castArrayRef<X>(memory).begin());
  });
}

} // namespace onnx_mlir
