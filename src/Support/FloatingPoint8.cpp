/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- FloatingPoint8.cpp -------------------------===//
//
// 8 bit floating point types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/FloatingPoint8.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

namespace {
uint64_t bitcastAPFloat(APFloat f, const llvm::fltSemantics &semantics) {
  bool ignored;
  f.convert(semantics, APFloat::rmNearestTiesToEven, &ignored);
  APInt i = f.bitcastToAPInt();
  return i.getZExtValue();
}
} // namespace

namespace detail {

template <typename FP8>
APFloat FP8Base<FP8>::toAPFloat() const {
  return APFloat(FP8::semantics(), APInt(8, u8));
}

/*static*/
template <typename FP8>
FP8 FP8Base<FP8>::fromAPFloat(APFloat a) {
  bitcasttype u8 = bitcastAPFloat(a, FP8::semantics());
  return bitcastFromU8(u8);
}

// Explicit instantiation for all the classes derived from FP8Base.
template class FP8Base<float_8e4m3fn>;
template class FP8Base<float_8e4m3fnuz>;
template class FP8Base<float_8e5m2>;
template class FP8Base<float_8e5m2fnuz>;

} // namespace detail

} // namespace onnx_mlir