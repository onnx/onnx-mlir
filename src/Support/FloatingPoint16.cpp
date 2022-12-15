/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- FloatingPoint16.cpp --------------------------===//
//
// 16 bit floating point types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/FloatingPoint16.hpp"

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

template <typename FP16>
APFloat FP16Base<FP16>::toAPFloat() const {
  return APFloat(FP16::semantics(), APInt(16, u16));
}

/*static*/
template <typename FP16>
FP16 FP16Base<FP16>::fromAPFloat(APFloat a) {
  bitcasttype u16 = bitcastAPFloat(a, FP16::semantics());
  return bitcastFromU16(u16);
}

// Explicit instantiation for all the classes derived from FP16Base.
template class FP16Base<float_16>;
template class FP16Base<bfloat_16>;

} // namespace detail

} // namespace onnx_mlir