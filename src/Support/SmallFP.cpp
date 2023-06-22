/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- SmallFP.cpp -----------------------------===//
//
// 8 and 16 bits floating point types.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SmallFP.hpp"

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

template <typename FP, unsigned BITWIDTH>
APFloat SmallFPBase<FP, BITWIDTH>::toAPFloat() const {
  return APFloat(FP::semantics(), APInt(BITWIDTH, ui));
}

/*static*/
template <typename FP, unsigned BITWIDTH>
FP SmallFPBase<FP, BITWIDTH>::fromAPFloat(APFloat a) {
  bitcasttype ui = bitcastAPFloat(a, FP::semantics());
  return bitcastFromUInt(ui);
}

// Explicit instantiation for all the classes derived from SmallFPBase.
template class SmallFPBase<float_16, 16>;
template class SmallFPBase<bfloat_16, 16>;
template class SmallFPBase<float_8e4m3fn, 8>;
template class SmallFPBase<float_8e4m3fnuz, 8>;
template class SmallFPBase<float_8e5m2, 8>;
template class SmallFPBase<float_8e5m2fnuz, 8>;

} // namespace detail

} // namespace onnx_mlir