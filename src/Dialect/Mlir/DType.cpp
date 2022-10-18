/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- DType.cpp ------------------------------===//
//
// Basic data types.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/DType.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using llvm::APFloat;
using llvm::APInt;

namespace onnx_mlir {

float U16ToF32(uint16_t u) {
  APFloat a(APFloat::IEEEhalf(), APInt(16, u));
  return a.convertToFloat();
}

uint16_t F32ToU16(float f) {
  APFloat a(f);
  bool ignored;
  a.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &ignored);
  return a.bitcastToAPInt().getZExtValue();
}

} // namespace onnx_mlir