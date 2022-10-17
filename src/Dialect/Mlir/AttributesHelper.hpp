/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"

namespace onnx_mlir {

mlir::ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, size_t align = 1);

template <typename NumericType>
mlir::ElementsAttr makeDenseIntOrFPElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<NumericType> numbers) {
  constexpr size_t BYTE_BITWIDTH = 8;
  size_t w = type.getElementType().getIntOrFloatBitWidth();
  (void)w;
  assert(w == sizeof(NumericType) * (w == 1 ? 1 : BYTE_BITWIDTH) &&
         "static template type must match dynamic argument type");
  llvm::ArrayRef<char> bytes(reinterpret_cast<const char *>(numbers.data()),
      numbers.size() * sizeof(NumericType));
  return makeDenseIntOrFPElementsAttrFromRawBuffer(
      type, bytes, alignof(NumericType));
}

llvm::ArrayRef<char> getDenseIntOrFPRawData(mlir::ElementsAttr elements);

} // namespace onnx_mlir