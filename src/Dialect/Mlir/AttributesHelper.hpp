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

mlir::ElementsAttr makeDenseIntOrFPElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<char> bytes, size_t align = 1);

template <typename NumericType>
mlir::ElementsAttr makeDenseIntOrFPElementsAttr(
    mlir::ShapedType type, llvm::ArrayRef<NumericType> numbers) {
  constexpr size_t BYTE_BITWIDTH = 8;
  assert(sizeof(NumericType) ==
             type.getElementType().getIntOrFloatBitWidth() * BYTE_BITWIDTH &&
         "static template type must match dynamic argument type");
  llvm::ArrayRef<char> bytes(reinterpret_cast<const char *>(numbers.data()),
      numbers.size() * sizeof(NumericType));
  return makeDenseIntOrFPElementsAttr(type, bytes, alignof(NumericType));
}

} // namespace onnx_mlir