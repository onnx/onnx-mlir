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
  llvm::ArrayRef<char> bytes(reinterpret_cast<const char *>(numbers.data()),
      numbers.size() * sizeof(NumericType));
  return makeDenseIntOrFPElementsAttrFromRawBuffer(
      type, bytes, alignof(NumericType));
}

typedef llvm::function_ref<void(llvm::MutableArrayRef<char>)>
    FillDenseRawBufferFn;

mlir::ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    mlir::ShapedType type, FillDenseRawBufferFn fill, size_t align);

llvm::ArrayRef<char> getDenseIntOrFPRawData(mlir::ElementsAttr elements);

} // namespace onnx_mlir