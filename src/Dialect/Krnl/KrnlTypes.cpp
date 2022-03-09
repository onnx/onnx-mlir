/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- KrnlTypes.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains definition of krnl types.
//
//===----------------------------------------------------------------------===//

#include "KrnlTypes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

void customizeTypeConverter(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    Type elementType = type.getElementType();
    if (!elementType.isa<krnl::StringType>())
      return llvm::None;

    elementType =
        elementType.cast<krnl::StringType>().getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](krnl::StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });
}

} // namespace krnl
} // namespace onnx_mlir
