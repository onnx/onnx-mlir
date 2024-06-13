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
  typeConverter.addConversion([&](MemRefType type) -> std::optional<Type> {
    Type elementType = type.getElementType();
    if (!mlir::isa<krnl::StringType>(elementType))
      return std::nullopt;

    elementType = mlir::cast<krnl::StringType>(elementType)
                      .getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](krnl::StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });

  typeConverter.addConversion([&](NoneType type) -> Type { return type; });
}

} // namespace krnl
} // namespace onnx_mlir
