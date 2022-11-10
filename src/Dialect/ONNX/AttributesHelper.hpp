/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.hpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/FloatingPoint16.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"

// Enable DenseElementsAttr to operate on float_16, bfloat_16 data types.
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::float_16> {
  static constexpr bool value = true;
};
template <>
struct mlir::DenseElementsAttr::is_valid_cpp_fp_type<onnx_mlir::bfloat_16> {
  static constexpr bool value = true;
};

namespace onnx_mlir {

// Makes deep copy of elements, unless they are already a DenseElementsAttr.
mlir::DenseElementsAttr toDenseElementsAttr(mlir::ElementsAttr elements);

} // namespace onnx_mlir