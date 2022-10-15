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

mlir::ElementsAttr makeDenseElementsAttr(
    mlir::ShapedType type, char *data, size_t size);

}