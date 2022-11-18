/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ONNXAttributes.hpp -----------------------------===//
//
// This file defines attributes in the ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "src/Dialect/ONNX/ONNXAttributes.hpp.inc"

namespace mlir {
class ElementsAttr;
}

namespace onnx_mlir {

// Prints elements the same way as DenseElementsAttr.
void printIntOrFPElementsAttrAsDense(
    mlir::ElementsAttr attr, mlir::AsmPrinter &printer);

void printIntOrFPElementsAttrAsDenseWithoutType(
    mlir::ElementsAttr attr, mlir::AsmPrinter &printer);

} // namespace onnx_mlir
