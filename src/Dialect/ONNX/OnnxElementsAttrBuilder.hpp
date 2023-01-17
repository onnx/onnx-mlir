/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- OnnxElementsAttrBuilder.hpp ---------------------===//
//
// ElementsAttrBuilder for ONNXDialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"

namespace mlir {
class MLIRContext;
}

namespace onnx_mlir {

// The purpose of this wrapper around ElementsAttrBuilder is that it has a
// convenient constructor that takes an mlir context argument. It does the work
// of retrieving the DisposablePool needed to instantiate ElementsAttrBuilder.
struct OnnxElementsAttrBuilder : ElementsAttrBuilder {
  OnnxElementsAttrBuilder(mlir::MLIRContext *context);
};

} // namespace onnx_mlir
