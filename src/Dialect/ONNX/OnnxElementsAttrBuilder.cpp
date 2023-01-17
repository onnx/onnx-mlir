/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- OnnxElementsAttrBuilder.cpp ---------------------===//
//
// ElementsAttrBuilder for ONNXDialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

using namespace mlir;

namespace onnx_mlir {

OnnxElementsAttrBuilder::OnnxElementsAttrBuilder(MLIRContext *context)
    : ElementsAttrBuilder(*DisposablePool::get<ONNXDialect>(context)) {}

} // namespace onnx_mlir
