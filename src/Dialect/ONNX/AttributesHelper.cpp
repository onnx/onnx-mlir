/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.cpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/AttributesHelper.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Support/Arrays.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

DenseElementsAttr makeDenseElementsAttrFromRawBytes(
    ShapedType type, ArrayRef<char> bytes) {
  size_t bytewidth = getIntOrFloatByteWidth(type.getElementType());
  assert(bytes.size() == type.getNumElements() * bytewidth &&
         "data size must match type");
  if (type.getElementType().isInteger(1)) {
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(type, castArrayRef<bool>(bytes));
  }
  return DenseElementsAttr::getFromRawBuffer(type, bytes);
}

} // namespace

DenseElementsAttr toDenseElementsAttr(ElementsAttr elements) {
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense;
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    ArrayBuffer<char> bytes = disposable.getRawBytes();
    return makeDenseElementsAttrFromRawBytes(disposable.getType(), bytes.get());
  }
  // TODO: consider reading data from elements.getValues() instead of giving up
  llvm_unreachable("unexpected ElementsAttr instance");
}

} // namespace onnx_mlir
