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
#include "src/Support/DType.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Support/WideNum.hpp"

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

namespace {

ArrayBuffer<char> getElementsRawBytes(ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.getRawBytes();
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    if (dense.getElementType().isInteger(1)) {
      // bool is bit packed in dense, so we copy it out
      size_t size = dense.isSplat() ? 1 : dense.getNumElements();
      ArrayBuffer<char>::Vector vec;
      vec.resize_for_overwrite(size);
      std::copy_n(dense.value_begin<bool>(), size, vec.begin());
      return std::move(vec);
    }
    return dense.getRawData(); // Single splat value or a full array.
  }
  llvm_unreachable("unexpected ElementsAttr instance");
}

void readElements(ElementsAttr elements, MutableArrayRef<WideNum> dst) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    disposable.readElements(dst);
    return;
  }
  ArrayBuffer<char> src = getElementsRawBytes(elements);
  dispatchByMlirType(elements.getElementType(), [&](auto dtype) {
    using W = WideDType<dtype>;
    fillOrTransform(
        castArrayRef<typename W::narrowtype>(src.get()), dst, W::widen);
  });
}

} // namespace

void readIntElements(ElementsAttr elements, MutableArrayRef<int64_t> ints) {
  assert(elements.getType().getElementType().isa<IntegerType>());
  readElements(elements, castMutableArrayRef<WideNum>(ints));
}

void readFPElements(ElementsAttr elements, MutableArrayRef<double> fps) {
  assert(elements.getType().getElementType().isa<FloatType>());
  readElements(elements, castMutableArrayRef<WideNum>(fps));
}

} // namespace onnx_mlir
