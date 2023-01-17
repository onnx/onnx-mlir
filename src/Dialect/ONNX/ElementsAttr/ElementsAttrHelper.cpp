/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrHelper.cpp -----------------------===//
//
// Helper functions for accessing ElementsAttr contents.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {

WideNum getElementsSplatWideNum(ElementsAttr elms) {
  if (auto disposable = elms.dyn_cast<DisposableElementsAttr>())
    return disposable.getSplatValue<WideNum>();
  BType btype = btypeOfMlirType(elms.getElementType());
  if (isFloatBType(btype))
    return WideNum::fromAPFloat(btype, elms.getSplatValue<APFloat>());
  if (isIntBType(btype))
    return WideNum::fromAPInt(btype, elms.getSplatValue<APInt>());
  llvm_unreachable("WideNum only supports integer and float types");
}

namespace {
void readDenseElementsWideNums(
    ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (elms.isSplat())
    return std::fill(dst.begin(), dst.end(), getElementsSplatWideNum(elms));
  // TODO: Implement the following in a more efficient way.
  BType btype = btypeOfMlirType(elms.getElementType());
  if (isFloatBType(btype)) {
    auto range = elms.getValues<APFloat>();
    std::transform(range.begin(), range.end(), dst.begin(),
        [btype](APFloat f) { return WideNum::fromAPFloat(btype, f); });
  } else if (isIntBType(btype)) {
    auto range = elms.getValues<APInt>();
    std::transform(range.begin(), range.end(), dst.begin(),
        [btype](APInt i) { return WideNum::fromAPInt(btype, i); });
  } else {
    llvm_unreachable("WideNum only supports integer and float types");
  }
}
} // namespace

// Returns a pointer to the underlying data as a flat WideNum array, if
// everything aligns, otherwise makes and returns a copy.
// Precondition: elms.getElementType.isIntOrFloat().
ArrayBuffer<WideNum> getElementsWideNums(ElementsAttr elms) {
  if (auto disposable = elms.dyn_cast<DisposableElementsAttr>())
    return disposable.getWideNums();

  // TODO: If elms is DenseElementsAttr and elm type is wide, return raw data.

  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(elms.size());
  readDenseElementsWideNums(elms, dst);
  return std::move(dst);
}

// Copies out the elements in a flat WideNum array in row-major order.
// Precondition: elms.getElementType.isIntOrFloat().
void readElementsWideNums(ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (auto disposable = elms.dyn_cast<DisposableElementsAttr>())
    return disposable.readWideNums(dst);
  assert(dst.size() == static_cast<size_t>(elms.size()));
  readDenseElementsWideNums(elms, dst);
}

} // namespace onnx_mlir
