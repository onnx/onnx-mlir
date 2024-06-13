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
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {

WideNum getElementsSplatWideNum(ElementsAttr elms) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.getSplatValue<WideNum>();
  Type elementType = elms.getElementType();
  if (isa<FloatType>(elementType))
    return WideNum::fromAPFloat(elms.getSplatValue<APFloat>());
  if (auto itype = dyn_cast<IntegerType>(elementType))
    return WideNum::fromAPInt(elms.getSplatValue<APInt>(), !itype.isUnsigned());
  llvm_unreachable("WideNum only supports integer and float types");
}

namespace {
void readDenseElementsWideNums(
    ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (elms.isSplat())
    return std::fill(dst.begin(), dst.end(), getElementsSplatWideNum(elms));
  // TODO: Implement the following in a more efficient way.
  Type elementType = elms.getElementType();
  if (isa<FloatType>(elementType)) {
    llvm::transform(elms.getValues<APFloat>(), dst.begin(),
        [](APFloat f) { return WideNum::fromAPFloat(f); });
  } else if (auto itype = dyn_cast<IntegerType>(elementType)) {
    bool isSigned = !itype.isUnsigned();
    llvm::transform(elms.getValues<APInt>(), dst.begin(),
        [isSigned](APInt i) { return WideNum::fromAPInt(i, isSigned); });
  } else {
    llvm_unreachable("WideNum only supports integer and float types");
  }
}
} // namespace

// Returns a pointer to the underlying data as a flat WideNum array, if
// everything aligns, otherwise makes and returns a copy.
// Precondition: elms.getElementType.isIntOrFloat().
ArrayBuffer<WideNum> getElementsWideNums(ElementsAttr elms) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.getWideNums();

  // Return raw data if non-splat DenseElementsAttr and element type is wide.
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(elms)) {
    auto isWideType = [](Type t) { return t.isInteger(64) || t.isF64(); };
    if (isWideType(dense.getElementType()) && !dense.isSplat())
      return castArrayRef<WideNum>(dense.getRawData());
  }

  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(elms.size());
  readDenseElementsWideNums(elms, dst);
  return std::move(dst);
}

// Copies out the elements in a flat WideNum array in row-major order.
// Precondition: elms.getElementType.isIntOrFloat().
void readElementsWideNums(ElementsAttr elms, MutableArrayRef<WideNum> dst) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms))
    return disposable.readWideNums(dst);
  assert(dst.size() == static_cast<size_t>(elms.size()));
  readDenseElementsWideNums(elms, dst);
}

} // namespace onnx_mlir
