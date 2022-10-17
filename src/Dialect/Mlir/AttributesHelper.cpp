/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ AttributesHelper.cpp ------------------------===//
//
// Attributes helper functions.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Mlir/AttributesHelper.hpp"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "src/Dialect/Mlir/ResourcePool.hpp"

using namespace mlir;

namespace onnx_mlir {

ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    ShapedType type, ArrayRef<char> bytes, size_t align) {
  assert(type.getElementType().isIntOrFloat());
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    // TODO: consider aligning everything to something large that works well for
    //       everything, e.g. 8 for double and i64, or 16 or 64 for SIMD ops
    mlir::DenseResourceElementsHandle r = resourcePool->createResource(
        HeapAsmResourceBlob::allocateAndCopy(bytes, align));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
}

ArrayRef<char> getDenseIntOrFPRawData(ElementsAttr elements) {
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense.getRawData();
  if (auto x = elements.dyn_cast<DenseResourceElementsAttr>())
    return x.getRawHandle().getResource()->getBlob()->getData();
  llvm_unreachable("unexpected ElementsAttr instance");
}

} // namespace onnx_mlir