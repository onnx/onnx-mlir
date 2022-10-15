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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "src/Dialect/Mlir/ResourcePool.hpp"

using namespace mlir;

namespace onnx_mlir {

ElementsAttr makeDenseIntOrFPElementsAttr(
    ShapedType type, char *data, size_t size) {
  // TODO: test type.getElementType() is int or fp
  ArrayRef<char> bytes = llvm::makeArrayRef(data, size);
  ResourcePool *resourcePool = ResourcePool::get(type.getContext());
  if (resourcePool && resourcePool->isActive()) {
    // TODO: consider aligning everything to something large that works well for
    //       everything, e.g. 8 for double and i64, or 16 or 64 for SIMD ops
    mlir::DenseResourceElementsHandle r =
        ResourcePool::get(type.getContext())
            ->createResource(
                HeapAsmResourceBlob::allocateAndCopy(bytes, alignof(char)));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
}

} // namespace onnx_mlir