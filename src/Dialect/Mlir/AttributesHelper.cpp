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

namespace {
// Always align to the largest possible element type.
// TODO: Consider aligning for SIMD ops.
constexpr size_t ALIGN = std::max(alignof(int64_t), alignof(double));

size_t byteWidth(size_t bitWidth) {
  if (bitWidth == 1)
    return 1;
  constexpr size_t BYTE_BITWIDTH = 8;
  assert(
      bitWidth % BYTE_BITWIDTH == 0 && "non-boolean types must fill out bytes");
  return bitWidth / BYTE_BITWIDTH;
}
} // namespace

ElementsAttr makeDenseIntOrFPElementsAttrFromRawBuffer(
    ShapedType type, ArrayRef<char> bytes) {
  assert(static_cast<size_t>(type.getNumElements()) ==
             bytes.size() /
                 byteWidth(type.getElementType().getIntOrFloatBitWidth()) &&
         "data size must match type");
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    DenseResourceElementsHandle r = resourcePool->createResource(
        HeapAsmResourceBlob::allocateAndCopy(bytes, ALIGN, false));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    return DenseElementsAttr::getFromRawBuffer(type, bytes);
  }
}

ElementsAttr makeDenseIntOrFPElementsAttrWithRawBuffer(
    ShapedType type, FillDenseRawBufferFn fill) {
  size_t size = type.getNumElements() *
                byteWidth(type.getElementType().getIntOrFloatBitWidth());
  if (ResourcePool *resourcePool = ResourcePool::get(type.getContext());
      resourcePool && resourcePool->isActive()) {
    AsmResourceBlob blob = HeapAsmResourceBlob::allocate(size, ALIGN);
    fill(blob.getMutableData());
    DenseResourceElementsHandle r =
        resourcePool->createResource(std::move(blob));
    return DenseResourceElementsAttr::get(type, r);
  } else {
    std::vector<char> bytes(size, 0);
    fill(bytes);
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