/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testBoolsValidityAndSplatness(ArrayRef<char> bytes) {
  return !bytes.empty() && llvm::all_of(bytes, [bytes](char x) {
    assert(x == 0 || x == 1);
    return x == bytes[0];
  });
}

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testRawBytesValidityAndSplatness(
    ShapedType type, DType bufferDType, ArrayRef<char> bytes) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  assert(wideDTypeOfDType(dtype) == wideDTypeOfDType(bufferDType));
  if (bufferDType == DType::BOOL) {
    assert(static_cast<size_t>(type.getNumElements()) == bytes.size());
    return testBoolsValidityAndSplatness(bytes);
  }
  ShapedType bufferType =
      dtype == bufferDType
          ? type
          : type.clone(mlirTypeOfDType(bufferDType, type.getContext()));
  bool isSplat;
  if (!DenseElementsAttr::isValidRawBuffer(bufferType, bytes, isSplat))
    llvm_unreachable("invalid dense int or fps raw buffer");
  return isSplat;
}

} // namespace

ElementsAttrBuilder::ElementsAttrBuilder(DisposablePool &disposablePool)
    : disposablePool(disposablePool) {}

ElementsAttrBuilder::ElementsAttrBuilder(mlir::MLIRContext *context)
    : disposablePool(*DisposablePool::get(context)) {}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromElementsAttr(
    mlir::ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    // TODO: call fromRawBytes to reduce code duplication
    bool isSplat = dense.isSplat();
    std::unique_ptr<llvm::MemoryBuffer> buffer;
    if (dense.getElementType().isInteger(1)) {
      size_t size = isSplat ? 1 : dense.getNumElements();
      std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
          llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
      std::copy_n(
          dense.value_begin<bool>(), size, writeBuffer->getBuffer().begin());
      buffer = std::move(writeBuffer);
    } else {
      StringRef s = asStringRef(dense.getRawData());
      buffer = llvm::MemoryBuffer::getMemBuffer(
          s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
    }
    ArrayRef<int64_t> emptyStrides; // empty strides when splat
    return isSplat ? create(dense.getType(), std::move(buffer), emptyStrides)
                   : create(dense.getType(), std::move(buffer));
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, DType bufferDType, ArrayRef<char> bytes, bool mustCopy) {
  bool isSplat = testRawBytesValidityAndSplatness(type, bufferDType, bytes);
  StringRef s = asStringRef(
      isSplat ? bytes.take_front(bitwidthOfDType(bufferDType)) : bytes);
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  if (mustCopy) {
    buffer = llvm::MemoryBuffer::getMemBufferCopy(s);
  } else {
    buffer = llvm::MemoryBuffer::getMemBuffer(
        s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
  ArrayRef<int64_t> emptyStrides; // empty strides when splat
  return isSplat ? create(type, std::move(buffer), emptyStrides, bufferDType)
                 : create(type, std::move(buffer), None, bufferDType);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, DType bufferDType, const Filler<char> &bytesFiller) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  ShapedType bufferType =
      dtype == bufferDType
          ? type
          : type.clone(mlirTypeOfDType(bufferDType, type.getContext()));
  size_t size = type.getNumElements() * bytewidthOfDType(bufferDType);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  bytesFiller(writeBuffer->getBuffer());
  bool isSplat = testRawBytesValidityAndSplatness(
      type, bufferDType, writeBuffer->getBuffer());
  (void)isSplat;
  // TODO: consider replacing writeBuffer with single element buffer if isSplat
  return create(type, std::move(writeBuffer));
}

mlir::DisposableElementsAttr ElementsAttrBuilder::transform(
    mlir::DisposableElementsAttr elms, Type transformedElementType,
    Transformer transformer) {
  return elms.transform(*this, transformedElementType, transformer);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::castElementType(
    mlir::DisposableElementsAttr elms, Type newElementType) {
  return elms.castElementType(*this, newElementType);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::transpose(
    mlir::DisposableElementsAttr elms, ArrayRef<uint64_t> perm) {
  return elms.transpose(*this, perm);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::reshape(
    mlir::DisposableElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  return elms.reshape(*this, reshapedShape);
}

// Broadcasts like the ONNX Expand op.
DisposableElementsAttr ElementsAttrBuilder::expand(
    mlir::DisposableElementsAttr elms, ArrayRef<int64_t> expandedShape) {
  return elms.expand(*this, expandedShape);
}

} // namespace onnx_mlir