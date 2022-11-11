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

using namespace mlir;

namespace onnx_mlir {

namespace {

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testBoolsValidityAndSplatness(ArrayRef<char> bytes) {
  for (char c : bytes)
    assert(c == 0 || c == 1);
  return bytes.size() == 1;
}

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testRawBytesValidityAndSplatness(
    ShapedType type, DType bufferDType, ArrayRef<char> bytes) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  assert(wideDTypeOfDType(dtype) == wideDTypeOfDType(bufferDType));
  if (bufferDType == DType::BOOL) {
    size_t numElements = type.getNumElements();
    assert(bytes.size() == numElements || bytes.size() == 1);
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
    ShapedType type = dense.getType();
    DType dtype = dtypeOfMlirType(type.getElementType());
    std::unique_ptr<llvm::MemoryBuffer> buffer;
    if (dtype == DType::BOOL) {
      if (dense.isSplat()) {
        char b = dense.getSplatValue<bool>();
        return fromRawBytes(
            type, dtype, llvm::makeArrayRef(b), /*mustCopy=*/true);
      } else {
        return fromRawBytes(type, dtype, [dense](MutableArrayRef<char> dst) {
          std::copy_n(
              dense.value_begin<bool>(), dense.getNumElements(), dst.begin());
        });
      }
    } else {
      return fromRawBytes(type, dtype, dense.getRawData(), /*mustCopy=*/false);
    }
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, DType bufferDType, ArrayRef<char> bytes, bool mustCopy) {
  bool isSplat = testRawBytesValidityAndSplatness(type, bufferDType, bytes);
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  StringRef s = asStringRef(bytes);
  if (mustCopy) {
    buffer = llvm::MemoryBuffer::getMemBufferCopy(s);
  } else {
    buffer = llvm::MemoryBuffer::getMemBuffer(
        s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
  if (isSplat) {
    SmallVector<int64_t, 4> zerosStrides(type.getRank(), 0);
    return create(
        type, std::move(buffer), makeArrayRef(zerosStrides), bufferDType);
  }
  return create(type, std::move(buffer), None, bufferDType);
}

mlir::DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, DType bufferDType, const Filler<char> &bytesFiller) {
  size_t size = type.getNumElements() * bytewidthOfDType(bufferDType);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  bytesFiller(writeBuffer->getBuffer());
  // We trust bytesFiller and skip testRawBytesValidityAndSplatness()
  return create(type, std::move(writeBuffer), None, bufferDType);
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