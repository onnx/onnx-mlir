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
    ShapedType type, BType bufferBType, ArrayRef<char> bytes) {
  BType btype = btypeOfMlirType(type.getElementType());
  assert(wideBTypeOfBType(btype) == wideBTypeOfBType(bufferBType));
  if (bufferBType == BType::BOOL) {
    size_t numElements = type.getNumElements();
    assert(bytes.size() == numElements || bytes.size() == 1);
    return testBoolsValidityAndSplatness(bytes);
  }
  ShapedType bufferType =
      btype == bufferBType
          ? type
          : type.clone(mlirTypeOfBType(bufferBType, type.getContext()));
  bool isSplat;
  if (!DenseElementsAttr::isValidRawBuffer(bufferType, bytes, isSplat))
    llvm_unreachable("invalid dense int or fps raw buffer");
  return isSplat;
}

} // namespace

std::atomic<size_t> ElementsAttrBuilder::counter{0};

ElementsAttrBuilder::ElementsAttrBuilder(DisposablePool &disposablePool)
    : disposablePool(disposablePool) {}

ElementsAttrBuilder::ElementsAttrBuilder(MLIRContext *context)
    : disposablePool(*DisposablePool::get(context)) {}

DisposableElementsAttr ElementsAttrBuilder::fromMemoryBuffer(ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  return create(type, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::fromElementsAttr(
    ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ShapedType type = dense.getType();
    BType btype = btypeOfMlirType(type.getElementType());
    std::unique_ptr<llvm::MemoryBuffer> buffer;
    if (btype == BType::BOOL) {
      if (dense.isSplat()) {
        char b = dense.getSplatValue<bool>();
        return fromRawBytes(
            type, btype, llvm::makeArrayRef(b), /*mustCopy=*/true);
      } else {
        return fromRawBytes(type, btype, [dense](MutableArrayRef<char> dst) {
          std::copy_n(
              dense.value_begin<bool>(), dense.getNumElements(), dst.begin());
        });
      }
    } else {
      return fromRawBytes(type, btype, dense.getRawData(), /*mustCopy=*/false);
    }
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, BType bufferBType, ArrayRef<char> bytes, bool mustCopy) {
  bool isSplat = testRawBytesValidityAndSplatness(type, bufferBType, bytes);
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
        type, std::move(buffer), makeArrayRef(zerosStrides), bufferBType);
  }
  return create(type, std::move(buffer), None, bufferBType);
}

DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, BType bufferBType, const Filler<char> &bytesFiller) {
  size_t size = type.getNumElements() * bytewidthOfBType(bufferBType);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  bytesFiller(writeBuffer->getBuffer());
  // We trust bytesFiller and skip testRawBytesValidityAndSplatness()
  return create(type, std::move(writeBuffer), None, bufferBType);
}

DisposableElementsAttr ElementsAttrBuilder::fromWideNums(
    ShapedType type, llvm::ArrayRef<WideNum> wideData, bool mustCopy) {
  BType bufferBType = wideBTypeOfBType(btypeOfMlirType(type.getElementType()));
  return fromRawBytes(
      type, bufferBType, castArrayRef<char>(wideData), mustCopy);
}

DisposableElementsAttr ElementsAttrBuilder::fromWideNums(
    ShapedType type, const Filler<WideNum> &wideDataFiller) {
  BType bufferBType = wideBTypeOfBType(btypeOfMlirType(type.getElementType()));
  return fromRawBytes(
      type, bufferBType, [&wideDataFiller](llvm::MutableArrayRef<char> bytes) {
        wideDataFiller(castMutableArrayRef<WideNum>(bytes));
      });
}

DisposableElementsAttr ElementsAttrBuilder::transform(
    DisposableElementsAttr elms, Type transformedElementType,
    Transformer transformer) {
  return elms.transform(*this, transformedElementType, transformer);
}

DisposableElementsAttr ElementsAttrBuilder::castElementType(
    DisposableElementsAttr elms, Type newElementType) {
  return elms.castElementType(*this, newElementType);
}

DisposableElementsAttr ElementsAttrBuilder::transpose(
    DisposableElementsAttr elms, ArrayRef<uint64_t> perm) {
  return elms.transpose(*this, perm);
}

DisposableElementsAttr ElementsAttrBuilder::reshape(
    DisposableElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  return elms.reshape(*this, reshapedShape);
}

// Broadcasts like the ONNX Expand op.
DisposableElementsAttr ElementsAttrBuilder::expand(
    DisposableElementsAttr elms, ArrayRef<int64_t> expandedShape) {
  return elms.expand(*this, expandedShape);
}

} // namespace onnx_mlir