/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttributeStorage.hpp"

#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <string>

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

template <BType BTYPE>
inline constexpr CppType<BTYPE> narrow(WideNum n) {
  return n.narrow<BTYPE>();
}

// Copies src to dstBytes while narrowing to the given datatype.
void narrowArray(
    BType bt, ArrayRef<WideNum> src, MutableArrayRef<char> dstBytes) {
  dispatchByBType(bt, [src, dstBytes](auto btype) {
    auto dst = castMutableArrayRef<CppType<btype>>(dstBytes);
    assert(src.size() == dst.size() && "narrowArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), narrow<btype>);
  });
}

// Copies srcBytes to dst while widening from the given datatype.
void widenArray(
    BType bt, ArrayRef<char> srcBytes, MutableArrayRef<WideNum> dst) {
  dispatchByBType(bt, [srcBytes, dst](auto btype) {
    auto src = castArrayRef<CppType<btype>>(srcBytes);
    assert(src.size() == dst.size() && "widenArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), WideNum::widen<btype>);
  });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    size_t id, BType bufferBType, ArrayRef<int64_t> strides,
    const Buffer &buffer, Transformer transformer) {
  BType btype = btypeOfMlirType(type.getElementType());
  assert((transformer != nullptr ||
             wideBTypeOfBType(bufferBType) == wideBTypeOfBType(btype)) &&
         "buffer wide type mismatch requires transformer");
  bool isContiguous = areStridesContiguous(type.getShape(), strides);
  DisposableElementsAttr a = Base::get(
      type.getContext(), type, strides, bufferBType, btype, isContiguous, id);
  DisposableElementsAttributeStorage &s = *a.getImpl();
  s.buffer = buffer;
  s.transformer = std::move(transformer);
  return a;
}

void DisposableElementsAttr::dispose() {
  getImpl()->buffer.reset();
  getImpl()->transformer = nullptr;
}

bool DisposableElementsAttr::isSplat() const {
  return areStridesSplat(getStrides()) && getBuffer()->getBufferSize() != 0;
}

BType DisposableElementsAttr::getBType() const { return getImpl()->btype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

bool DisposableElementsAttr::isDisposed() const { return !getImpl()->buffer; }

size_t DisposableElementsAttr::getId() const { return getImpl()->id; }

ArrayRef<int64_t> DisposableElementsAttr::getStrides() const {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

auto DisposableElementsAttr::getTransformer() const -> const Transformer & {
  assert(!isDisposed());
  return getImpl()->transformer;
}

bool DisposableElementsAttr::isContiguous() const {
  return getImpl()->isContiguous;
}

bool DisposableElementsAttr::isTransformed() const {
  return getImpl()->transformer != nullptr;
}

bool DisposableElementsAttr::isTransformedOrCast() const {
  return isTransformed() || getBType() != getBufferBType();
}

BType DisposableElementsAttr::getBufferBType() const {
  return getImpl()->bufferBType;
}

unsigned DisposableElementsAttr::getBufferElementBytewidth() const {
  return bytewidthOfBType(getBufferBType());
}

int64_t DisposableElementsAttr::getNumBufferElements() const {
  return getBuffer()->getBufferSize() / getBufferElementBytewidth();
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  if (isContiguous()) {
    return getBufferAsWideNums();
  }
  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(getNumElements());
  readWideNums(dst);
  return std::move(dst);
}

void DisposableElementsAttr::readWideNums(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    readBytesAsWideNums(getBufferBytes(), dst);
    return;
  }
  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  restrideArray<WideNum>(getShape(), getStrides(), src.get(), dst);
}

DenseElementsAttr DisposableElementsAttr::toDenseElementsAttr() const {
  if (isSplat())
    return DenseElementsAttr::get(getType(), {getSplatValue<Attribute>()});
  ArrayBuffer<char> bytes = getRawBytes();
  if (getElementType().isInteger(1))
    // don't use getFromRawBuffer which requires bit packing
    return DenseElementsAttr::get(getType(), castArrayRef<bool>(bytes.get()));
  return DenseElementsAttr::getFromRawBuffer(getType(), bytes.get());
}

namespace {
// Perform byte swap if system endianness is BE and elements are multi-byte.
bool shouldSwapLEBytes(unsigned elementByteWidth) {
  return elementByteWidth > 1 &&
         llvm::endianness::native != llvm::endianness::little;
}
} // namespace

/*static*/
std::unique_ptr<llvm::MemoryBuffer> DisposableElementsAttr::parse(
    AsmParser &parser, ShapedType type) {
  size_t id = 0; // The parsed id is ignored.
  std::string str;
  if (parser.parseLess() || parser.parseInteger(id) || parser.parseColon() ||
      parser.parseString(&str))
    return nullptr;
  StringRef hex = str;
  std::string bytes;
  if (!hex.consume_front("0x") || (hex.size() & 1) ||
      !llvm::tryGetFromHex(hex, bytes)) {
    parser.emitError(parser.getCurrentLocation(), "ill-formed hex string");
    return nullptr;
  }
  if (bytes.size() != static_cast<size_t>(getSizeInBytes(type))) {
    parser.emitError(
        parser.getCurrentLocation(), "data size doesn't match type size");
    return nullptr;
  }
  if (!shouldSwapLEBytes(getIntOrFloatByteWidth(type.getElementType()))) {
    return llvm::MemoryBuffer::getMemBufferCopy(bytes);
  } else {
    // Reorder bytes from little-endian on big-endian platforms:
    std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
        llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes.size());
    DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
        {bytes.data(), bytes.size()}, writeBuffer->getBuffer(), type);
    return writeBuffer;
  }
}

void DisposableElementsAttr::printWithoutType(AsmPrinter &printer) const {
  // It would be ideal if we could read the printer flags from printer instead
  // of constructing them here, because printer may have been constructed with
  // an override of elideLargeElementsAttrs which we cannot see here.
  // Oh well, at least OpPrintingFlags().shouldElideElementsAttr(ElementsAttr)
  // lets us respect the --mlir-elide-elementsattrs-if-larger command line flag.
  static OpPrintingFlags printerFlags{};
  printer << getMnemonic() << "<" << getImpl()->id << ":";
  if (!printerFlags.shouldElideElementsAttr(*this)) {
    auto rawBytes = getRawBytes();
    SmallVector<char> buffer;
    ArrayRef<char> bytes;
    if (!shouldSwapLEBytes(getIntOrFloatByteWidth(getElementType()))) {
      bytes = rawBytes.get();
    } else {
      // Reorder raw bytes to little-endian on big-endian platforms:
      buffer.resize_for_overwrite(rawBytes.get().size());
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawBytes.get(), buffer, getType());
      ArrayRef<char> bufferRef(buffer);
      bytes = bufferRef;
    }
    printer << "\"0x" << llvm::toHex(castArrayRef<uint8_t>(bytes)) << "\"";
  } else {
    printer << "__elided__";
  }
  printer << ">";
}

void DisposableElementsAttr::printAsDenseElementsAttr(
    AsmPrinter &printer) const {
  static OpPrintingFlags printerFlags{};
  if (isSplat() || !printerFlags.shouldElideElementsAttr(*this)) {
    // Take shortcut by first converting to DenseElementsAttr.
    // NOTE: This creates a copy which is never garbage collected. This is not
    // only slow but also defeats the garbage collection benefits of
    // DisposableElementsAttr, depending on when the printing
    // takes place (the print at the end of onnx-mlir-opt in lit tests is ok).
    printer.printAttribute(toDenseElementsAttr());
    // TODO: Do the work to print without constructing DenseElementsAttr.
  } else {
    // In this special case it's easy to avoid conversion to DenseElementsAttr.
    printer << "dense<__elided__> : " << getType();
  }
}

void DisposableElementsAttr::readBytesAsWideNums(
    ArrayRef<char> srcBytes, llvm::MutableArrayRef<WideNum> dst) const {
  widenArray(getBufferBType(), srcBytes, dst);
  if (const Transformer &transformer = getTransformer())
    transformer(dst);
}

ArrayRef<char> DisposableElementsAttr::getBufferBytes() const {
  return asArrayRef(getBuffer()->getBuffer());
}

ArrayBuffer<WideNum> DisposableElementsAttr::getBufferAsWideNums() const {
  if (!isTransformed() && getBufferElementBytewidth() == sizeof(WideNum)) {
    return castArrayRef<WideNum>(getBufferBytes());
  }
  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(getNumBufferElements());
  readBytesAsWideNums(getBufferBytes(), dst);
  return std::move(dst);
}

WideNum DisposableElementsAttr::atFlatIndex(size_t flatIndex) const {
  size_t pos = flatIndexToBufferPos(flatIndex);
  unsigned bufBytewidth = getBufferElementBytewidth();
  ArrayRef<char> bytes =
      getBufferBytes().slice(pos * bufBytewidth, bufBytewidth);
  WideNum n;
  readBytesAsWideNums(bytes, llvm::MutableArrayRef(n));
  return n;
}

size_t DisposableElementsAttr::flatIndexToBufferPos(size_t flatIndex) const {
  if (flatIndex == 0 || isContiguous())
    return flatIndex;
  if (isSplat())
    return 0;
  auto indices = unflattenIndex(getShape(), flatIndex);
  return getStridesPosition(indices, getStrides());
}

void DisposableElementsAttr::readRawBytes(
    MutableArrayRef<char> dstBytes) const {
  BType btype = getBType();
  unsigned elemBytewidth = bytewidthOfBType(btype);
  if (!isTransformedOrCast()) {
    auto srcBytes = getBufferBytes();
    restrideArray(elemBytewidth, getShape(), getStrides(), srcBytes, dstBytes);
  } else if (elemBytewidth == sizeof(WideNum)) {
    readWideNums(castMutableArrayRef<WideNum>(dstBytes));
  } else {
    SmallVector<WideNum, 1> dst;
    dst.resize_for_overwrite(getNumElements());
    readWideNums(dst);
    narrowArray(btype, dst, dstBytes);
  }
}

ArrayBuffer<char> DisposableElementsAttr::getRawBytes() const {
  if (!isTransformedOrCast() && isContiguous())
    return getBufferBytes();
  unsigned elemBytewidth = bytewidthOfBType(getBType());
  ArrayBuffer<char>::Vector dstBytes;
  dstBytes.resize_for_overwrite(getNumElements() * elemBytewidth);
  readRawBytes(dstBytes);
  return std::move(dstBytes);
}

} // namespace mlir
