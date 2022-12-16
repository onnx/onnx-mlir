/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.cpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttributeStorage.hpp"

#include "src/Dialect/ONNX/ONNXAttributes.hpp"
#include "src/Support/Strides.hpp"

#include "llvm/ADT/StringExtras.h"

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

// Copies wideData to bytes while narrowing to the elementType datatype.
void narrowArray(
    Type elementType, ArrayRef<WideNum> wideData, MutableArrayRef<char> bytes) {
  dispatchByMlirType(elementType, [wideData, bytes](auto btype) {
    using W = WideBType<btype>;
    auto dst = castMutableArrayRef<typename W::narrowtype>(bytes);
    std::transform(wideData.begin(), wideData.end(), dst.begin(), W::narrow);
  });
}

// Copies s to dst while widening from the BTYPE datatype.
template <BType BTYPE>
void identityReader(StringRef s, MutableArrayRef<WideNum> dst) {
  using W = WideBType<BTYPE>;
  auto src = asArrayRef<typename W::narrowtype>(s);
  std::transform(src.begin(), src.end(), dst.begin(), W::widen);
}

auto getIdentityReader(BType btype) {
  return dispatchByBType(
      btype, [](auto staticBType) { return identityReader<staticBType>; });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type, size_t id,
    const Buffer &buffer, Optional<Strides> optionalStrides) {
  BType btype = btypeOfMlirType(type.getElementType());
  return get(type, id, buffer, optionalStrides, btype);
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type, size_t id,
    const Buffer &buffer, Optional<Strides> optionalStrides, BType bufferBType,
    Reader reader) {
  SmallVector<int64_t, 4> strides;
  if (optionalStrides.has_value()) {
    strides.assign(optionalStrides->begin(), optionalStrides->end());
  } else {
    strides = getDefaultStrides(type.getShape());
  }
  return create(type, id, buffer, strides, bufferBType, std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    size_t id, const Buffer &buffer, Strides strides, BType bufferBType,
    Reader reader) {
  BType btype = btypeOfMlirType(type.getElementType());
  assert((reader != nullptr ||
             wideBTypeOfBType(bufferBType) == wideBTypeOfBType(btype)) &&
         "buffer wide type mismatch requires transforming reader");
  bool isContiguous = areStridesContiguous(type.getShape(), strides);
  DisposableElementsAttr a = Base::get(
      type.getContext(), type, strides, bufferBType, btype, isContiguous, id);
  DisposableElementsAttributeStorage &s = *a.getImpl();
  s.buffer = buffer;
  s.reader = std::move(reader);
  return a;
}

void DisposableElementsAttr::dispose() {
  getImpl()->buffer.reset();
  getImpl()->reader = nullptr;
}

bool DisposableElementsAttr::isDisposed() const { return !getImpl()->buffer; }

size_t DisposableElementsAttr::getId() const { return getImpl()->id; }

auto DisposableElementsAttr::getStrides() const -> Strides {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

// TODO: For better efficiency fix getIdentityReader() to return a const& in a
//       static const table so we don't need to create the identity reader here
//       and can return const& from getReader().
auto DisposableElementsAttr::getReader() const -> Reader {
  const auto &reader = getReaderOrNull();
  return reader ? reader : getIdentityReader(getBufferBType());
}

auto DisposableElementsAttr::getReaderOrNull() const -> Reader {
  assert(!isDisposed());
  return getImpl()->reader;
}

bool DisposableElementsAttr::isContiguous() const {
  return getImpl()->isContiguous;
}

bool DisposableElementsAttr::isTransformed() const {
  return getImpl()->reader != nullptr;
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

StringRef DisposableElementsAttr::getBufferString() const {
  return getBuffer()->getBuffer();
}

ArrayRef<char> DisposableElementsAttr::getBufferBytes() const {
  return asArrayRef(getBufferString());
}

bool DisposableElementsAttr::isSplat() const {
  return areStridesSplat(getStrides()) && getBuffer()->getBufferSize() != 0;
}

BType DisposableElementsAttr::getBType() const { return getImpl()->btype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

WideNum DisposableElementsAttr::readBufferPos(size_t pos) const {
  StringRef s = getBufferString();
  unsigned bufBytewidth = getBufferElementBytewidth();
  StringRef bytes = s.substr(pos * bufBytewidth, bufBytewidth);
  WideNum n;
  getReader()(bytes, llvm::makeMutableArrayRef(n));
  return n;
}

WideNum DisposableElementsAttr::readFlatIndex(size_t flatIndex) const {
  return readBufferPos(flatIndexToBufferPos(flatIndex));
}

size_t DisposableElementsAttr::flatIndexToBufferPos(size_t flatIndex) const {
  if (flatIndex == 0 || isContiguous())
    return flatIndex;
  if (isSplat())
    return 0;
  auto indices = unflattenIndex(getShape(), flatIndex);
  return getStridesPosition(indices, getStrides());
}

ArrayBuffer<WideNum> DisposableElementsAttr::getBufferAsWideNums() const {
  if (!isTransformed() && getBufferElementBytewidth() == sizeof(WideNum)) {
    return asArrayRef<WideNum>(getBufferString());
  }
  ArrayBuffer<WideNum>::Vector wideBufferData;
  wideBufferData.resize_for_overwrite(getNumBufferElements());
  getReader()(getBufferString(), wideBufferData);
  return std::move(wideBufferData);
}

auto DisposableElementsAttr::getSplatWideNum() const -> WideNum {
  assert(isSplat() && "expected the attribute to be a splat");
  return readBufferPos(0);
}

void DisposableElementsAttr::readWideNums(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    getReader()(getBufferString(), dst);
    return;
  }
  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  restrideArray<WideNum>(getShape(), {getStrides(), src.get()}, dst);
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  if (!isTransformed() && isContiguous() &&
      getBufferElementBytewidth() == sizeof(WideNum)) {
    return asArrayRef<WideNum>(getBufferString());
  }
  ArrayBuffer<WideNum>::Vector wideData;
  wideData.resize_for_overwrite(getNumElements());
  readWideNums(wideData);
  return std::move(wideData);
}

void DisposableElementsAttr::readRawBytes(MutableArrayRef<char> dst) const {
  unsigned attrBytewidth = bytewidthOfBType(getBType());
  if (!isTransformedOrCast()) {
    auto src = getBufferBytes();
    restrideArray(attrBytewidth, getShape(), {getStrides(), src}, dst);
  } else if (attrBytewidth == sizeof(WideNum)) {
    readWideNums(castMutableArrayRef<WideNum>(dst));
  } else {
    SmallVector<WideNum, 1> wideData;
    wideData.resize_for_overwrite(getNumElements());
    readWideNums(wideData);
    narrowArray(getElementType(), wideData, dst);
  }
}

ArrayBuffer<char> DisposableElementsAttr::getRawBytes() const {
  if (!isTransformedOrCast() && isContiguous())
    return getBufferBytes();
  unsigned attrBytewidth = bytewidthOfBType(getBType());
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(getNumElements() * attrBytewidth);
  readRawBytes(vec);
  return std::move(vec);
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

void DisposableElementsAttr::printWithoutType(AsmPrinter &printer) const {
  // It would be ideal if we could read the printer flags from printer instead
  // of constructing them here, because printer may have been constructed with
  // an override of elideLargeElementsAttrs which we cannot see here.
  // Oh well, at least OpPrintingFlags().shouldElideElementsAttr(ElementsAttr)
  // lets us respect the --mlir-elide-elementsattrs-if-larger command line flag.
  static OpPrintingFlags printerFlags{};
  printer << "dense_disposable<#" << getImpl()->id << ":";
  if (isSplat() || !printerFlags.shouldElideElementsAttr(*this)) {
    auto bytes = getRawBytes();
    StringRef s = asStringRef(bytes.get());
    printer << "\"0x" << llvm::toHex(s) << "\"";
  } else {
    printer << "__elided__";
  }
  printer << ">";
}

} // namespace mlir
