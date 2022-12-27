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

#include <utility>

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

// Copies src to dstBytes while narrowing to the given datatype.
void narrowArray(
    BType bt, ArrayRef<WideNum> src, MutableArrayRef<char> dstBytes) {
  dispatchByBType(bt, [src, dstBytes](auto btype) {
    using W = WideBType<btype>;
    auto dst = castMutableArrayRef<typename W::narrowtype>(dstBytes);
    assert(src.size() == dst.size() && "narrowArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), W::narrow);
  });
}

// Copies srcBytes to dst while widening from the given datatype.
void widenArray(
    BType bt, ArrayRef<char> srcBytes, MutableArrayRef<WideNum> dst) {
  dispatchByBType(bt, [srcBytes, dst](auto btype) {
    using W = WideBType<btype>;
    auto src = castArrayRef<typename W::narrowtype>(srcBytes);
    assert(src.size() == dst.size() && "widenArray size mismatch");
    std::transform(src.begin(), src.end(), dst.begin(), W::widen);
  });
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
    Transformer transformer) {
  SmallVector<int64_t, 4> strides;
  if (optionalStrides.has_value()) {
    strides.assign(optionalStrides->begin(), optionalStrides->end());
  } else {
    strides = getDefaultStrides(type.getShape());
  }
  return create(type, id, buffer, strides, bufferBType, std::move(transformer));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    size_t id, const Buffer &buffer, Strides strides, BType bufferBType,
    Transformer transformer) {
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

bool DisposableElementsAttr::isDisposed() const { return !getImpl()->buffer; }

size_t DisposableElementsAttr::getId() const { return getImpl()->id; }

auto DisposableElementsAttr::getStrides() const -> Strides {
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

ArrayRef<char> DisposableElementsAttr::getBufferBytes() const {
  return asArrayRef(getBuffer()->getBuffer());
}

bool DisposableElementsAttr::isSplat() const {
  return areStridesSplat(getStrides()) && getBuffer()->getBufferSize() != 0;
}

BType DisposableElementsAttr::getBType() const { return getImpl()->btype; }

ShapedType DisposableElementsAttr::getType() const { return getImpl()->type; }

void DisposableElementsAttr::readBytesAsWideNums(
    ArrayRef<char> srcBytes, llvm::MutableArrayRef<WideNum> dst) const {
  widenArray(getBufferBType(), srcBytes, dst);
  if (const Transformer &transformer = getTransformer())
    transformer(dst);
}

WideNum DisposableElementsAttr::readBufferPos(size_t pos) const {
  unsigned bufBytewidth = getBufferElementBytewidth();
  ArrayRef<char> bytes =
      getBufferBytes().slice(pos * bufBytewidth, bufBytewidth);
  WideNum n;
  readBytesAsWideNums(bytes, llvm::makeMutableArrayRef(n));
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
    return castArrayRef<WideNum>(getBufferBytes());
  }
  ArrayBuffer<WideNum>::Vector dst;
  dst.resize_for_overwrite(getNumBufferElements());
  readBytesAsWideNums(getBufferBytes(), dst);
  return std::move(dst);
}

auto DisposableElementsAttr::getSplatWideNum() const -> WideNum {
  assert(isSplat() && "expected the attribute to be a splat");
  return readBufferPos(0);
}

void DisposableElementsAttr::readWideNums(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    readBytesAsWideNums(getBufferBytes(), dst);
    return;
  }
  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  restrideArray<WideNum>(getShape(), {getStrides(), src.get()}, dst);
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

void DisposableElementsAttr::readRawBytes(
    MutableArrayRef<char> dstBytes) const {
  BType btype = getBType();
  unsigned elemBytewidth = bytewidthOfBType(btype);
  if (!isTransformedOrCast()) {
    auto srcBytes = getBufferBytes();
    restrideArray(
        elemBytewidth, getShape(), {getStrides(), srcBytes}, dstBytes);
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
