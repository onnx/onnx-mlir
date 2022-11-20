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

#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"

#include "llvm/ADT/StringExtras.h"

using namespace onnx_mlir;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)

namespace mlir {

namespace {

// TODO: share implementation with widenArray
template <DType DTYPE>
void identityReader(StringRef s, MutableArrayRef<WideNum> dst) {
  using W = WideDType<DTYPE>;
  auto src = asArrayRef<typename W::narrowtype>(s);
  std::transform(src.begin(), src.end(), dst.begin(), W::widen);
}

auto getIdentityReader(DType dtype) {
  return dispatchByDType(
      dtype, [](auto staticDType) { return identityReader<staticDType>; });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type, size_t id,
    const Buffer &buffer, Optional<Strides> optionalStrides) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  return get(type, id, buffer, optionalStrides, dtype);
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type, size_t id,
    const Buffer &buffer, Optional<Strides> optionalStrides, DType bufferDType,
    Reader reader) {
  SmallVector<int64_t, 4> strides;
  if (optionalStrides.has_value()) {
    strides.assign(optionalStrides->begin(), optionalStrides->end());
  } else {
    strides = getDefaultStrides(type.getShape());
  }
  return create(type, id, buffer, strides, bufferDType, std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    size_t id, const Buffer &buffer, Strides strides, DType bufferDType,
    Reader reader) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  assert((reader != nullptr ||
             wideDTypeOfDType(bufferDType) == wideDTypeOfDType(dtype)) &&
         "buffer wide type mismatch requires transforming reader");
  bool isContiguous = areStridesContiguous(type.getShape(), strides);
  DisposableElementsAttr a = Base::get(
      type.getContext(), type, strides, bufferDType, dtype, isContiguous, id);
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
  return reader ? reader : getIdentityReader(getBufferDType());
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
  return isTransformed() || getDType() != getBufferDType();
}

DType DisposableElementsAttr::getBufferDType() const {
  return getImpl()->bufferDType;
}

unsigned DisposableElementsAttr::getBufferElementBytewidth() const {
  return bytewidthOfDType(getBufferDType());
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

DType DisposableElementsAttr::getDType() const { return getImpl()->dtype; }

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
  unsigned attrBytewidth = bytewidthOfDType(getDType());
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
  unsigned attrBytewidth = bytewidthOfDType(getDType());
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(getNumElements() * attrBytewidth);
  readRawBytes(vec);
  return std::move(vec);
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

// TODO: move all the following to ElementsAttrBuilder

namespace {
auto composeReadTransform(
    const std::function<void(StringRef, MutableArrayRef<WideNum>)> &reader,
    ElementsAttrBuilder::Transformer transformer) {
  return [read = reader, transform = std::move(transformer)](
             StringRef s, MutableArrayRef<WideNum> dst) {
    read(s, dst);
    transform(dst);
  };
}

template <DType SRC_TAG, DType DST_TAG>
void wideCaster(MutableArrayRef<WideNum> nums) {
  using S = WideDType<SRC_TAG>;
  using D = WideDType<DST_TAG>;
  for (WideNum &n : nums)
    n = D::pack(static_cast<typename D::type>(S::unpack(n)));
}

ElementsAttrBuilder::Transformer wideCaster(DType src, DType dst) {
  constexpr DType DBL = DType::DOUBLE, I64 = DType::INT64, U64 = DType::UINT64;
  // clang-format off
  if (src == DBL && dst == I64) return wideCaster<DBL, I64>;
  if (src == DBL && dst == U64) return wideCaster<DBL, U64>;
  if (src == I64 && dst == DBL) return wideCaster<I64, DBL>;
  if (src == I64 && dst == U64) return wideCaster<I64, U64>;
  if (src == U64 && dst == DBL) return wideCaster<U64, DBL>;
  if (src == U64 && dst == I64) return wideCaster<U64, I64>;
  // clang-format on
  llvm_unreachable("wideCaster must be called with 2 different wide types");
}
} // namespace

DisposableElementsAttr DisposableElementsAttr::transform(
    ElementsAttrBuilder &elmsBuilder, Type transformedElementType,
    Transformer transformer) const {
  ShapedType transformedType = getType().clone(transformedElementType);
  return elmsBuilder.create(transformedType, getBuffer(), getStrides(),
      dtypeOfMlirType(transformedElementType),
      composeReadTransform(getReader(), std::move(transformer)));
}

DisposableElementsAttr DisposableElementsAttr::castElementType(
    ElementsAttrBuilder &elmsBuilder, Type newElementType) const {
  if (newElementType == getElementType())
    return *this;

  ShapedType newType = getType().clone(newElementType);
  DType newDType = dtypeOfMlirType(newElementType);
  DType newWideType = wideDTypeOfDType(newDType);
  DType oldWideType = wideDTypeOfDType(getDType());

  if (oldWideType == newWideType)
    return elmsBuilder.create(newType, getBuffer(), getStrides(),
        getBufferDType(), getReaderOrNull());

  Transformer transformer = wideCaster(oldWideType, newWideType);
  Reader reader = composeReadTransform(getReader(), std::move(transformer));
  return elmsBuilder.create(
      newType, getBuffer(), getStrides(), getBufferDType(), std::move(reader));
}

namespace {
bool isIdentityPermutation(ArrayRef<uint64_t> perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != i)
      return false;
  }
  return true;
}
} // namespace

DisposableElementsAttr DisposableElementsAttr::transpose(
    ElementsAttrBuilder &elmsBuilder, ArrayRef<uint64_t> perm) const {
  if (isIdentityPermutation(perm))
    return *this;

  ShapedType type = getType();
  auto shape = type.getShape();
  auto transposedShape = transposeDims(shape, perm);
  ShapedType transposedType = type.clone(transposedShape);
  auto strides = getStrides();
  auto transposedStrides = transposeDims(strides, perm);
  return elmsBuilder.create(transposedType, getBuffer(),
      makeArrayRef(transposedStrides), getBufferDType(), getReaderOrNull());
}

DisposableElementsAttr DisposableElementsAttr::reshape(
    ElementsAttrBuilder &elmsBuilder, ArrayRef<int64_t> reshapedShape) const {
  ShapedType type = getType();
  auto shape = type.getShape();
  if (reshapedShape == shape)
    return *this;

  ShapedType reshapedType = type.clone(reshapedShape);
  auto strides = getStrides();
  if (auto reshapedStrides = reshapeStrides(shape, strides, reshapedShape)) {
    return elmsBuilder.create(reshapedType, getBuffer(),
        makeArrayRef(*reshapedStrides), getBufferDType(), getReaderOrNull());
  }

  if (!isTransformed()) { // Skip WideNums if there's no element-wise transform.
    return elmsBuilder.fromRawBytes(
        reshapedType, getBufferDType(), [this](MutableArrayRef<char> dst) {
          auto src = getBufferBytes();
          restrideArray(getBufferElementBytewidth(), getShape(),
              {getStrides(), src}, dst);
        });
  }

  return elmsBuilder.fromWideNums(reshapedType,
      [this](MutableArrayRef<WideNum> wideData) { readWideNums(wideData); });
}

DisposableElementsAttr DisposableElementsAttr::expand(
    ElementsAttrBuilder &elmsBuilder, ArrayRef<int64_t> expandedShape) const {
  ShapedType type = getType();
  if (expandedShape == type.getShape())
    return *this;

  ShapedType expandedType = type.clone(expandedShape);
  auto expandedStrides = expandStrides(getStrides(), expandedShape);
  return elmsBuilder.create(expandedType, getBuffer(),
      makeArrayRef(expandedStrides), getBufferDType(), getReaderOrNull());
}

} // namespace mlir