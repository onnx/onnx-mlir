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

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXAttributes.hpp"

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

DisposableElementsAttr::Reader getIdentityReader(DType dtype) {
  return dispatchByDType(
      dtype, [](auto staticDType) { return identityReader<staticDType>; });
}

} // namespace

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(
    ShapedType type, const Buffer &buffer, Optional<Strides> optionalStrides) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  return get(type, buffer, optionalStrides, dtype);
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::get(ShapedType type,
    const Buffer &buffer, Optional<Strides> optionalStrides, DType bufferDType,
    Reader reader) {
  DType dtype = dtypeOfMlirType(type.getElementType());
  assert(wideDTypeOfDType(dtype) == wideDTypeOfDType(bufferDType) ||
         reader != nullptr);
  Properties properties = {.dtype = dtype,
      .bufferDType = bufferDType,
      // .isContiguous is set below
      .isTransformed = reader != nullptr};
  SmallVector<int64_t, 4> strides;
  if (optionalStrides.has_value()) {
    strides.assign(optionalStrides->begin(), optionalStrides->end());
    properties.isContiguous = areStridesContiguous(type.getShape(), strides);
  } else {
    strides = getDefaultStrides(type.getShape());
    properties.isContiguous = true;
  }
  return create(type, buffer, strides, properties, std::move(reader));
}

/*static*/
DisposableElementsAttr DisposableElementsAttr::create(ShapedType type,
    const Buffer &buffer, Strides strides, Properties properties,
    Reader reader) {
  DisposableElementsAttr a =
      Base::get(type.getContext(), type, strides, properties);
  Storage &s = *a.getImpl();
  s.buffer = buffer;
  if (reader) {
    s.reader = std::move(reader);
  } else {
    assert(wideDTypeOfDType(properties.bufferDType) ==
               wideDTypeOfDType(properties.dtype) &&
           "buffer wide type mismatch requires transforming reader");
    s.reader = getIdentityReader(properties.bufferDType);
  }
  return a;
}

auto DisposableElementsAttr::getStrides() const -> Strides {
  return getImpl()->strides;
}

auto DisposableElementsAttr::getProperties() const -> const Properties & {
  return getImpl()->properties;
}

auto DisposableElementsAttr::getBuffer() const -> const Buffer & {
  assert(!isDisposed());
  return getImpl()->buffer;
}

auto DisposableElementsAttr::getReader() const -> const Reader & {
  assert(!isDisposed());
  return getImpl()->reader;
}

auto DisposableElementsAttr::getReaderOrNull() const -> Reader {
  if (getProperties().isTransformed)
    return getReader();
  else
    return nullptr;
}

bool DisposableElementsAttr::isDisposed() const { return !getImpl()->buffer; }

bool DisposableElementsAttr::isContiguous() const {
  return getProperties().isContiguous;
}

DType DisposableElementsAttr::getBufferDType() const {
  return getProperties().bufferDType;
}

unsigned DisposableElementsAttr::getBufferElementBytewidth() const {
  return bytewidthOfDType(getProperties().bufferDType);
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

DType DisposableElementsAttr::getDType() const { return getProperties().dtype; }

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
  const Properties &properties = getProperties();
  if (!properties.isTransformed &&
      getBufferElementBytewidth() == sizeof(WideNum)) {
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

void DisposableElementsAttr::readElements(MutableArrayRef<WideNum> dst) const {
  if (isContiguous()) {
    getReader()(getBufferString(), dst);
    return;
  }
  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  restrideArray<WideNum>(getShape(), {getStrides(), src.get()}, dst);
}

ArrayBuffer<WideNum> DisposableElementsAttr::getWideNums() const {
  const Properties &properties = getProperties();
  if (!properties.isTransformed && properties.isContiguous &&
      getBufferElementBytewidth() == sizeof(WideNum)) {
    return asArrayRef<WideNum>(getBufferString());
  }
  ArrayBuffer<WideNum>::Vector wideData;
  wideData.resize_for_overwrite(getNumElements());
  readElements(wideData);
  return std::move(wideData);
}

ArrayBuffer<char> DisposableElementsAttr::getRawBytes() const {
  const Properties &properties = getProperties();
  bool requiresNoElementwiseTransformOrCast =
      !properties.isTransformed && properties.dtype == properties.bufferDType;
  if (requiresNoElementwiseTransformOrCast && properties.isContiguous)
    return getBufferBytes();
  unsigned attrBytewidth = bytewidthOfDType(properties.dtype);
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(getNumElements() * attrBytewidth);
  MutableArrayRef<char> bytes(vec);
  if (requiresNoElementwiseTransformOrCast) {
    auto src = getBufferBytes();
    restrideArray(attrBytewidth, getShape(), {getStrides(), src}, bytes);
  } else if (attrBytewidth == sizeof(WideNum)) {
    readElements(castMutableArrayRef<WideNum>(bytes));
  } else {
    SmallVector<WideNum, 1> wideData;
    wideData.resize_for_overwrite(getNumElements());
    readElements(wideData);
    narrowArray(getElementType(), wideData, bytes);
  }
  return std::move(vec);
}

void DisposableElementsAttr::printWithoutType(raw_ostream &os) const {
  printIntOrFPElementsAttrAsDenseWithoutType(*this, os);
}

// TODO: move all the following to ElementsAttrBuilder

namespace {
DisposableElementsAttr::Reader composeReadTransform(
    const DisposableElementsAttr::Reader &reader,
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
  if (auto transposedStrides = transposeStrides(shape, strides, perm)) {
    return elmsBuilder.create(transposedType, getBuffer(),
        makeArrayRef(*transposedStrides), getBufferDType(), getReaderOrNull());
  }

  // TODO: Consider transposing without transforming (just carry over the
  //       reader) when getNumBufferElements() == getNumElements(), i.e.
  //       strides have no zeros.

  ArrayBuffer<WideNum> src = getBufferAsWideNums();
  auto newStrides = getDefaultStrides(transposedShape);
  auto reverseStrides = untransposeDims(newStrides, perm);
  return elmsBuilder.fromWideNums(transposedType, [&](MutableArrayRef<WideNum>
                                                          dst) {
    restrideArray<WideNum>(shape, {strides, src.get()}, {reverseStrides, dst});
  });
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

  // TODO: Consider reshaping without transforming (just carry over the
  //       reader) when getNumBufferElements() == getNumElements(), i.e.
  //       strides have no zeros.

  return elmsBuilder.fromWideNums(
      reshapedType, [this](MutableArrayRef<WideNum> wideData) {
        this->readElements(wideData);
      });
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