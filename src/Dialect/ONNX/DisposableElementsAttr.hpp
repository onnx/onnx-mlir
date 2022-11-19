/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.hpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Support/Arrays.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/WideNum.hpp"

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>

namespace onnx_mlir {
class DisposablePool;
class ElementsAttrBuilder;
}; // namespace onnx_mlir

namespace mlir {

struct DisposableElementsAttributeStorage;

// DisposableElementsAttr is an alternative to DenseElementsAttr
// with the following features:
//
// 1. The memory can be heap allocated or mmap'ed from a file and will be
// released (heap allocation freed or file closed) between compiler passes
// when it is no longer reachable from the operation graph.
//
// 2. The data can be represented with higher precision than the element
// data type to avoid cumulative precision loss during constant propagation.
//
// 3. Element wise transformations are recorded lazily as a lambda and
// only materialized on read thus avoiding some memory allocations and
// copies.
//
// 4. Similarly, some tensor shape transformations can be recorded as
// 'strides' metadata without rewriting the underlying data. In particular,
// tensors can be broadcast, reshaped, and transposed in this fashion,
// subject to some constraints, like Numpy arrays and PyTorch tensors.
//
// 5. A set of helper functions makes it possible to work with
// DisposableElementsAttr and DenseElementsAttr interchangeably, and
// DisposableElementsAttr prints the same as DenseElementsAttr so
// we can switch between them without changing lit tests.
// TODO: explain caveats...
//
// NOTE: DenseResourceElementsAttr is an alternative for heap allocated memory
//       (but without garbage collection or the other features listed above).
//
// NOTE: DisposableElementsAttr doesn't support complex numbers and strings.
//       It could be extended with 'DisposableStringElemenetsAttr` and
//       `DisposableComplexElementsAttr' in the same way that
//       DenseElementsAttr has different implementations for strings and
//       numbers.
//
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr, Attribute,
          DisposableElementsAttributeStorage, ElementsAttr::Trait,
          TypedAttr::Trait> {
  using Base::Base;

  // DType and WideNum are ubiquitous in the class definition and these using
  // statements are convenient as they let us omit their namespace qualifier.
  using DType = onnx_mlir::DType;
  using WideNum = onnx_mlir::WideNum;

  using Strides = ArrayRef<int64_t>;
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  // TODO: change reader to take ArrayRef<char> as first parameter
  using Reader = std::function<void(StringRef, MutableArrayRef<WideNum>)>;

  //===----------------------------------------------------------------------===//
  // Instantiation:
  //
  // The get methods are private and are only accessed from
  // ElementsAttrBuilder. Call ElementsAttrBuilder::create(..)
  // to instantiate DisposableElementsAttr.
  //
  // DisposablePool needs access to the private dispose() and getId() methods
  // to track and dispose instances.
  //===----------------------------------------------------------------------===//
public:
  friend class onnx_mlir::DisposablePool;
  friend class onnx_mlir::ElementsAttrBuilder;

private:
  static DisposableElementsAttr get(ShapedType type, size_t id,
      const Buffer &buffer, Optional<Strides> optionalStrides = None);

  // Assumes isTransformed if reader != nullptr.
  static DisposableElementsAttr get(ShapedType type, size_t id,
      const Buffer &buffer, Optional<Strides> optionalStrides,
      DType bufferDType, Reader reader = nullptr);

  // Internal method called by get(..) methods.
  static DisposableElementsAttr create(ShapedType type, size_t id,
      const Buffer &buffer, Strides strides, DType bufferDType,
      Reader reader /*= nullptr*/);

  // Clear the buffer payload shared_ptr which decreases the reference count
  // and, if it reaches zero, frees or closes the underlying MemoryBuffer's
  // heap allocation or file.
  void dispose();

public:
  DisposableElementsAttr(std::nullptr_t) {}

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? cast<ElementsAttr>() : nullptr;
  }

  //===----------------------------------------------------------------------===//
  // Instance properties:
  //===----------------------------------------------------------------------===//
private:
  bool isDisposed() const;

  size_t getId() const;

  Strides getStrides() const;

  const Buffer &getBuffer() const;

  Reader getReader() const;

  Reader getReaderOrNull() const;

  bool isContiguous() const;

  bool isTransformed() const;

  bool isTransformedOrCast() const;

  DType getBufferDType() const;

  unsigned getBufferElementBytewidth() const;

  int64_t getNumBufferElements() const;

public:
  // isSplat() is true if all elements are known to be the same
  // (and are represented as a single number with all-zeros strides).
  // Can return false even if all elements are identical.
  bool isSplat() const;

  // Same as dtypeOfMlirType(getElementType()).
  DType getDType() const;

  ShapedType getType() const;

  Type getElementType() const { return getType().getElementType(); }
  ArrayRef<int64_t> getShape() const { return getType().getShape(); }
  int64_t getRank() const { return getType().getRank(); }
  int64_t getNumElements() const { return getType().getNumElements(); }

  //===----------------------------------------------------------------------===//
  // Iteration:
  //
  // Use value_begin<X>(), value_end(), or getValues<X>() to iterate over the
  // the elements, where X can be any scalar cpp type, or APFloat (if element
  // type is floating point) or APInt (if element type is integer).
  //
  // Note that iteration is slow because it invokes getReader() for every
  // element and, furthermore, performs a slow calculation from flat index to
  // buffer position if the underlying buffer is not contiguous, namely when its
  // strides are not the default strides for the type shape. It's more efficient
  // to copy out data in bulk with readWideNums().
  //===----------------------------------------------------------------------===//
private:
  using IndexIterator = llvm::iota_range<size_t>::const_iterator;

  template <typename X>
  using IndexToX = std::function<X(size_t)>;

public:
  // All the iterable types are listed as NonContiguous here as no type
  // is guaranteed to be represented contiguously in the underlying buffer
  // because of strides and the possibility that bufferDType != dtype.
  using NonContiguousIterableTypesT =
      std::tuple<Attribute, IntegerAttr, FloatAttr, APInt, APFloat, WideNum,
          bool, int8_t, uint8_t, int16_t, int8_t, int32_t, uint32_t, int64_t,
          uint64_t, onnx_mlir::float_16, onnx_mlir::bfloat_16, float, double>;

  template <typename X>
  using iterator = llvm::mapped_iterator<IndexIterator, IndexToX<X>>;

  template <typename X>
  using iterator_range = llvm::iterator_range<iterator<X>>;

  // This implementation enables the value_begin() and getValues() methods
  // from the ElementsAttr interface, for the NonContiguousIterableTypesT types.
  template <typename X>
  FailureOr<iterator<X>> try_value_begin_impl(OverloadToken<X>) const;

  template <typename X>
  iterator<X> value_end() const {
    return getValues<X>().end();
  }

  //===----------------------------------------------------------------------===//
  // Other access to the elements:
  //===----------------------------------------------------------------------===//
private:
  StringRef getBufferString() const;

  ArrayRef<char> getBufferBytes() const;

  // Warning: this is somewhat inefficient because it invokes getReader().
  // It's more efficient to copy out data in bulk with readWideNums().
  WideNum readBufferPos(size_t pos) const;

  // Warning: this is inefficient unless isContiguous() or isSplat().
  WideNum readFlatIndex(size_t flatIndex) const;

  // Warning: this is inefficient because it calls unflattenIndex on flatIndex.
  size_t flatIndexToBufferPos(size_t flatIndex) const;

  onnx_mlir::ArrayBuffer<WideNum> getBufferAsWideNums() const;

public:
  WideNum getSplatWideNum() const;

  template <typename X>
  X getSplatValue() const;

  // Copies out the elements in a flat array in row-major order.
  void readWideNums(MutableArrayRef<WideNum> dst) const;

  // Returns a pointer to the underlying data, if everything aligns,
  // otherwise makes and returns a copy.
  onnx_mlir::ArrayBuffer<WideNum> getWideNums() const;

  // Copies out the elements in a flat array in row-major order.
  // If the element type is bool the data holds one byte (with value 0 or 1) per
  // bool (contrary to how DenseElementsAttr::getRawData() bit packs bools).
  void readRawBytes(MutableArrayRef<char> dst) const;

  // Returns a pointer to the underlying data, if everything aligns,
  // otherwise makes and returns a copy.
  // If the element type is bool the data holds one byte (with value 0 or 1) per
  // bool (contrary to how DenseElementsAttr::getRawData() bit packs bools).
  onnx_mlir::ArrayBuffer<char> getRawBytes() const;

  template <typename X>
  onnx_mlir::ArrayBuffer<X> getArray() const;

  void printWithoutType(AsmPrinter &printer) const;

private:
  // TODO: move the following to ElementsAttrBuilder

  using Transformer = std::function<void(MutableArrayRef<WideNum>)>;

  DisposableElementsAttr transform(onnx_mlir::ElementsAttrBuilder &elmsBuilder,
      Type transformedElementType, Transformer transformer) const;

  DisposableElementsAttr castElementType(
      onnx_mlir::ElementsAttrBuilder &elmsBuilder, Type newElementType) const;

  DisposableElementsAttr transpose(onnx_mlir::ElementsAttrBuilder &elmsBuilder,
      ArrayRef<uint64_t> perm) const;

  DisposableElementsAttr reshape(onnx_mlir::ElementsAttrBuilder &elmsBuilder,
      ArrayRef<int64_t> reshapedShape) const;

  // Broadcasts like the ONNX Expand op.
  DisposableElementsAttr expand(onnx_mlir::ElementsAttrBuilder &elmsBuilder,
      ArrayRef<int64_t> expandedShape) const;

}; // class DisposableElementsAttr

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//
// TODO: move so standalone DisposableElementsAttr.inc source file
//       like ShapeHelper.inc
//===----------------------------------------------------------------------===//

namespace detail {
// True for the types T in DisposableElementsAttr::NonContiguousIterableTypesT.
template <typename T>
constexpr bool isIterableType =
    std::is_same_v<T, Attribute> || std::is_same_v<T, IntegerAttr> ||
    std::is_same_v<T, FloatAttr> || std::is_same_v<T, APInt> ||
    std::is_same_v<T, APFloat> || std::is_same_v<T, onnx_mlir::WideNum> ||
    (onnx_mlir::CppTypeTrait<T>::dtype != onnx_mlir::DType::UNDEFINED &&
        onnx_mlir::CppTypeTrait<T>::isIntOrFloat);

// Supports all the types T in NonContiguousIterableTypesT.
template <typename T>
T getNumber(Type elementType, onnx_mlir::DType tag, onnx_mlir::WideNum n) {
  static_assert(isIterableType<T>);
  if constexpr (std::is_same_v<T, Attribute>)
    if (isFloatDType(tag))
      return FloatAttr::get(elementType, n.toAPFloat(tag));
    else
      return IntegerAttr::get(elementType, n.toAPInt(tag));
  else if constexpr (std::is_same_v<T, IntegerAttr>)
    return IntegerAttr::get(elementType, n.toAPInt(tag)); // fails if float
  else if constexpr (std::is_same_v<T, FloatAttr>)
    return FloatAttr::get(elementType, n.toAPFloat(tag)); // fails if !float
  else if constexpr (std::is_same_v<T, APInt>)
    return n.toAPInt(tag); // fails if isFloatDType(tag)
  else if constexpr (std::is_same_v<T, APFloat>)
    return n.toAPFloat(tag); // fails unless isFloatDType(tag)
  else if constexpr (std::is_same_v<T, onnx_mlir::WideNum>)
    return n;
  else
    return n.to<T>(tag);
}
} // namespace detail

template <typename X>
inline X DisposableElementsAttr::getSplatValue() const {
  return detail::getNumber<X>(getElementType(), getDType(), getSplatWideNum());
}

template <typename X>
inline auto DisposableElementsAttr::try_value_begin_impl(OverloadToken<X>) const
    -> FailureOr<iterator<X>> {
  if constexpr (detail::isIterableType<X>) {
    DType dtype = getDType();
    if constexpr (std::is_same_v<X, llvm::APFloat>) {
      if (!isFloatDType(dtype))
        return failure();
    } else if constexpr (std::is_same_v<X, llvm::APInt>) {
      if (isFloatDType(dtype))
        return failure();
    }
    // Translate "this" to a DisposableElementsAttr to work around that "this"
    // becomes something strange as we wind our way to try_value_begin_impl()
    // via interfaces from the original call to this->value_end()/getValues().
    DisposableElementsAttr attr = *this;
    auto range = llvm::seq<size_t>(0, getNumElements());
    return iterator<X>(range.begin(), [dtype, attr](size_t flatIndex) -> X {
      WideNum n = attr.readFlatIndex(flatIndex);
      return detail::getNumber<X>(attr.getElementType(), dtype, n);
    });
  } else {
    return failure();
  }
}

template <typename X>
inline onnx_mlir::ArrayBuffer<X> DisposableElementsAttr::getArray() const {
  assert(onnx_mlir::toDType<X> == getDType());
  if (!isTransformedOrCast() && isContiguous())
    return onnx_mlir::castArrayRef<X>(getBufferBytes());
  typename onnx_mlir::ArrayBuffer<X>::Vector vec;
  vec.resize_for_overwrite(getNumElements());
  readRawBytes(onnx_mlir::castMutableArrayRef<char>(makeMutableArrayRef(vec)));
  return std::move(vec);
}

} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)
