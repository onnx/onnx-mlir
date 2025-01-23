/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- DisposableElementsAttr.hpp --------------------===//
//
// DisposableElementsAttr, garbage collectible alternative to DenseElementsAttr.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DISPOSABLE_ELEMENTS_ATTR_H
#define ONNX_MLIR_DISPOSABLE_ELEMENTS_ATTR_H

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <functional>
#include <memory>

namespace onnx_mlir {
class DisposablePool;
class ElementsAttrBuilder;
} // namespace onnx_mlir

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
// DisposableElementsAttr and DenseElementsAttr interchangeably, and the
// ONNXConstantOp custom assembly format prints DisposableElementsAttr the
// same as DenseElementsAttr so we can switch between them without changing
// lit tests.
//
// DisposableElementsAttr instances can only be constructed with DiposablePool
// which tracks all instances and garbage collects unreachable instances
// between compiler passes.
//
// NOTE: DisposablePool bypasses the storage uniquer and creates a unique
//       underlying DisposableElementsAttributeStorage every time it constructs
//       a DisposableElementsAttr instance.
//       See the explanation in DisposableElementsAttributeStorage.hpp.
//
// NOTE: DenseResourceElementsAttr is an alternative for heap allocated memory
//       (but without garbage collection or the other features listed above).
//
// NOTE: DisposableElementsAttr doesn't support complex numbers and strings.
//       It could be extended with 'DisposableStringElementsAttr` and
//       `DisposableComplexElementsAttr' in the same way that
//       DenseElementsAttr has different implementations for strings and
//       numbers.
//
class DisposableElementsAttr
    : public Attribute::AttrBase<DisposableElementsAttr, Attribute,
          DisposableElementsAttributeStorage, TypedAttr::Trait,
          ElementsAttr::Trait> {
  using Base::Base;

  // BType and WideNum are ubiquitous in the class definition and these using
  // statements are convenient as they let us omit their namespace qualifier.
  using BType = onnx_mlir::BType;
  using WideNum = onnx_mlir::WideNum;

  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

public:
  // DisposablePool needs access to the private create(), getId(), and dispose()
  // methods to create, track, and dispose instances.
  friend class onnx_mlir::DisposablePool;

  // ElementsAttrBuilder needs access to buffer, transformer, and metadata to
  // construct new instances with new metadata around the same underlying data.
  friend class onnx_mlir::ElementsAttrBuilder;

  static constexpr const char *name = "disposable.attr";

  //===--------------------------------------------------------------------===//
  // Instantiation:
  //===--------------------------------------------------------------------===//
  DisposableElementsAttr(std::nullptr_t) {}

  // Allow implicit conversion to ElementsAttr.
  operator ElementsAttr() const {
    return *this ? mlir::cast<ElementsAttr>(*this) : nullptr;
  }

  // Clears the buffer payload shared_ptr which decreases the reference count
  // and, if it reaches zero, frees or closes the underlying MemoryBuffer's
  // heap allocation or file. Called from DisposablePool.
  void dispose();

private:
  // Called from DisposablePool who calls with a unique id and records the
  // created instance.
  static DisposableElementsAttr create(ShapedType type, size_t id,
      BType bufferBType, ArrayRef<int64_t> strides, const Buffer &buffer,
      Transformer transformer);

public:
  //===--------------------------------------------------------------------===//
  // Instance properties:
  //===--------------------------------------------------------------------===//

  // isSplat() is true if all elements are known to be the same
  // (and are represented as a single number with all-zeros strides).
  // Can return false even if all elements are identical.
  bool isSplat() const;

  // Same as btypeOfMlirType(getElementType()).
  BType getBType() const;

  ShapedType getType() const;

  Type getElementType() const { return getType().getElementType(); }
  ArrayRef<int64_t> getShape() const { return getType().getShape(); }
  int64_t getRank() const { return getType().getRank(); }
  int64_t getNumElements() const { return getType().getNumElements(); }

private:
  bool isDisposed() const;

  size_t getId() const;

  ArrayRef<int64_t> getStrides() const;

  const Buffer &getBuffer() const;

  const Transformer &getTransformer() const;

  bool isContiguous() const;

  bool isTransformed() const;

  bool isTransformedOrCast() const;

  BType getBufferBType() const;

  unsigned getBufferElementBytewidth() const;

  int64_t getNumBufferElements() const;

public:
  //===--------------------------------------------------------------------===//
  // Iteration:
  //
  // Use value_begin<X>(), value_end(), or getValues<X>() to iterate over the
  // the elements, where X can be any scalar cpp type, or APFloat (if element
  // type is floating point) or APInt (if element type is integer).
  //
  // Note that iteration is slow because it invokes getTransformer() for every
  // element and, furthermore, performs a slow calculation from flat index to
  // buffer position if the underlying buffer is not contiguous, namely when its
  // strides are not the default strides for the type shape. It's more efficient
  // to copy out data in bulk with getArray/WideNums() and readArray/WideNums().
  //===--------------------------------------------------------------------===//

  // All the iterable types are listed as NonContiguous here as no type
  // is guaranteed to be represented contiguously in the underlying buffer
  // because of strides and the possibility that bufferBType != btype.
  using NonContiguousIterableTypesT =
      std::tuple<Attribute, IntegerAttr, FloatAttr, APInt, APFloat, WideNum,
          bool, int8_t, uint8_t, int16_t, int8_t, int32_t, uint32_t, int64_t,
          uint64_t, onnx_mlir::float_16, onnx_mlir::bfloat_16, float, double>;

  // An underlying iota_range sequence iterator returns size_t flat indices
  // which are mapped to elements of type X by a function<X(size_t)>.
  // (Same construction as in mlir::SparseElementsAttr.)
  template <typename X>
  using iterator =
      llvm::mapped_iterator<llvm::iota_range<size_t>::const_iterator,
          std::function<X(size_t)>>;

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

  //===--------------------------------------------------------------------===//
  // Other access to the elements:
  //
  // getArray<X>() and getWideNums() return the elements as their C++ type X
  // or "widened" (promoted) to type WideNum, as an ArrayBuffer which either has
  // an array reference to the underlying data, if it happens to be of that type
  // and contiguous and not transformed, or owns a copy of the contiguous and
  // transformed data. The caller can call the ref() method on the ArrayBuffer
  // to access the data as an ArrayRef while it is in scope.
  //
  // readArray<X>(dst) and readWideNums(dst) copy the elements into the dst
  // array.
  //
  // For example, if data, indices, and updates are of type
  // DisposableElementsAttr then ScatterNDElementsAttr(data, indices, updates)
  // can be implemented with:
  //
  //   return ElementsAttrBuilder(ctx).fromWideNums(data.getType(),
  //     [&](MutableArrayRef<WideNum> dst) {
  //       data.readWideNums(dst);
  //       ArrayBuffer<WideNum> indicesData = indices.getWideNums();
  //       ArrayBuffer<WideNum> updatesData = updates.getWideNums();
  //
  //       // Write updatesData.get() into dst according to indicesData.get()...
  //
  //       // ArrayBuffers end of lexical scope: any data copies are freed.
  //     });
  //===--------------------------------------------------------------------===//
  template <typename X>
  X getSplatValue() const;

  // Returns a pointer to the underlying data as a flat aray, if
  // everything aligns, otherwise makes and returns a copy.
  // Presents the same view of the data as GetValues() but as a
  // contiguous array and only for the type X matching the element type.
  // Precondition: X must correspond to getElementType().
  template <typename X>
  onnx_mlir::ArrayBuffer<X> getArray() const;

  // Copies out the elements in a flat array in row-major order.
  // Precondition: X must correspond to getElementType().
  template <typename X>
  void readArray(MutableArrayRef<X> dst) const;

  // Returns a pointer to the underlying data as a flat byte array, if
  // everything aligns, otherwise makes and returns a copy.
  // If the element type is bool the data holds one byte (with value 0 or 1) per
  // bool (contrary to how DenseElementsAttr::getRawData() bit packs bools).
  onnx_mlir::ArrayBuffer<char> getRawBytes() const;

  // Returns a pointer to the underlying data as a flat WideNum array, if
  // everything aligns, otherwise makes and returns a copy.
  onnx_mlir::ArrayBuffer<WideNum> getWideNums() const;

  // Copies out the elements in a flat WideNum array in row-major order.
  void readWideNums(MutableArrayRef<WideNum> dst) const;

  // Makes deep copy.
  DenseElementsAttr toDenseElementsAttr() const;

  static constexpr StringLiteral getMnemonic() { return {"dense_disposable"}; }

  // Returns the underlying data as a flat byte array in row-major order.
  // If the element type is bool the data holds one byte (with value 0 or 1) per
  // bool (contrary to how DenseElementsAttr::getRawData() bit packs bools).
  static std::unique_ptr<llvm::MemoryBuffer> parse(
      AsmParser &parser, ShapedType type);

  void printWithoutType(AsmPrinter &printer) const;

  void printAsDenseElementsAttr(AsmPrinter &printer) const;

private:
  // Widens and transforms bytes into WideNums in accordance with
  // bufferDType and transformer.
  void readBytesAsWideNums(
      ArrayRef<char> bytes, llvm::MutableArrayRef<WideNum>) const;

  // Similar to DenseElementsAttr::getRawData() in that it returns the
  // underlying raw data, but the data representation can be further removed
  // from the view presented by getValues() and getArray() because of
  // strides, bufferDType cast, and transformer. Another difference is that for
  // element type bool the data holds one byte (with value 0 or 1) per bool.
  ArrayRef<char> getBufferBytes() const;

  onnx_mlir::ArrayBuffer<WideNum> getBufferAsWideNums() const;

  // Warning: This is inefficient. First, it calculates the buffer position from
  // strides with divisions and modulo, unless isContiguous() or isSplat().
  // Second, it widens the buffer data type and computes any transformation for
  // a single element without the fast inner loop of getArray/WideNums() and
  // readArray/WideNums(),
  // which read out all elements in bulk with faster amortized speed per
  // element.
  WideNum atFlatIndex(size_t flatIndex) const;

  // Warning: This is inefficient because it calls unflattenIndex on flatIndex.
  size_t flatIndexToBufferPos(size_t flatIndex) const;

  // Copies out the elements in a flat array in row-major order.
  // If the element type is bool the data holds one byte (with value 0 or 1) per
  // bool (contrary to how DenseElementsAttr::getRawData() bit packs bools).
  void readRawBytes(MutableArrayRef<char> dst) const;

}; // class DisposableElementsAttr

// Include template implementations.
#include "DisposableElementsAttr.hpp.inc"

} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::DisposableElementsAttr)
#endif
