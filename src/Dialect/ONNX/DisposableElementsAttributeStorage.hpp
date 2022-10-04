/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- DisposableElementsAttributeStorage.hpp ---------------===//
//
// Storage for DisposableElementsAttr. For information hiding purposes this
// should not be included by users of DisposableElementsAttr. It is needed for
// the implementation DisposableElementsAttr itself, DisposablePool needs it
// to create instances, and ONNXDialect needs it for dialect registration.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"

#include "mlir/IR/AttributeSupport.h"

using namespace onnx_mlir;

namespace mlir {

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Strides = DisposableElementsAttr::Strides;
  using Properties = DisposableElementsAttr::Properties;
  using Buffer = DisposableElementsAttr::Buffer;
  using Reader = DisposableElementsAttr::Reader;
  using KeyTy = std::tuple<ShapedType, Strides, Properties>;

  // Constructs only type and strides and properties while the caller sets
  // buffer and reader after construction to minimize copying.
  DisposableElementsAttributeStorage(
      ShapedType type, Strides strides, Properties properties)
      : type(type), strides(strides), properties(properties) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare readers for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and reader were disposed by garbage collection.
  bool operator==(const KeyTy &key) const { return false; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    // Generates a unique number on each call to defeat the storage
    // uniquer.
    static std::atomic<size_t> counter{0};
    return ++counter;
  }

  static DisposableElementsAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    ShapedType type = std::get<0>(key);
    Strides strides = std::get<1>(key);
    Properties properties = std::get<2>(key);
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(
            type, allocator.copyInto(strides), properties);
  }

  // The tensor shape and element type that this object represents.
  // The underlying data in buffer may not directly match the type's element
  // type or number of elements, depending on strides and reader.
  ShapedType type;

  // Specifies how to map positions expressed in type's shape to the flat
  // indices in buffer. strides can express that buffer is not in the default
  // row-major order (maybe as a result of a transpose) or requires broadcast
  // to fill in type's shape. A special case is when the buffer holds a single
  // splat value that broadcasts to shape's size with all-zero strides.
  //
  // Strides cannot have leading zeros. Leading zeros are implicit. Pad with
  // leading zeros up to type's rank whenever you want explicit leading zeros
  // (see the padStrides(shape, strides) function in Support/Strides).
  Strides strides;

  Properties properties;

  // shared_ptr to an underlying MemoryBuffer which can be either heap allocated
  // or a mmap'ed file or point to the raw data of a DenseElementsAttr.
  //
  // The buffer elements' data type may not match type's element type, namely
  // when the transform function transforms the buffer data type to another
  // data type.
  //
  // Garbage collection clears the buffer when the DisposableElementsAttr is
  // disposed.
  //
  // Multiple DisposableElementsAttr can point to the same MemoryBuffer.
  // The MemoryBuffer is destroyed (and heap allocated data freed or mmap'ed
  // file closed) when no one points to it anymore.
  Buffer buffer;

  // Reads the buffer elements to WideNums corresponding to type's
  // element type. Is set to the identity reader function if data is not
  // transformed, namely when properties.isTransformed is false.
  // In this case the buffer data type and the type's element type must promote
  // to the same double/i64/u64 widetype
  // (both are float, or both are signed ints, or both are unsigned ints).
  //
  // Garbage collection clears the reader when the DisposableElementsAttr is
  // disposed.
  Reader reader;
};

} // namespace mlir