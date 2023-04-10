/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- DisposableElementsAttributeStorage.hpp ---------------===//
//
// Storage for DisposableElementsAttr. For information hiding purposes this
// should not be included by users of DisposableElementsAttr. It is needed for
// the implementation DisposableElementsAttr itself and ONNXDialect needs it for
// dialect registration.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

#include "mlir/IR/AttributeSupport.h"

using namespace onnx_mlir;

namespace mlir {

struct DisposableElementsAttributeStorage : public AttributeStorage {
  using Buffer = std::shared_ptr<llvm::MemoryBuffer>;
  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;
  using KeyTy = std::tuple<ShapedType, ArrayRef<int64_t>, onnx_mlir::BType,
      onnx_mlir::BType, bool, size_t>;
  static constexpr int TYPE = 0;
  static constexpr int STRIDES = 1;
  static constexpr int BUFFER_BTYPE = 2;
  static constexpr int BTYPE = 3;
  static constexpr int IS_CONTIGUOUS = 4;
  static constexpr int ID = 5;

  // Constructs only type and strides and properties while the caller sets
  // buffer and transformer after construction to minimize copying.
  DisposableElementsAttributeStorage(ShapedType type, ArrayRef<int64_t> strides,
      onnx_mlir::BType bufferBType, onnx_mlir::BType btype, bool isContiguous,
      size_t id)
      : type(type), strides(strides), bufferBType(bufferBType), btype(btype),
        isContiguous(isContiguous), id(id) {}

  // Equality and hashKey are engineered to defeat the storage uniquer.
  // We don't want uniqueing because we can't compare transformers for equality
  // and we could be in a sitation later where we have the same data or the
  // same buffer address but there is an undetectable mismatch because the
  // buffer and transformer were disposed by garbage collection.
  bool operator==(const KeyTy &key) const { return id == std::get<ID>(key); }
  static llvm::hash_code hashKey(const KeyTy &key) { return std::get<ID>(key); }

  static DisposableElementsAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    ShapedType type = std::get<TYPE>(key);
    ArrayRef<int64_t> strides = std::get<STRIDES>(key);
    onnx_mlir::BType bufferBType = std::get<BUFFER_BTYPE>(key);
    onnx_mlir::BType btype = std::get<BTYPE>(key);
    bool isContiguous = std::get<IS_CONTIGUOUS>(key);
    size_t id = std::get<ID>(key);
    return new (allocator.allocate<DisposableElementsAttributeStorage>())
        DisposableElementsAttributeStorage(type, allocator.copyInto(strides),
            bufferBType, btype, isContiguous, id);
  }

  // The tensor shape and element type that this object represents. The
  // underlying data in buffer may not directly match the type's element type
  // or number of elements, depending on bufferBType, transformer, and strides.
  ShapedType type;

  // Specifies how to map positions expressed in type's shape to the flat
  // indices in buffer. strides can express that buffer is not in the default
  // row-major order (maybe as a result of a transpose) or requires broadcast
  // to fill in type's shape. A special case is when the buffer holds a single
  // splat value that broadcasts to shape's size with all-zero strides.
  ArrayRef<int64_t> strides;

  // Data type of the elements in buffer before transform.
  onnx_mlir::BType bufferBType;

  // Data type (BOOL, INT8, FLOAT16, etc) of the type's elements.
  // Redundant as btype == btypeOfMlirType(type.getElementType())
  // but we store it for fast access.
  onnx_mlir::BType btype;

  // Do the strides match the type's shape?
  // Redundant as isContiguous == areStridesContiguous(type.getShape(), strides)
  // but we store it for fast access.
  bool isContiguous;

  // Serial number that distinguishes instances.
  size_t id;

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
  // element type. Is null if data is not transformed.
  // In this case the buffer data type and the type's element type must promote
  // to the same double/i64/u64 widetype
  // (both are float, or both are signed ints, or both are unsigned ints).
  //
  // Garbage collection clears the transformer when the DisposableElementsAttr
  // is disposed.
  Transformer transformer;
};

} // namespace mlir