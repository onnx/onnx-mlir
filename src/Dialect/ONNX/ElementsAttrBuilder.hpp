/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// ElementsAttrBuilder builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/BType.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/WideNum.hpp"

#include <functional>
#include <memory>

namespace onnx_mlir {

// Builds DisposableElementsAttr instances.
// Every instance is inserted into DisposablePool which garbage collects
// unreachable instances between compiler passes.
class ElementsAttrBuilder {
public:
  // Uses context to look up the ONNX dialect's DisposablePool for recording
  // DisposableElementsAttr instances created by the builder methods.
  ElementsAttrBuilder(mlir::MLIRContext *context);

  // Creates a DisposableElementsAttr instance backed by the data in membuf.
  // The created instance takes ownership of membuf and will release it when the
  // instance is disposed by garbage collection, unless it has shared membuf
  // with other DisposableElementsAttr instances that are longer lived.
  mlir::ElementsAttr fromMemoryBuffer(
      mlir::ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Wraps elements in a DisposableElementsAttr if it isn't already a
  // DisposableElementsAttr, provided the underlying DisposablePool is active.
  // If elements is DenseElementsAttr the wrapper points into elements' raw
  // data, except if the element type is bool, then a deep copy is made that
  // unpacks the bits because DisposableElementsAttr doesn't bit pack bools.
  mlir::DisposableElementsAttr toDisposableElementsAttr(
      mlir::ElementsAttr elements);

  // Converts elements to DenseElementsAttr if it's DisposableElementsAttr,
  // otherwise returns elements if it's already DenseElementsAttr.
  mlir::DenseElementsAttr toDenseElementsAttr(mlir::ElementsAttr elements);

  template <typename T>
  using Filler = std::function<void(llvm::MutableArrayRef<T>)>;

  // Constructs a DisposableElementsAttr and calls wideDataFiller to populate
  // its memory buffer.
  mlir::ElementsAttr fromWideNums(
      mlir::ShapedType type, const Filler<WideNum> &wideDataFiller);

  // Constructs a DisposableElementsAttr and calls arrayFiller to populate
  // its memory buffer.
  template <typename T>
  mlir::ElementsAttr fromArray(
      mlir::ShapedType type, const Filler<T> &arrayFiller) {
    return fromRawBytes(
        type, toBType<T>, [&arrayFiller](llvm::MutableArrayRef<char> bytes) {
          arrayFiller(castMutableArrayRef<T>(bytes));
        });
  }

  // A transformer mutates elements.
  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

  // Constructs a transformer that changes every element to the result of
  // applying the given function to the element.
  static Transformer functionTransformer(WideNum (*fun)(WideNum));

  // Returns an ElementsAttr where each element is transformed
  // by running the given transformer on all the elements.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr transform(mlir::ElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

  // Returns an ElementsAttr that is the result of applying a binary function
  // pairwise on the elements lhs and rhs after broadcast to combinedType.
  // Note that combiner is a plain function pointer to make it cheap and easy
  // to copy it into a transformer lambda in the returned ElementsAttr.
  //
  // Constructs new underlying data by applying the combiner, except in the
  // case where one of the arguments is splat, in that case reuses the other
  // argument's underlying data and just adds the necessary transformation
  // and broadcast.
  mlir::ElementsAttr combine(mlir::ElementsAttr lhs, mlir::ElementsAttr rhs,
      mlir::ShapedType combinedType, WideNum (*combiner)(WideNum, WideNum));

  // Returns an ElementsAttr with the elements cast to the given newElementType.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr castElementType(
      mlir::ElementsAttr elms, mlir::Type newElementType);

  // Returns a transposed ElementsAttr.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr transpose(
      mlir::ElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  // Returns a reshaped ElementsAttr.
  //
  // Reuses elms' underlying data without a data copy, unless the underlying
  // data is transposed in a way that requires the data to be "restrided".
  mlir::ElementsAttr reshape(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  //
  // Reuses elms' underlying data without a data copy.
  mlir::ElementsAttr expand(
      mlir::ElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

private:
  struct ElementsProperties;

  ElementsProperties getElementsProperties(mlir::ElementsAttr elements) const;

  mlir::ElementsAttr fromRawBytes(
      mlir::ShapedType type, BType bufferBType, llvm::ArrayRef<char> bytes);

  mlir::ElementsAttr fromRawBytes(mlir::ShapedType type, BType bufferBType,
      const Filler<char> &bytesFiller);

  mlir::ElementsAttr createWithDefaultStrides(mlir::ShapedType type,
      BType bufferBType, std::unique_ptr<llvm::MemoryBuffer> membuf);

  mlir::ElementsAttr createSplat(mlir::ShapedType type, BType bufferBType,
      std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Create a DisposableElementsAttr and put it in disposablePool.
  mlir::ElementsAttr create(mlir::ShapedType type, BType bufferBType,
      llvm::ArrayRef<int64_t> strides,
      const std::shared_ptr<llvm::MemoryBuffer> &buffer,
      Transformer transformer = nullptr);

  DisposablePool &disposablePool;
};

} // namespace onnx_mlir
