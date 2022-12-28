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
  mlir::DisposableElementsAttr fromMemoryBuffer(
      mlir::ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Wraps elements in a DisposableElementsAttr if it isn't already a
  // DisposableElementsAttr. If elements is DenseElementsAttr the wrapper
  // points into elements' raw data, except if the element type is bool, then
  // a deep copy is made that unpacks the bits because DisposableElementsAttr
  // doesn't bit pack bools.
  mlir::DisposableElementsAttr fromElementsAttr(mlir::ElementsAttr elements);

  template <typename T>
  using Filler = std::function<void(llvm::MutableArrayRef<T>)>;

  // Constructs a DisposableElementsAttr and calls wideDataFiller to populate
  // its memory buffer.
  mlir::DisposableElementsAttr fromWideNums(
      mlir::ShapedType type, const Filler<WideNum> &wideDataFiller);

  // Constructs a DisposableElementsAttr and calls arrayFiller to populate
  // its memory buffer.
  template <typename T>
  mlir::DisposableElementsAttr fromArray(
      mlir::ShapedType type, const Filler<T> &arrayFiller);

  // A transformer mutates elements.
  using Transformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

  // Constructs a transformer that changes every element to the result of
  // applying the given function to the element.
  template <typename UnaryFunction = std::function<WideNum(WideNum)>>
  static Transformer functionTransformer(UnaryFunction fun);

  // Returns a DisposableElementsAttr where each element is transformed
  // by running the given transformer on all the elements.
  mlir::DisposableElementsAttr transform(mlir::DisposableElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

  // Returns a DisposableElementsAttr that is the result of applying
  // a binary function pairwise on the elements lhs and rhs after broadcast
  // to combinedType.
  template <typename BinaryCombiner = std::function<WideNum(WideNum, WideNum)>>
  mlir::DisposableElementsAttr combine(mlir::DisposableElementsAttr lhs,
      mlir::DisposableElementsAttr rhs, mlir::ShapedType combinedType,
      BinaryCombiner combiner);

  // Returns a DisposableElementsAttr with the elements cast to the given
  // newElementType.
  mlir::DisposableElementsAttr castElementType(
      mlir::DisposableElementsAttr elms, mlir::Type newElementType);

  // Returns a transposed DisposableElementsAttr.
  mlir::DisposableElementsAttr transpose(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  // Returns a reshaped DisposableElementsAttr.
  mlir::DisposableElementsAttr reshape(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  mlir::DisposableElementsAttr expand(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

private:
  mlir::DisposableElementsAttr fromRawBytes(
      mlir::ShapedType type, BType bufferBType, llvm::ArrayRef<char> bytes);

  mlir::DisposableElementsAttr fromRawBytes(mlir::ShapedType type,
      BType bufferBType, const Filler<char> &bytesFiller);

  mlir::DisposableElementsAttr transformAndExpand(
      mlir::DisposableElementsAttr elms, mlir::ShapedType resultType,
      Transformer transformer) {
    auto transformed =
        transform(elms, resultType.getElementType(), std::move(transformer));
    return expand(transformed, resultType.getShape());
  }

  // Create a DisposableElementsAttr and put it in disposablePool.
  mlir::DisposableElementsAttr create(mlir::ShapedType type, BType bufferBType,
      llvm::ArrayRef<int64_t> strides,
      const std::shared_ptr<llvm::MemoryBuffer> &buffer,
      Transformer transformer = nullptr);

  mlir::DisposableElementsAttr createWithDefaultStrides(mlir::ShapedType type,
      BType bufferBType, const std::shared_ptr<llvm::MemoryBuffer> &buffer,
      Transformer transformer = nullptr);

  mlir::DisposableElementsAttr createSplat(mlir::ShapedType type,
      BType bufferBType, std::unique_ptr<llvm::MemoryBuffer> membuf);

  DisposablePool &disposablePool;
};

// Include template implementations.
#include "ElementsAttrBuilder.hpp.inc"

} // namespace onnx_mlir
