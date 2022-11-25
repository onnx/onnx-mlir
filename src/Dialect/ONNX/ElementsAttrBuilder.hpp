/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/WideNum.hpp"

#include <atomic>

namespace onnx_mlir {

class ElementsAttrBuilder {
public:
  ElementsAttrBuilder(DisposablePool &disposablePool);

  ElementsAttrBuilder(mlir::MLIRContext *context);

  mlir::DisposableElementsAttr fromMemoryBuffer(
      mlir::ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // TODO: Remove this. Only used in tests.
  mlir::DisposableElementsAttr fromSplatMemoryBuffer(
      mlir::ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf);

  // Makes a DisposableElementsAttr that points to elements' raw data if
  // elements is DenseElementsAttr, except if the element type is bool, then
  // it makes a deep copy because DisposableElementsAttr doesn't bit pack bools.
  mlir::DisposableElementsAttr fromElementsAttr(mlir::ElementsAttr elements);

  template <typename T>
  using Filler = std::function<void(llvm::MutableArrayRef<T>)>;

  mlir::DisposableElementsAttr fromRawBytes(mlir::ShapedType type,
      BType bufferBType, llvm::ArrayRef<char> bytes, bool mustCopy);

  mlir::DisposableElementsAttr fromRawBytes(mlir::ShapedType type,
      BType bufferBType, const Filler<char> &bytesFiller);

  mlir::DisposableElementsAttr fromWideNums(
      mlir::ShapedType type, llvm::ArrayRef<WideNum> wideData, bool mustCopy);

  mlir::DisposableElementsAttr fromWideNums(
      mlir::ShapedType type, const Filler<WideNum> &wideDataFiller);

  template <typename T>
  mlir::DisposableElementsAttr fromArray(
      mlir::ShapedType type, llvm::ArrayRef<T> array, bool mustCopy);

  template <typename T>
  mlir::DisposableElementsAttr fromArray(
      mlir::ShapedType type, const Filler<T> &arrayFiller);

  using Transformer = mlir::DisposableElementsAttr::Transformer;

  template <typename UnaryFunction = std::function<WideNum(WideNum)>>
  static Transformer functionTransformer(UnaryFunction fun);

  mlir::DisposableElementsAttr transform(mlir::DisposableElementsAttr elms,
      mlir::Type transformedElementType, Transformer transformer);

  template <typename BinaryCombiner = std::function<WideNum(WideNum, WideNum)>>
  mlir::DisposableElementsAttr combine(mlir::DisposableElementsAttr lhs,
      mlir::DisposableElementsAttr rhs, mlir::ShapedType combinedType,
      BinaryCombiner combiner);

  mlir::DisposableElementsAttr castElementType(
      mlir::DisposableElementsAttr elms, mlir::Type newElementType);

  mlir::DisposableElementsAttr transpose(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<uint64_t> perm);

  mlir::DisposableElementsAttr reshape(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> reshapedShape);

  // Broadcasts like the ONNX Expand op.
  mlir::DisposableElementsAttr expand(
      mlir::DisposableElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape);

  mlir::DisposableElementsAttr transformAndExpand(
      mlir::DisposableElementsAttr elms, mlir::ShapedType resultType,
      Transformer transformer) {
    auto transformed =
        transform(elms, resultType.getElementType(), transformer);
    return expand(transformed, resultType.getShape());
  }

  // Called only from DisposableElementsAttr internal builder methods.
  // TODO: Remove this and move those methods into ElementsAttrBuilder.
  mlir::DisposableElementsAttr cloneMemoryBuffer(mlir::ShapedType type,
      const mlir::DisposableElementsAttr::Buffer &buffer,
      mlir::DisposableElementsAttr::Strides strides, BType bufferBType,
      mlir::DisposableElementsAttr::Reader reader);

private:
  // Create a DisposableElementsAttr and put it in the pool.
  template <typename... Args>
  mlir::DisposableElementsAttr create(mlir::ShapedType type, Args &&... args);

  static std::atomic<size_t> counter;

  DisposablePool &disposablePool;
};

//===----------------------------------------------------------------------===//
// Deferred Method Definitions
//
// TODO: move so standalone ElementsAttrBuilder.inc source file
//       like ShapeHelper.inc
//===----------------------------------------------------------------------===//

template <typename T>
mlir::DisposableElementsAttr ElementsAttrBuilder::fromArray(
    mlir::ShapedType type, llvm::ArrayRef<T> array, bool mustCopy) {
  return fromRawBytes(type, toBType<T>, castArrayRef<char>(array), mustCopy);
}

template <typename T>
mlir::DisposableElementsAttr ElementsAttrBuilder::fromArray(
    mlir::ShapedType type, const Filler<T> &arrayFiller) {
  return fromRawBytes(
      type, toBType<T>, [&arrayFiller](llvm::MutableArrayRef<char> bytes) {
        arrayFiller(castMutableArrayRef<T>(bytes));
      });
}

/*static*/
template <typename UnaryFunction>
ElementsAttrBuilder::Transformer ElementsAttrBuilder::functionTransformer(
    UnaryFunction fun) {
  return [fun = std::forward<UnaryFunction>(fun)](
             llvm::MutableArrayRef<WideNum> data) -> void {
    for (WideNum &n : data)
      n = fun(n);
  };
}

template <typename BinaryCombiner>
mlir::DisposableElementsAttr ElementsAttrBuilder::combine(
    mlir::DisposableElementsAttr lhs, mlir::DisposableElementsAttr rhs,
    mlir::ShapedType combinedType, BinaryCombiner combiner) {
  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.getSplatWideNum();
    return transformAndExpand(rhs, combinedType,
        functionTransformer(
            [lhsNum, combiner = std::forward<BinaryCombiner>(combiner)](
                WideNum n) { return combiner(lhsNum, n); }));
  }
  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.getSplatWideNum();
    return transformAndExpand(lhs, combinedType,
        functionTransformer(
            [rhsNum, combiner = std::forward<BinaryCombiner>(combiner)](
                WideNum n) { return combiner(n, rhsNum); }));
  }

  auto shape = combinedType.getShape();
  auto lhsStrides = expandStrides(lhs.getStrides(), shape);
  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> lhsStrided{lhsStrides, lhsNums.get()};
  auto rhsStrides = expandStrides(rhs.getStrides(), shape);
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> rhsStrided{rhsStrides, rhsNums.get()};
  return fromWideNums(
      combinedType, [&](llvm::MutableArrayRef<WideNum> dstNums) {
        auto dstStrides = getDefaultStrides(shape);
        Strided<llvm::MutableArrayRef<WideNum>> dstStrided{dstStrides, dstNums};
        transformAndRestrideTwoWideArrays(
            shape, lhsStrided, rhsStrided, dstStrided, combiner);
      });
}

} // namespace onnx_mlir
