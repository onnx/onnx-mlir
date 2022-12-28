//===-------------------- ElementsAttrBuilder.hpp.inc ---------------------===//
//
//===----------------------------------------------------------------------===//
// ElementsAttrBuilder template implementations
//===----------------------------------------------------------------------===//

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
  return [fun = std::move(fun)](llvm::MutableArrayRef<WideNum> data) -> void {
    for (WideNum &n : data)
      n = fun(n);
  };
}

template <typename BinaryCombiner>
mlir::DisposableElementsAttr ElementsAttrBuilder::combine(
    mlir::DisposableElementsAttr lhs, mlir::DisposableElementsAttr rhs,
    mlir::ShapedType combinedType, BinaryCombiner combiner) {
  auto combinedShape = combinedType.getShape();
  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.getSplatWideNum();
    auto transformed = transform(rhs, combinedType.getElementType(),
        functionTransformer([lhsNum, combiner = std::move(combiner)](
                                WideNum n) { return combiner(lhsNum, n); }));
    return expand(transformed, combinedShape);
  }
  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.getSplatWideNum();
    auto transformed = transform(lhs, combinedType.getElementType(),
        functionTransformer([rhsNum, combiner = std::move(combiner)](
                                WideNum n) { return combiner(n, rhsNum); }));
    return expand(transformed, combinedShape);
  }
  auto lhsStrides = expandStrides(lhs.getStrides(), combinedShape);
  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> lhsStrided{lhsStrides, lhsNums.get()};
  auto rhsStrides = expandStrides(rhs.getStrides(), combinedShape);
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
  Strided<llvm::ArrayRef<WideNum>> rhsStrided{rhsStrides, rhsNums.get()};
  return fromWideNums(
      combinedType, [&](llvm::MutableArrayRef<WideNum> dstNums) {
        transformAndRestrideTwoWideArrays(
            combinedShape, lhsStrided, rhsStrided, dstNums, combiner);
      });
}
