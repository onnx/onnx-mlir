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
        transformAndRestrideTwoWideArrays(
            shape, lhsStrided, rhsStrided, dstNums, combiner);
      });
}
