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
mlir::ElementsAttr ElementsAttrBuilder::combine(mlir::ElementsAttr lhsElms,
    mlir::ElementsAttr rhsElms, mlir::ShapedType combinedType,
    BinaryCombiner combiner) {
  // TODO: Use getElementsProperties to avoid the construction of a
  //       DisposableElementsAttr if lhs or rhs is a DenseElementsAttr.
  mlir::DisposableElementsAttr lhs = toDisposableElementsAttr(lhsElms);
  mlir::DisposableElementsAttr rhs = toDisposableElementsAttr(rhsElms);

  auto combinedShape = combinedType.getShape();
  auto lhsStrides = expandStrides(lhs.getStrides(), combinedShape);
  auto rhsStrides = expandStrides(rhs.getStrides(), combinedShape);

  using Transformer = ElementsAttrBuilder::Transformer;
  auto composeTransformerFunction = [](Transformer transformer,
                                        auto fun) -> Transformer {
    return [transformer = std::move(transformer), fun = std::move(fun)](
               llvm::MutableArrayRef<WideNum> data) -> void {
      if (transformer != nullptr)
        transformer(data);
      for (WideNum &n : data)
        n = fun(n);
    };
  };

  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.atFlatIndex(0);
    return create(combinedType, rhs.getBufferBType(), rhsStrides,
        rhs.getBuffer(),
        composeTransformerFunction(rhs.getTransformer(),
            [lhsNum, combiner = std::move(combiner)](
                WideNum n) { return combiner(lhsNum, n); }));
  }

  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.atFlatIndex(0);
    return create(combinedType, lhs.getBufferBType(), lhsStrides,
        lhs.getBuffer(),
        composeTransformerFunction(lhs.getTransformer(),
            [rhsNum, combiner = std::move(combiner)](
                WideNum n) { return combiner(n, rhsNum); }));
  }

  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
  StridedArrayRef<WideNum> stridedLhs(lhsNums.get(), lhsStrides);
  StridedArrayRef<WideNum> stridedRhs(rhsNums.get(), rhsStrides);
  return fromWideNums(
      combinedType, [&](llvm::MutableArrayRef<WideNum> dstNums) {
        mapStrides<WideNum, WideNum, WideNum>(
            combinedShape, dstNums, stridedLhs, stridedRhs, combiner);
      });
}
