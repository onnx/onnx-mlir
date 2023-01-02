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
