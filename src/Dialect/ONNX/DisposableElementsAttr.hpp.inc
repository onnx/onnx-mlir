//===------------------- DisposableElementsAttr.hpp.inc -------------------===//
//
//===----------------------------------------------------------------------===//
// DisposableElementsAttr template implementations
//===----------------------------------------------------------------------===//

namespace detail {
// True for the types T in DisposableElementsAttr::NonContiguousIterableTypesT.
template <typename T>
constexpr bool isIterableType =
    std::is_same_v<T, Attribute> || std::is_same_v<T, IntegerAttr> ||
    std::is_same_v<T, FloatAttr> || std::is_same_v<T, APInt> ||
    std::is_same_v<T, APFloat> || std::is_same_v<T, onnx_mlir::WideNum> ||
    (onnx_mlir::CppTypeTrait<T>::btype != onnx_mlir::BType::UNDEFINED &&
        onnx_mlir::CppTypeTrait<T>::isIntOrFloat);

// Supports all the types T in NonContiguousIterableTypesT.
template <typename T>
T getNumber(Type elementType, onnx_mlir::BType tag, onnx_mlir::WideNum n) {
  static_assert(isIterableType<T>);
  (void)elementType; // Suppresses compiler warning.
  (void)tag;         // Suppresses compiler warning.
  if constexpr (std::is_same_v<T, Attribute>)
    if (isFloatBType(tag))
      return FloatAttr::get(elementType, n.toAPFloat(tag));
    else
      return IntegerAttr::get(elementType, n.toAPInt(tag));
  else if constexpr (std::is_same_v<T, IntegerAttr>)
    return IntegerAttr::get(elementType, n.toAPInt(tag)); // fails if float
  else if constexpr (std::is_same_v<T, FloatAttr>)
    return FloatAttr::get(elementType, n.toAPFloat(tag)); // fails if !float
  else if constexpr (std::is_same_v<T, APInt>)
    return n.toAPInt(tag); // fails if isFloatBType(tag)
  else if constexpr (std::is_same_v<T, APFloat>)
    return n.toAPFloat(tag); // fails unless isFloatBType(tag)
  else if constexpr (std::is_same_v<T, onnx_mlir::WideNum>)
    return n;
  else
    return n.to<T>(tag);
}
} // namespace detail

template <typename X>
inline X DisposableElementsAttr::getSplatValue() const {
  return detail::getNumber<X>(getElementType(), getBType(), getSplatWideNum());
}

template <typename X>
inline auto DisposableElementsAttr::try_value_begin_impl(OverloadToken<X>) const
    -> FailureOr<iterator<X>> {
  if constexpr (detail::isIterableType<X>) {
    BType btype = getBType();
    if constexpr (std::is_same_v<X, llvm::APFloat>) {
      if (!isFloatBType(btype))
        return failure();
    } else if constexpr (std::is_same_v<X, llvm::APInt>) {
      if (isFloatBType(btype))
        return failure();
    }
    // Translate "this" to a DisposableElementsAttr to work around that "this"
    // becomes something strange as we wind our way to try_value_begin_impl()
    // via interfaces from the original call to this->value_end()/getValues().
    DisposableElementsAttr attr = *this;
    auto range = llvm::seq<size_t>(0, getNumElements());
    return iterator<X>(range.begin(), [btype, attr](size_t flatIndex) -> X {
      WideNum n = attr.readFlatIndex(flatIndex);
      return detail::getNumber<X>(attr.getElementType(), btype, n);
    });
  } else {
    return failure();
  }
}

template <typename X>
inline onnx_mlir::ArrayBuffer<X> DisposableElementsAttr::getArray() const {
  assert(onnx_mlir::toBType<X> == getBType());
  if (!isTransformedOrCast() && isContiguous())
    return onnx_mlir::castArrayRef<X>(getBufferBytes());
  typename onnx_mlir::ArrayBuffer<X>::Vector vec;
  vec.resize_for_overwrite(getNumElements());
  readRawBytes(onnx_mlir::castMutableArrayRef<char>(makeMutableArrayRef(vec)));
  return std::move(vec);
}
