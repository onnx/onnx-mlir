//===--------------------- ElementsAttrHelper.hpp.inc ---------------------===//
//
//===----------------------------------------------------------------------===//
// ElementsAttrHelper template implementations
//===----------------------------------------------------------------------===//

template <typename X>
ArrayBuffer<X> getElementsArray(mlir::ElementsAttr elms) {
  if (auto disposable = mlir::dyn_cast<mlir::DisposableElementsAttr>(elms))
    return disposable.getArray<X>();
  if (elms.isSplat())
    return
        typename ArrayBuffer<X>::Vector(elms.size(), elms.getSplatValue<X>());
  if (!elms.getElementType().isInteger(1)) {
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(elms)) {
      llvm::ArrayRef<X> data = castArrayRef<X>(dense.getRawData());
      assert(data.size() == static_cast<size_t>(elms.size()));
      return data;
    }
  }
  // elms is not dense or element type is bool, in either case
  // read the elements from getValues<X> the hard way:
  return typename ArrayBuffer<X>::Vector(elms.getValues<X>());
}

template <typename X>
void readElementsArray(mlir::ElementsAttr elms, llvm::MutableArrayRef<X> dst) {
  if (auto disposable = mlir::dyn_cast<mlir::DisposableElementsAttr>(elms))
    return disposable.readArray<X>(dst);
  if (elms.isSplat()) {
    assert(dst.size() == static_cast<size_t>(elms.size()));
    return std::fill(dst.begin(), dst.end(), elms.getSplatValue<X>());
  }
  if (!elms.getElementType().isInteger(1)) {
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(elms)) {
      llvm::ArrayRef<X> data = castArrayRef<X>(dense.getRawData());
      auto end = std::copy(data.begin(), data.end(), dst.begin());
      assert(end == dst.end());
      return;
    }
  }
  // elms is not dense or element type is bool, in either case
  // read the elements from getValues<X> the hard way:
  auto range = elms.getValues<X>();
  auto end = std::copy(range.begin(), range.end(), dst.begin());
  assert(end == dst.end());
}
