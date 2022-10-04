/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- Arrays.hpp -----------------------------===//
//
// Arrays helper functions and data structures.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Type;
}

namespace onnx_mlir {

union WideNum;

// Light-weight version of MemoryBuffer. Can either point to external memory or
// hold internal memory. An ArrayBuffer can only be moved, not copied.
template <typename T>
class ArrayBuffer {
public:
  using Vector = llvm::SmallVector<T, 8 / sizeof(T)>;

  ArrayBuffer() = default; // empty
  ArrayBuffer(Vector &&vec) : vec(std::move(vec)), ref(this->vec) {}
  ArrayBuffer(llvm::ArrayRef<T> ref) : vec(), ref(ref) {}
  ArrayBuffer(const ArrayBuffer &) = delete;
  ArrayBuffer(ArrayBuffer &&other)
      : vec(std::move(other.vec)),
        ref(vec.empty() ? other.ref : llvm::makeArrayRef(vec)) {}

  llvm::ArrayRef<T> get() const { return ref; }

  static ArrayBuffer make(size_t length,
      const std::function<void(llvm::MutableArrayRef<T>)> &filler) {
    Vector vec;
    vec.resize_for_overwrite(length);
    filler(llvm::makeMutableArrayRef(vec.begin(), length));
    return std::move(vec);
  }

private:
  const Vector vec;
  const llvm::ArrayRef<T> ref;
};

template <typename New, typename Old = char>
llvm::ArrayRef<New> castArrayRef(llvm::ArrayRef<Old> a) {
  return llvm::makeArrayRef(reinterpret_cast<const New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename New = char>
llvm::ArrayRef<New> asArrayRef(llvm::StringRef s) {
  return llvm::makeArrayRef(
      reinterpret_cast<const New *>(s.data()), s.size() / sizeof(New));
}

template <typename Old = char>
llvm::StringRef asStringRef(llvm::ArrayRef<Old> a) {
  llvm::ArrayRef<char> c = castArrayRef<char>(a);
  return llvm::StringRef(c.begin(), c.size());
}

template <typename New, typename Old = char>
llvm::MutableArrayRef<New> castMutableArrayRef(llvm::MutableArrayRef<Old> a) {
  return llvm::makeMutableArrayRef(reinterpret_cast<New *>(a.data()),
      (a.size() * sizeof(Old)) / sizeof(New));
}

template <typename Src, typename Dst, typename Fn>
void fillOrTransform(
    llvm::ArrayRef<Src> src, llvm::MutableArrayRef<Dst> dst, Fn fn) {
  if (src.size() == 1)
    std::fill(dst.begin(), dst.end(), fn(src.front()));
  else
    std::transform(src.begin(), src.end(), dst.begin(), fn);
}

void widenArray(mlir::Type elementType, llvm::ArrayRef<char> bytes,
    llvm::MutableArrayRef<WideNum> wideData);

void narrowArray(mlir::Type elementType, llvm::ArrayRef<WideNum> wideData,
    llvm::MutableArrayRef<char> bytes);

ArrayBuffer<WideNum> widenOrReturnArray(
    mlir::Type elementType, llvm::ArrayRef<char> bytes);

ArrayBuffer<char> narrowOrReturnArray(
    mlir::Type elementType, llvm::ArrayRef<WideNum> wideData);

} // namespace onnx_mlir