/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------- Arrays.cpp -----------------------------===//
//
// Arrays helper functions and data structures.
//
//===----------------------------------------------------------------------===//

#include "src/Support/Arrays.hpp"

#include "src/Support/BType.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "src/Support/WideNum.hpp"

#include "mlir/IR/Types.h"

using namespace mlir;

namespace onnx_mlir {

void widenArray(
    Type elementType, ArrayRef<char> bytes, MutableArrayRef<WideNum> wideData) {
  dispatchByMlirType(elementType, [bytes, wideData](auto btype) {
    using W = WideBType<btype>;
    auto src = castArrayRef<typename W::narrowtype>(bytes);
    std::transform(src.begin(), src.end(), wideData.begin(), W::widen);
  });
}

void narrowArray(
    Type elementType, ArrayRef<WideNum> wideData, MutableArrayRef<char> bytes) {
  dispatchByMlirType(elementType, [wideData, bytes](auto btype) {
    using W = WideBType<btype>;
    auto dst = castMutableArrayRef<typename W::narrowtype>(bytes);
    std::transform(wideData.begin(), wideData.end(), dst.begin(), W::narrow);
  });
}

ArrayBuffer<WideNum> widenOrReturnArray(
    Type elementType, ArrayRef<char> bytes) {
  unsigned bytewidth = getIntOrFloatByteWidth(elementType);
  if (bytewidth == sizeof(WideNum)) {
    return castArrayRef<WideNum>(bytes);
  }
  ArrayBuffer<WideNum>::Vector vec;
  vec.resize_for_overwrite(bytes.size() / bytewidth);
  widenArray(elementType, bytes, vec);
  return std::move(vec);
}

ArrayBuffer<char> narrowOrReturnArray(
    Type elementType, ArrayRef<WideNum> wideData) {
  unsigned bytewidth = getIntOrFloatByteWidth(elementType);
  if (bytewidth == sizeof(WideNum)) {
    return castArrayRef<char>(wideData);
  }
  ArrayBuffer<char>::Vector vec;
  vec.resize_for_overwrite(wideData.size() * bytewidth);
  narrowArray(elementType, wideData, vec);
  return std::move(vec);
}

} // namespace onnx_mlir