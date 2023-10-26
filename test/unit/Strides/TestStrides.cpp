/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============================-- TestStrides.cpp ---========================//
//
// Tests Strides.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/IR/Builders.h"

#include <iostream>

using namespace mlir;
using namespace onnx_mlir;

namespace {

class Test {
public:
  int test_reshapeStrides_success() {
    std::cout << "test_reshapeStrides_success:" << std::endl;

    SmallVector<int64_t, 4> shape{2, 1, 5};
    auto strides = getDefaultStrides(shape);

    SmallVector<int64_t, 4> reshapedShape{5, 2};
    auto reshapedStrides = reshapeStrides(shape, strides, reshapedShape);
    assert(reshapedStrides == getDefaultStrides(reshapedShape));

    return 0;
  }

  // This example triggered a bug.
  int test_reshapeStrides_unsqueeze_last() {
    std::cout << "test_reshapeStrides_unsqueeze_last:" << std::endl;

    SmallVector<int64_t, 4> shape{1, 2, 124, 1};
    SmallVector<int64_t, 4> strides{0, 0, 1, 0};

    SmallVector<int64_t, 4> reshapedShape{1, 2, 124, 1, 1};
    SmallVector<int64_t, 4> expectedReshapedStrides{0, 0, 1, 0, 0};
    auto reshapedStrides = reshapeStrides(shape, strides, reshapedShape);
    assert(reshapedStrides == expectedReshapedStrides);

    return 0;
  }

  int test_reshapeStrides_failure() {
    std::cout << "test_reshapeStrides_failure:" << std::endl;

    SmallVector<int64_t, 4> shape{2, 1, 5};
    auto strides = getDefaultStrides(shape);

    SmallVector<int64_t, 4> expandedShape{2, 3, 5};
    auto expandedStrides = expandStrides(strides, expandedShape);
    assert(strides == expandedStrides);

    SmallVector<int64_t, 4> reshapedShape{3, 2, 5};
    auto reshapedExpandedStrides =
        reshapeStrides(expandedShape, expandedStrides, reshapedShape);
    assert(std::nullopt == reshapedExpandedStrides);

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_reshapeStrides_success();
  failures += test.test_reshapeStrides_unsqueeze_last();
  failures += test.test_reshapeStrides_failure();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
