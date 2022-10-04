/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============================-- TestStrides.cpp ---========================//
//
// Tests Strides.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Support/Strides.hpp"

#include "mlir/IR/Builders.h"

#include <iostream>

using namespace mlir;
using namespace onnx_mlir;

namespace {

class Test {
public:
  int test_getStridesNumElements() {
    std::cout << "test_getStridesNumElements:" << std::endl;

    SmallVector<int64_t, 4> shape{2, 3, 5};
    auto strides = getDefaultStrides(shape);
    assert(ShapedType::getNumElements(shape) ==
           getStridesNumElements(shape, strides));

    return 0;
  }

  int test_reshapeStrides_success() {
    std::cout << "test_reshapeStrides_success:" << std::endl;

    SmallVector<int64_t, 4> shape{2, 1, 5};
    auto strides = getDefaultStrides(shape);

    SmallVector<int64_t, 4> reshapedShape{5, 2};
    auto reshapedStrides = reshapeStrides(shape, strides, reshapedShape);
    assert(getStridesNumElements(shape, strides) ==
           getStridesNumElements(reshapedShape, *reshapedStrides));

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
    assert(None == reshapedExpandedStrides);

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_getStridesNumElements();
  failures += test.test_reshapeStrides_success();
  failures += test.test_reshapeStrides_failure();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
