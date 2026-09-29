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

#include <algorithm>
#include <iostream>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

class Test {
public:
  template <typename Element>
  int test_restrideArray(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
      uint64_t valueMask = 255) {
    const int64_t count = ShapedType::getNumElements(shape);
    auto expandedStrides = expandStrides(strides, shape);
    size_t sourceSize = count == 0 ? 0 : 1;
    if (count != 0)
      for (size_t axis = 0; axis < shape.size(); ++axis)
        sourceSize += (shape[axis] - 1) * expandedStrides[axis];
    std::vector<Element> source(sourceSize);
    for (size_t index = 0; index < sourceSize; ++index)
      source[index] =
          static_cast<Element>((index * 31 + index / 251) & valueMask);
    std::vector<Element> expected(count);
    for (int64_t index = 0; index < count; ++index) {
      auto coordinates = unflattenIndex(shape, index);
      expected[index] =
          source[getStridesPosition(coordinates, expandedStrides)];
    }
    std::vector<Element> actual(count + 2, static_cast<Element>(93));
    restrideArray<Element>(shape, strides, source,
        MutableArrayRef<Element>(actual).slice(1, count));
    if (actual.front() != static_cast<Element>(93) ||
        actual.back() != static_cast<Element>(93) ||
        !std::equal(expected.begin(), expected.end(), actual.begin() + 1)) {
      std::cerr << "restrideArray mismatch for shape";
      for (int64_t dimension : shape)
        std::cerr << " " << dimension;
      std::cerr << "\n";
      return 1;
    }
    return 0;
  }

  int test_restrideArray_transpose() {
    std::cout << "test_restrideArray_transpose:" << std::endl;
    int failures = 0;
    for (int64_t rows : {2, 7, 15, 16, 17, 31, 32, 33, 65}) {
      for (int64_t columns : {2, 13, 15, 16, 17, 31, 32, 33, 129}) {
        failures += test_restrideArray<uint8_t>({rows, columns}, {1, rows});
        failures += test_restrideArray<int8_t>({rows, columns}, {1, rows});
        failures += test_restrideArray<uint8_t>(
            {1, 1, rows, columns}, {0, 0, 1, rows}, 15);
        failures += test_restrideArray<uint8_t>(
            {1, rows, 1, columns, 1}, {17, 1, 9, rows, 3});
        failures +=
            test_restrideArray<uint8_t>({1, 1, rows, columns}, {1, rows});
      }
    }
    return failures;
  }

  int test_restrideArray_fallback() {
    std::cout << "test_restrideArray_fallback:" << std::endl;
    int failures = 0;
    failures += test_restrideArray<uint8_t>({}, {});
    failures += test_restrideArray<uint8_t>({0, 13}, {1, 0});
    failures += test_restrideArray<uint8_t>({7, 0, 13}, {1, 0, 7});
    failures += test_restrideArray<uint8_t>({1, 13}, {0, 1});
    failures += test_restrideArray<uint8_t>({7, 1}, {1, 0});
    failures += test_restrideArray<uint8_t>({7, 13}, {13, 1});
    failures += test_restrideArray<uint8_t>({7, 13}, {0, 0});
    failures += test_restrideArray<uint8_t>({7, 13}, {1, 0});
    failures += test_restrideArray<uint8_t>({7, 13}, {0, 1});
    failures += test_restrideArray<uint8_t>({7, 13}, {1, 8});
    failures += test_restrideArray<uint8_t>({7, 13}, {2, 14});
    failures += test_restrideArray<uint8_t>({2, 7, 13}, {91, 1, 7});
    failures += test_restrideArray<uint8_t>({2, 7, 13}, {0, 1, 7});
    failures += test_restrideArray<uint8_t>({2, 7, 13}, {1, 7});
    failures += test_restrideArray<uint16_t>({33, 65}, {1, 33}, 65535);
    failures += test_restrideArray<uint32_t>({33, 65}, {1, 33}, 0xffffffff);
    failures += test_restrideArray<uint64_t>({33, 65}, {1, 33}, ~uint64_t(0));
    return failures;
  }

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
  failures += test.test_restrideArray_transpose();
  failures += test.test_restrideArray_fallback();
  failures += test.test_reshapeStrides_success();
  failures += test.test_reshapeStrides_unsqueeze_last();
  failures += test.test_reshapeStrides_failure();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
