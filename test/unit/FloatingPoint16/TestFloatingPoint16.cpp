/*
 * SPDX-License-Identifier: Apache-2.0
 */

//============================-- FloatingPoint16.cpp ---======================//
//
// Tests FloatingPoint16.
//
//===----------------------------------------------------------------------===//

#include "src/Support/FloatingPoint16.hpp"

#include <iostream>

using namespace onnx_mlir;

namespace {

class Test {

public:
  int test_two_values() {
    std::cout << "test_two_values:" << std::endl;

    float_16 f9984(9984);
    bfloat_16 fminus1(-1);
    float_16 bfminus1(fminus1);
    bfloat_16 bf9984(f9984);
    assert(bfminus1.toFloat() == fminus1.toFloat());
    assert(static_cast<float_16>(bf9984).toFloat() ==
           static_cast<bfloat_16>(f9984).toFloat());

    return 0;
  }

  template <typename FP16>
  int test_fp16_cast(const char *fp16_name) {
    std::cout << "test_fp_cast " << fp16_name << ":" << std::endl;

    for (float f = 32768.0; f >= 5.96e-8; f /= 2) {
      assert(f == static_cast<float>(FP16(f)));
      assert(f == FP16(f).toFloat());
      assert(-f == static_cast<float>(FP16(-f)));
      assert(static_cast<int64_t>(f) == static_cast<int64_t>(FP16(f)));
      assert(static_cast<int64_t>(-f) == static_cast<int64_t>(FP16(-f)));

      FP16 fromFloat = FP16::fromFloat(f);
      FP16 staticCast = static_cast<FP16>(f);
      // FP16 has no operator== so compare bitcasts instead;
      assert(fromFloat.bitcastToU16() == staticCast.bitcastToU16());
    }

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_two_values();
  failures += test.test_fp16_cast<float_16>("float_16");
  failures += test.test_fp16_cast<bfloat_16>("bfloat_16");
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
