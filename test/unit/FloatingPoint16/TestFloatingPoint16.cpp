/*
 * SPDX-License-Identifier: Apache-2.0
 */

//=======================-- TestFloatingPoint16.cpp ---=======================//
//
// Tests FloatingPoint16.
//
//===----------------------------------------------------------------------===//

#include "src/Support/FloatingPoint16.hpp"

#include <benchmark/benchmark.h>

#include <cmath>
#include <iostream>
#include <limits>

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
    std::cout << "test_fp16_cast " << fp16_name << ":" << std::endl;

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

  template <typename FP16>
  int test_fp16_equals(const char *fp16_name) {
    std::cout << "test_fp16_equals " << fp16_name << ":" << std::endl;

    // 0 equals minus 0:
    auto zero = FP16::fromFloat(0.0);
    auto minusZero = FP16::fromFloat(-0.0);
    assert(zero.bitcastToU16() != minusZero.bitcastToU16());
    assert(zero == minusZero);

    for (uint32_t u = 0; u <= std::numeric_limits<uint16_t>::max(); ++u) {
      auto uf = FP16::bitcastFromU16(u);

      // NaN is not equal to itself:
      assert((uf != uf) == std::isnan(uf.toFloat()));

      // bitcast is 1-1 for non-zero numbers:
      // sample some non-zero numbers and check that they are equal to uf
      // iff they bitcast to u
      for (float f32 = 32768.0; f32 >= 5.96e-8; f32 /= 2) {
        FP16 f16 = FP16::fromFloat(f32);
        assert(u == f16.bitcastToU16() || uf != f16);
        FP16 f16neg = FP16::fromFloat(-f32);
        assert(u == f16neg.bitcastToU16() || uf != f16neg);
      }
    }

    return 0;
  }
};

template <typename FP16>
void BM_F32_TO_FP16(benchmark::State &state) {
  constexpr uint32_t u16max = std::numeric_limits<uint16_t>::max();
  float f32s[u16max + 1];
  for (uint32_t u = 0; u <= u16max; ++u) {
    f32s[u] = FP16::bitcastFromU16(u).toFloat();
  }
  for (auto _ : state) {
    // This code gets timed
    uint16_t u16 = 0;
    for (uint32_t u = 0; u <= u16max; ++u) {
      benchmark::DoNotOptimize(u16 += FP16::fromFloat(f32s[u]).bitcastToU16());
    }
  }
}
BENCHMARK(BM_F32_TO_FP16<float_16>);
BENCHMARK(BM_F32_TO_FP16<bfloat_16>);

template <typename FP16>
void BM_FP16_TO_F32(benchmark::State &state) {
  constexpr uint32_t u16max = std::numeric_limits<uint16_t>::max();
  FP16 fp16s[u16max + 1];
  for (uint32_t u = 0; u <= u16max; ++u) {
    fp16s[u] = FP16::bitcastFromU16(u);
  }
  for (auto _ : state) {
    // This code gets timed
    float f32 = 0.0;
    for (uint32_t u = 0; u <= u16max; ++u) {
      benchmark::DoNotOptimize(f32 += fp16s[u].toFloat());
    }
  }
}
BENCHMARK(BM_FP16_TO_F32<float_16>);
BENCHMARK(BM_FP16_TO_F32<bfloat_16>);

} // namespace

int main(int argc, char *argv[]) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  Test test;
  int failures = 0;
  failures += test.test_two_values();
  failures += test.test_fp16_cast<float_16>("float_16");
  failures += test.test_fp16_cast<bfloat_16>("bfloat_16");
  failures += test.test_fp16_equals<float_16>("float_16");
  failures += test.test_fp16_equals<bfloat_16>("bfloat_16");
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
