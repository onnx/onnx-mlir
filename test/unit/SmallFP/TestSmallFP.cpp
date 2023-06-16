/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========================-- TestSmallFP.cpp ---===========================//
//
// Tests SmallFP.
//
//===----------------------------------------------------------------------===//

#include "src/Support/SmallFP.hpp"

#include <benchmark/benchmark.h>

#include <cmath>
#include <iostream>
#include <limits>

using namespace onnx_mlir;

namespace {

class Test {

public:
  template <typename FP>
  int test_fp_cast(const char *fp_name, float fpmin, float fpmax) {
    std::cout << "test_fp_cast " << fp_name << ":" << std::endl;

    for (float f = fpmax; f >= fpmin; f /= 2) {
      float g = static_cast<float>(FP(f));
      if (f != g)
        std::cout << f << " != " << g << "\n";
      assert(f == static_cast<float>(FP(f)));
      assert(f == FP(f).toFloat());
      assert(-f == static_cast<float>(FP(-f)));
      assert(static_cast<int64_t>(f) == static_cast<int64_t>(FP(f)));
      assert(static_cast<int64_t>(-f) == static_cast<int64_t>(FP(-f)));

      FP fromFloat = FP::fromFloat(f);
      FP staticCast = static_cast<FP>(f);
      // FP has no operator== so compare bitcasts instead;
      assert(fromFloat.bitcastToUInt() == staticCast.bitcastToUInt());
    }

    return 0;
  }

  template <typename FP>
  int test_fp_equals(const char *fp_name, float fpmin, float fpmax,
      bool hasNegativeZero = true) {
    std::cout << "test_fp_equals " << fp_name << ":" << std::endl;

    // 0 equals minus 0:
    auto zero = FP::fromFloat(0.0f);
    auto minusZero = FP::fromFloat(-0.0f);
    assert(zero == minusZero);
    if (hasNegativeZero)
      assert(zero.bitcastToUInt() != minusZero.bitcastToUInt());

    uint32_t umax = std::numeric_limits<typename FP::bitcasttype>::max();
    for (uint32_t u = 0; u <= umax; ++u) {
      auto uf = FP::bitcastFromUInt(u);

      // NaN is not equal to itself:
      assert((uf != uf) == std::isnan(uf.toFloat()));

      // bitcast is 1-1 for non-zero numbers:
      // sample some non-zero numbers and check that they are equal to uf
      // iff they bitcast to u
      for (float f32 = fpmax; f32 >= fpmin; f32 /= 2) {
        FP fp = FP::fromFloat(f32);
        unsigned v = fp.bitcastToUInt();
        if (!(u == v || uf != fp))
          std::cout << u << " != " << v << "\n";
        assert(u == fp.bitcastToUInt() || uf != fp);
        FP fpneg = FP::fromFloat(-f32);
        assert(u == fpneg.bitcastToUInt() || uf != fpneg);
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
    f32s[u] = FP16::bitcastFromUInt(u).toFloat();
  }
  for (auto _ : state) {
    // This code gets timed
    uint16_t u16 = 0;
    for (uint32_t u = 0; u <= u16max; ++u) {
      benchmark::DoNotOptimize(u16 += FP16::fromFloat(f32s[u]).bitcastToUInt());
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
    fp16s[u] = FP16::bitcastFromUInt(u);
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

  const bool noNegZero = false;
  const float fp8min = 0.005f;
  const float fp8max = 192.0f;
  failures += test.test_fp_cast<float_8e4m3fn>("float_8e4m3fn", fp8min, fp8max);
  failures +=
      test.test_fp_equals<float_8e4m3fn>("float_8e4m3fn", fp8min, fp8max);
  failures +=
      test.test_fp_cast<float_8e4m3fnuz>("float_8e4m3fnuz", fp8min, fp8max);
  failures += test.test_fp_equals<float_8e4m3fnuz>(
      "float_8e4m3fnuz", fp8min, fp8max, noNegZero);
  failures += test.test_fp_cast<float_8e5m2>("float_8e5m2", fp8min, fp8max);
  failures += test.test_fp_equals<float_8e5m2>("float_8e5m2", fp8min, fp8max);
  failures +=
      test.test_fp_cast<float_8e5m2fnuz>("float_8e5m2fnuz", fp8min, fp8max);
  failures += test.test_fp_equals<float_8e5m2fnuz>(
      "float_8e5m2fnuz", fp8min, fp8max, noNegZero);
  const float fp16min = 5.96e-8f;
  const float fp16max = 32768.0f;
  failures += test.test_fp_cast<float_16>("float_16", fp16min, fp16max);
  failures += test.test_fp_cast<bfloat_16>("bfloat_16", fp16min, fp16max);
  failures += test.test_fp_equals<float_16>("float_16", fp16min, fp16max);
  failures += test.test_fp_equals<bfloat_16>("bfloat_16", fp16min, fp16max);

  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
