#include <assert.h>
#include <string>

#include <benchmark/benchmark.h>

#include "include/OnnxMlirCompiler.h"
#include "test/modellib/ModelLib.hpp"

const std::string modelName("./perfbench");

static void SetCompilerOpt(const benchmark::State &state) {
  std::string O3("3");
  int rc = onnx_mlir::omSetCompilerOption(
      onnx_mlir::OptionKind::CompilerOptLevel, O3.c_str());
  rc += onnx_mlir::omSetCompilerOptionsFromEnv("TEST_PERF");
  assert(rc == 0 && "Failed to setup opt level");
}

static void BM_MatmulSquare(benchmark::State &state) {
  int IJK = state.range(0);
  MatMul2DLibBuilder model(modelName, IJK, IJK, IJK);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed matmul");

  for (auto _ : state)
    model.run();
}
BENCHMARK(BM_MatmulSquare)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Setup(SetCompilerOpt);

static void BM_GemmSquare(benchmark::State &state) {
  int IJK = state.range(0);
  GemmLibBuilder model(modelName, IJK, IJK, IJK, false, false, 1, 1.0, 1.0);
  assert(model.build() && model.compileAndLoad() && model.prepareInputs() &&
         "failed gemm");

  for (auto _ : state)
    model.run();
}
// BENCHMARK(BM_GemmSquare)->RangeMultiplier(2)->Range(16, 32);
