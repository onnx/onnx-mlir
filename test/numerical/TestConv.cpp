/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidcheck.h>

#include "llvm/Support/FileSystem.h"

#include "test/modellib/ModelLib.hpp"

#define DEBUG 0

using namespace mlir;

static const llvm::StringRef SHARED_LIB_BASE("./TestConv_main_graph");

// Made global so that we can repeat the test with different strides and
// dilations. Had to make them global to conform with the signatures of lambda
// requested by RapidTest.
int stride, dilation, isDynamic;
namespace onnx_mlir {
namespace test {

// Returns whether onnx-mlir compiled convolution is producing the same results
// as a naive implementation of convolution for a specific set of convolution
// parameters/configuration. Stride and dilation are square (same along H and
// W).
bool isOMConvTheSameAsNaiveImplFor(const int N, const int C, const int H,
    const int W, const int kH, const int kW, int pHBegin, int pHEnd,
    int pWBegin, int pWEnd, const ConvAutoPad autoPad) {
  static int testNum = 0;
  if (DEBUG)
    printf(
        "attempt %d with N %d, C %d, H %d, W %d, kH %d, kW %d, pHBegin %d, "
        "pHEnd %d, pWBegin %d, pWEnd %d, autopad %s, isDynamic %d, stride %d, "
        "dilation %d\n",
        ++testNum, N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd,
        Conv2DLibBuilder::getAutoPadName(autoPad).c_str(), isDynamic, stride,
        dilation);

  Conv2DLibBuilder conv(SHARED_LIB_BASE.str(), N, C, H, W, kH, kW, autoPad,
      pHBegin, pHEnd, pWBegin, pWEnd, stride, dilation, isDynamic);
  return conv.build() && conv.compileAndLoad() && conv.prepareInputs() &&
         conv.run() && conv.verifyOutputs();
}

} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestConv\n", nullptr, "TEST_ARGS");

  // Had to explicitly iterate over dynamic as otherwise the random algorithm
  // never got to testing the dynamic cases.
  for (isDynamic = 0; isDynamic < 2; ++isDynamic) {

    // First test: check auto pads that set the pad values.
    printf("test case generation with auto pad = VALID or SAME and %s.\n",
        (isDynamic ? "dynamic" : "static"));
    bool success = rc::check("convolution implementation correctness", []() {
      const auto S = *rc::gen::inRange(1, 3);
      stride = S;
      const auto D = *rc::gen::inRange(1, 3);
      dilation = D;
      const auto autoPad = (ConvAutoPad)*rc::gen::inRange(
          (int)ConvAutoPad::VALID, (int)ConvAutoPad::UB);
      const auto N = *rc::gen::inRange(1, 5);
      const auto C = *rc::gen::inRange(1, 10);
      const auto H = *rc::gen::inRange(5, 32 * stride);
      const auto W = *rc::gen::inRange(5, 32 * stride);
      const auto kH = *rc::gen::inRange(1, 6);
      const auto kW = *rc::gen::inRange(1, 6);
      // Make sure we have at least 1 output per dimension.
      RC_PRE((H / stride >= kH * dilation) && (W / stride > kW * dilation));
      RC_ASSERT(isOMConvTheSameAsNaiveImplFor(
          N, C, H, W, kH, kW, 0, 0, 0, 0, autoPad));
    });
    if (!success)
      return 1;

    // Second test: test NOTSET over a wide range of image and kernel sizes. Had
    // to manually iterate over strides and dilation to ensure sufficient
    // coverage.
    for (stride = 1; stride < 3; ++stride) {
      for (dilation = 1; dilation < 3; ++dilation) {
        printf("\nRun with stride %d, dilation %d and %s.\n", stride, dilation,
            (isDynamic ? "dynamic" : "static"));
        // For debugging, if helpful.
        if (false && stride == 1 && dilation == 1) {
          printf("  Skip no stride and no dilations\n");
          continue;
        }
        if (false && (stride < 2 || dilation < 2)) {
          printf("  Skip no stride or no dilations\n");
          continue;
        }
        // RapidCheck test case generation for a given stride and dilation.
        bool success =
            rc::check("convolution implementation correctness", []() {
              const auto N = *rc::gen::inRange(1, 5);
              const auto C = *rc::gen::inRange(1, 10);
              const auto H = *rc::gen::inRange(5, 32 * stride);
              const auto W = *rc::gen::inRange(5, 32 * stride);
              const auto kH = *rc::gen::inRange(1, 6);
              const auto kW = *rc::gen::inRange(1, 6);
              // We don't want an entire window of padding.
              const auto pHBegin = *rc::gen::inRange(0, kH);
              const auto pHEnd = *rc::gen::inRange(0, kH);
              const auto pWBegin = *rc::gen::inRange(0, kW);
              const auto pWEnd = *rc::gen::inRange(0, kW);
              // Make sure we have at least 1 output per dimension.
              RC_PRE((H / stride >= kH * dilation) &&
                     (W / stride > kW * dilation));
              RC_ASSERT(isOMConvTheSameAsNaiveImplFor(N, C, H, W, kH, kW,
                  pHBegin, pHEnd, pWBegin, pWEnd, ConvAutoPad::NOTSET));
            });
        if (!success)
          return 1;
      }
    }

    // Third test, exhaustive test over a small range of values.
    printf("\nExhaustive test cases with unit stride and dilation, and %s.\n",
        (isDynamic ? "dynamic" : "static"));
    stride = dilation = 1;
    for (int pHBegin = 0; pHBegin < 3; pHBegin++)
      for (int pHEnd = 0; pHEnd < 3; pHEnd++)
        for (int pWBegin = 0; pWBegin < 3; pWBegin++)
          for (int pWEnd = 0; pWEnd < 3; pWEnd++)
            assert(isOMConvTheSameAsNaiveImplFor(2, 4, 5, 5, 3, 3, pHBegin,
                pHEnd, pWBegin, pWEnd, ConvAutoPad::NOTSET));

  } // End loop over static / dynamic
  return 0;
}
