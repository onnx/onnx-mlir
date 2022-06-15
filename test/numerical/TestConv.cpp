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
  return conv.build() && conv.compileAndLoad() &&
         conv.checkInstructionFromEnv("TestConvNNPA_INSTRUCTION") &&
         conv.prepareInputsFromEnv("TestConvNNPA_DATARANGE") && conv.run() &&
         conv.verifyOutputs();
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
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  // Had to explicitly iterate over dynamic as otherwise the random algorithm
  // never got to testing the dynamic cases.
#ifdef TEST_CONV_STATIC
  int dynamicCase = 1;
#else
  int dynamicCase = 2;
#endif
  for (isDynamic = 0; isDynamic < dynamicCase; ++isDynamic) {
    // First test: check auto pads that set the pad values.
    printf("test case generation with auto pad = VALID or SAME and %s.\n",
        (isDynamic ? "dynamic" : "static"));
    bool success = rc::check("convolution implementation correctness", []() {
      const auto S = *rc::gen::inRange(1, 3);
      stride = S;
#ifdef TEST_CONV_D1
      const auto D = 1;
#else
      const auto D = *rc::gen::inRange(1, 3);
#endif
      dilation = D;
#ifdef TEST_CONV_VALID_UPPER
      const auto autoPad = (ConvAutoPad)*rc::gen::element(
          (int)ConvAutoPad::VALID, (int)ConvAutoPad::UPPER);
#else
      const auto autoPad = (ConvAutoPad)*rc::gen::inRange(
          (int)ConvAutoPad::VALID, (int)ConvAutoPad::UB);
#endif
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

    // Second test: test NOTSET over a wide range of image and kernel sizes.
    // Had to manually iterate over strides and dilation to ensure sufficient
    // coverage.
#ifdef TEST_CONV_D1
    int maxDilation = 2;
#else
    int maxDilation = 3;
#endif
    for (stride = 1; stride < 3; ++stride) {
      for (dilation = 1; dilation < maxDilation; ++dilation) {
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
              auto pHBegin = *rc::gen::inRange(0, kH);
              auto pWBegin = *rc::gen::inRange(0, kW);
              auto pHEnd = *rc::gen::inRange(0, kH);
              auto pWEnd = *rc::gen::inRange(0, kW);
#ifdef TEST_CONV_VALID_UPPER
              if (pHBegin != 0 || pWBegin != 0 || pHEnd != 0 || pWEnd != 0) {
                // Update pads for SAME_UPPER
                const auto Hout = std::ceil(float(H) / float(stride));
                const auto Wout = std::ceil(float(W) / float(stride));
                const auto Hpad =
                    std::max(int((Hout - 1) * stride + kH - H), 0);
                const auto Wpad =
                    std::max(int((Wout - 1) * stride + kW - W), 0);
                pHBegin = Hpad / 2;
                pWBegin = Wpad / 2;
                pHEnd = Hpad - pHBegin;
                pWEnd = Wpad - pWBegin;
              }
#endif
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

#ifndef TEST_CONV_VALID_UPPER
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
#endif

  } // End loop over static / dynamic
  return 0;
}
