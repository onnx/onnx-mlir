/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "llvm/Support/FileSystem.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

#define DEBUG 0

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestConv_main_graph");

// Made global so that we can repeat the test with different strides and
// dilations. Had to make them global to conform with the signatures of lambda
// requested by RapidTest.
int stride, dilation, isDynamic;

// Support.
static int myCeil(int a, int b) { return ceil((1.0 * a) / (1.0 * b)); }
static int myFloor(int a, int b) { return floor((1.0 * a) / (1.0 * b)); }

//===----------------------------------------------------------------------===//
// Compute shape for various auto pad value and check results.
//===----------------------------------------------------------------------===//

// TODO: Ideally these values would be corroborated with Pytorch/TF. However
// Pytorch only supports same/valid with unit strides. Maybe check with TF?

static LogicalResult checkShapes(const int NIn, const int CIn, const int HIn,
    const int WIn, const int kH, const int kW, const int autoPad,
    const int NOut, const int COut, const int HOut, const int WOut,
    int &pHBegin, int &pHEnd, int &pWBegin, int &pWEnd) {

  // Check first params.
  if (NIn != NOut) {
    cerr << "N mismatch: in " << NIn << ", out " << NOut << endl;
    return failure();
  }
  if (CIn != COut) {
    cerr << "C mismatch: in " << CIn << ", out " << COut << endl;
    return failure();
  }

  // Gather variables in arrays to match ONNX descriptions.
  int I[] = {HIn, WIn};
  int K[] = {kH, kW};
  int pBegin[] = {pHBegin, pWBegin};
  int pEnd[] = {pHEnd, pWEnd};
  int p[] = {pHBegin + pHEnd, pWBegin + pWEnd};
  int s[] = {stride, stride};
  int d[] = {dilation, dilation};
  int O[] = {HOut, WOut};

  // Check dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
  int myO[2], myPBegin[2], myPEnd[2];
  for (int i = 0; i < 2; ++i) {
    if (autoPad == AUTO_PAD_NOTSET) {
      // NOSET:
      //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
      myO[i] = myFloor((I[i] + p[i] - ((K[i] - 1) * d[i] + 1)), s[i]) + 1;
      myPBegin[i] = pBegin[i];
      myPEnd[i] = pEnd[i];
    } else if (autoPad == AUTO_PAD_VALID) {
      // VALID:
      // * O[i] = ceil((I[i] - ((K[i] - 1) * d[i] + 1) + 1) / s[i])
      // * P = 0
      myO[i] = myCeil((I[i] - ((K[i] - 1) * d[i] + 1) + 1), s[i]);
      myPBegin[i] = myPEnd[i] = 0;
    } else {
      // SAME_LOWER or SAME_UPPER:
      // * O[i] = ceil(I[i] / s[i])
      // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
      // * P[i] = p' / 2, if odd, first or second are increased by one.
      myO[i] = myCeil(I[i], s[i]);
      int pSum = (myO[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i];
      pSum = pSum >= 0 ? pSum : 0;
      myPBegin[i] = myPEnd[i] = pSum / 2;
      if (pSum % 2 != 0) {
        if (autoPad == AUTO_PAD_UPPER)
          myPEnd[i] += 1;
        else
          myPBegin[i] += 1;
      }
    }
    if (myO[i] != O[i]) {
      cerr << "output sizes mismatch: computed " << myO[i] << ", got " << O[i]
           << endl;
      return failure();
    }
  }
  // Test all good, set padding values for computed ones.
  pHBegin = myPBegin[0];
  pWBegin = myPBegin[1];
  pHEnd = myPEnd[0];
  pWEnd = myPEnd[1];

  return success();
}

// Returns whether onnx-mlir compiled convolution is producing the same results
// as a naive implementation of convolution for a specific set of convolution
// parameters/configuration. Stride and dilation are square (same along H and
// W).
bool isOMConvTheSameAsNaiveImplFor(const int N, const int C, const int H,
    const int W, const int kH, const int kW, int pHBegin, int pHEnd,
    int pWBegin, int pWEnd, const int autoPad) {
  static int testNum = 0;
  if (DEBUG)
    printf(
        "attempt %d with N %d, C %d, H %d, W %d, kH %d, kW %d, pHBegin %d, "
        "pHEnd %d, pWBegin %d, pWEnd %d, autopad %s, isDynamic %d, stride %d, "
        "dilation %d\n",
        ++testNum, N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd,
        getAutoPadName(autoPad).c_str(), isDynamic, stride, dilation);

  int NOut, COut, HOut, WOut;
  if (!genConv2DModelAndCompile(
          /*compiler options */
          SHARED_LIB_BASE.str(),
          /*input conv param */ N, C, H, W, kH, kW, autoPad, pHBegin, pHEnd,
          pWBegin, pWEnd, stride, dilation, isDynamic,
          /*output conv param*/ NOut, COut, HOut, WOut))
    return false;

  onnx_mlir::ExecutionSession sess(getSharedLibName(SHARED_LIB_BASE.str()));

  std::vector<OMTensorUniquePtr> inputs;
  auto xOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>({N, C, H, W}), omTensorDestroy);
  inputs.emplace_back(move(xOmt));
  auto wOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>({C, C, kH, kW}), omTensorDestroy);
  inputs.emplace_back(move(wOmt));

  // Check shape and also compute pad H/W Begin/End params.
  LogicalResult res = checkShapes(/*in*/ N, C, H, W, kH, kW, autoPad, NOut,
      COut, HOut, WOut, /*in/out*/ pHBegin, pHEnd, pWBegin, pWEnd);
  if (failed(res)) {
    if (DEBUG) {
      cerr << "Conv after check shape, N out " << NOut << ", C out " << COut
           << ", H out " << HOut << ", W out " << WOut << ", ph begin "
           << pHBegin << ", ph end " << pHEnd << ", pw begin " << pWBegin
           << ", pw end " << pWEnd << endl;
    }
    return false;
  }

  auto ref = omTensorCreateWithShape<float>({NOut, COut, HOut, WOut});
  auto &img = inputs.at(0);
  auto &filter = inputs.at(1);
  for (int64_t n = 0; n < NOut; n++)
    for (int64_t c = 0; c < COut; c++)
      for (int64_t h = 0; h < HOut; h++)
        for (int64_t w = 0; w < WOut; w++) {
          omTensorGetElem<float>(ref, {n, c, h, w}) = 0;
          for (int64_t ci = 0; ci < C; ci++)
            for (int64_t kh = 0; kh < kH; kh++)
              for (int64_t kw = 0; kw < kW; kw++)
                if ((h * stride + kh * dilation - pHBegin >= 0 &&
                        h * stride + kh * dilation - pHBegin < H) &&
                    (w * stride + kw * dilation - pWBegin >= 0 &&
                        w * stride + kw * dilation - pWBegin < W))
                  omTensorGetElem<float>(ref, {n, c, h, w}) +=
                      omTensorGetElem<float>(img.get(),
                          {n, ci, h * stride + kh * dilation - pHBegin,
                              w * stride + kw * dilation - pWBegin}) *
                      omTensorGetElem<float>(filter.get(), {c, ci, kh, kw});
        }

  auto outputs = sess.run(move(inputs));
  auto &conv = outputs.at(0);

  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  return omTensorAreTwoOmtsClose<float>(conv.get(), ref, rtol, atol);
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE.str()));

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
      const auto autoPad = *rc::gen::inRange(AUTO_PAD_VALID, AUTO_PAD_UB);
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
                  pHBegin, pHEnd, pWBegin, pWEnd, AUTO_PAD_NOTSET));
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
                pHEnd, pWBegin, pWEnd, AUTO_PAD_NOTSET));

  } // End loop over static / dynamic
  return 0;
}
