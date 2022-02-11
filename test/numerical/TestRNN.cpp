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

static const llvm::StringRef SHARED_LIB_BASE("./TestRNN_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

// Returns whether onnx-mlir compiled RNN is producing the same results as a
// naive implementation of RNN for a specific set of RNN
// parameters/configuration.
bool isOMRNNTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, bool isDynamicS = false,
    bool isDynamicB = false) {

  int D;
  SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt = nullptr;
  OMTensor *rOmt = nullptr;
  OMTensor *bOmt = nullptr;
  if (!genRNNModelAndCompile(
          /* compile option */
          SHARED_LIB_BASE.str(),
          /* RNN param in*/
          direction, S, B, I, H, isDynamicS, isDynamicB,
          /* RNN param out*/
          D, xShape, hShape, wOmt, rOmt, bOmt))
    return false;

  onnx_mlir::ExecutionSession sess(getSharedLibName(SHARED_LIB_BASE.str()));

  std::vector<OMTensorUniquePtr> inputs;
  auto xOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(xShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(xOmt));
  auto hOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(hShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(hOmt));

  // Naive RNN implementation.
  // Equations for RNN.
  // - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  auto refY = omTensorCreateWithShape<float>({S, D, B, H});
  auto refYh = omTensorCreateWithShape<float>({D, B, H});
  auto &input = inputs.at(0);
  auto &initialH = inputs.at(1);

  auto weight = OMTensorUniquePtr(wOmt, omTensorDestroy);
  auto recurr = OMTensorUniquePtr(rOmt, omTensorDestroy);
  auto bias = OMTensorUniquePtr(bOmt, omTensorDestroy);

  // Initialize refYh.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++)
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH.get(), {d, b, h});

  // Main computation.
  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      auto XtWi = omTensorCreateWithShape<float>({B, H});
      auto HtRi = omTensorCreateWithShape<float>({B, H});
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input.get(), {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr.get(), {d, h, k});
          }
        }
      }
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          // - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
          float Ht = tanh(omTensorGetElem<float>(XtWi, {b, h}) +
                          omTensorGetElem<float>(HtRi, {b, h}) +
                          omTensorGetElem<float>(bias.get(), {d, h}) +
                          omTensorGetElem<float>(bias.get(), {d, h + H}));
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          omTensorGetElem<float>(refY, {seq, d, b, h}) = Ht;
        }
      }
    }
  }

  // onnx-mlir implementation.
  auto outputs = sess.run(move(inputs));
  auto &rnnY = outputs.at(0);
  auto &rnnYh = outputs.at(1);

  return (omTensorAreTwoOmtsClose<float>(rnnY.get(), refY) &&
          omTensorAreTwoOmtsClose<float>(rnnYh.get(), refYh));
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE.str()));

  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestRNN\n", nullptr, "TEST_ARGS");

  // RapidCheck test case generation.
  bool success = rc::check("RNN implementation correctness", []() {
    // The number of directions.
    // 1: forward, -1: reverse, 2: bidirectional
    const auto D = *rc::gen::element(1, -1, 2);
    // Sequence length.
    const auto S = *rc::gen::inRange(1, 5);
    // Batch size.
    const auto B = *rc::gen::inRange(5, 10);
    // Input size.
    const auto I = *rc::gen::inRange(5, 10);
    // Hidden size.
    const auto H = *rc::gen::inRange(5, 10);
    // Whether test dynamic dimension for sequence.
    const auto isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const auto isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(
        isOMRNNTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0, isDynB == 0));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++) {
          // Static dimensions.
          // forward
          assert(isOMRNNTheSameAsNaiveImplFor(1, s, b, i, h));
          // reverse
          assert(isOMRNNTheSameAsNaiveImplFor(-1, s, b, i, h));
          // bidirectional
          assert(isOMRNNTheSameAsNaiveImplFor(2, s, b, i, h));

          // Dynamic dimensions for sequence, batch size.
          // forward
          assert(isOMRNNTheSameAsNaiveImplFor(1, s, b, i, h, true, true));
          // reverse
          assert(isOMRNNTheSameAsNaiveImplFor(-1, s, b, i, h, true, true));
          // bidirectional
          assert(isOMRNNTheSameAsNaiveImplFor(2, s, b, i, h, true, true));
        }
  return 0;
}
