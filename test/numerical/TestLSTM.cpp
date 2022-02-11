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

static const llvm::StringRef SHARED_LIB_BASE("./TestLSTM_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

// Returns whether onnx-mlir compiled LSTM is producing the same results as a
// naive implementation of LSTM for a specific set of LSTM
// parameters/configuration.
bool isOMLSTMTheSameAsNaiveImplFor(const int direction, const int S,
    const int B, const int I, const int H, bool isDynamicS = false,
    bool isDynamicB = false) {

  int D;
  SmallVector<int64_t, 3> xShape, hShape, cShape;
  OMTensor *wOmt = nullptr;
  OMTensor *rOmt = nullptr;
  OMTensor *bOmt = nullptr;
  OMTensor *pOmt = nullptr;
  if (!genLSTMModelAndCompile(
          /* compile option */
          SHARED_LIB_BASE.str(),
          /* LSTM param in*/
          direction, S, B, I, H, isDynamicS, isDynamicB,
          /* LSTM param out*/
          D, xShape, hShape, cShape, wOmt, rOmt, bOmt, pOmt))
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
  auto cOmt = OMTensorUniquePtr(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(cShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(cOmt));

  auto refY = omTensorCreateWithShape<float>({S, D, B, H});
  auto refYh = omTensorCreateWithShape<float>({D, B, H});
  auto refYc = omTensorCreateWithShape<float>({D, B, H});
  // Naive LSTM implementation.
  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)

  auto weight = OMTensorUniquePtr(wOmt, omTensorDestroy);
  auto recurr = OMTensorUniquePtr(rOmt, omTensorDestroy);
  auto bias = OMTensorUniquePtr(bOmt, omTensorDestroy);
  auto peepholes = OMTensorUniquePtr(pOmt, omTensorDestroy);

  auto &input = inputs.at(0);
  auto &initialH = inputs.at(1);
  auto &initialC = inputs.at(2);

  // Initialize refYh and refYc.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++) {
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH.get(), {d, b, h});
        omTensorGetElem<float>(refYc, {d, b, h}) =
            omTensorGetElem<float>(initialC.get(), {d, b, h});
      }

  // Main computation.
  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      auto XtWi = omTensorCreateWithShape<float>({B, H});
      auto XtWo = omTensorCreateWithShape<float>({B, H});
      auto XtWf = omTensorCreateWithShape<float>({B, H});
      auto XtWc = omTensorCreateWithShape<float>({B, H});
      auto HtRi = omTensorCreateWithShape<float>({B, H});
      auto HtRo = omTensorCreateWithShape<float>({B, H});
      auto HtRf = omTensorCreateWithShape<float>({B, H});
      auto HtRc = omTensorCreateWithShape<float>({B, H});
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          omTensorGetElem<float>(XtWo, {b, h}) = 0;
          omTensorGetElem<float>(XtWf, {b, h}) = 0;
          omTensorGetElem<float>(XtWc, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input.get(), {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h, k});
            omTensorGetElem<float>(XtWo, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h + 1 * H, k});
            omTensorGetElem<float>(XtWf, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h + 2 * H, k});
            omTensorGetElem<float>(XtWc, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h + 3 * H, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          omTensorGetElem<float>(HtRo, {b, h}) = 0;
          omTensorGetElem<float>(HtRf, {b, h}) = 0;
          omTensorGetElem<float>(HtRc, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr.get(), {d, h, k});
            omTensorGetElem<float>(HtRo, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurr.get(), {d, h + 1 * H, k});
            omTensorGetElem<float>(HtRf, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurr.get(), {d, h + 2 * H, k});
            omTensorGetElem<float>(HtRc, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurr.get(), {d, h + 3 * H, k});
          }
        }
      }
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          float previousCt = omTensorGetElem<float>(refYc, {d, b, h});
          // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
          float it = sigmoid(
              omTensorGetElem<float>(XtWi, {b, h}) +
              omTensorGetElem<float>(HtRi, {b, h}) +
              omTensorGetElem<float>(peepholes.get(), {d, h}) * previousCt +
              omTensorGetElem<float>(bias.get(), {d, h}) +
              omTensorGetElem<float>(bias.get(), {d, h + 4 * H}));
          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
          float ft =
              sigmoid(omTensorGetElem<float>(XtWf, {b, h}) +
                      omTensorGetElem<float>(HtRf, {b, h}) +
                      omTensorGetElem<float>(peepholes.get(), {d, h + 2 * H}) *
                          previousCt +
                      omTensorGetElem<float>(bias.get(), {d, h + 2 * H}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 6 * H}));
          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          float ct = tanh(omTensorGetElem<float>(XtWc, {b, h}) +
                          omTensorGetElem<float>(HtRc, {b, h}) +
                          omTensorGetElem<float>(bias.get(), {d, h + 3 * H}) +
                          omTensorGetElem<float>(bias.get(), {d, h + 7 * H}));
          // Ct = ft (.) Ct-1 + it (.) ct
          float Ct = ft * previousCt + it * ct;
          omTensorGetElem<float>(refYc, {d, b, h}) = Ct;
          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
          float ot = sigmoid(
              omTensorGetElem<float>(XtWo, {b, h}) +
              omTensorGetElem<float>(HtRo, {b, h}) +
              omTensorGetElem<float>(peepholes.get(), {d, h + 1 * H}) * Ct +
              omTensorGetElem<float>(bias.get(), {d, h + 1 * H}) +
              omTensorGetElem<float>(bias.get(), {d, h + 5 * H}));
          // Ht = ot (.) h(Ct)
          float Ht = ot * tanh(Ct);
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          omTensorGetElem<float>(refY, {seq, d, b, h}) = Ht;
        }
      }
    }
  }

  // onnx-mlir implementation.
  auto outputs = sess.run(move(inputs));
  auto &lstmY = outputs.at(0);
  auto &lstmYh = outputs.at(1);
  auto &lstmYc = outputs.at(2);

  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  return (omTensorAreTwoOmtsClose<float>(lstmY.get(), refY, rtol, atol) &&
          omTensorAreTwoOmtsClose<float>(lstmYh.get(), refYh, rtol, atol) &&
          omTensorAreTwoOmtsClose<float>(lstmYc.get(), refYc, rtol, atol));
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE.str()));

  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestLSTM\n", nullptr, "TEST_ARGS");

  // RapidCheck test case generation.
  bool success = rc::check("LSTM implementation correctness", []() {
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
        isOMLSTMTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0, isDynB == 0));
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
          assert(isOMLSTMTheSameAsNaiveImplFor(1, s, b, i, h));
          // reverse
          assert(isOMLSTMTheSameAsNaiveImplFor(-1, s, b, i, h));
          // bidirectional
          assert(isOMLSTMTheSameAsNaiveImplFor(2, s, b, i, h));

          // Dynamic dimensions for sequence, batch size.
          // forward
          assert(isOMLSTMTheSameAsNaiveImplFor(1, s, b, i, h, true, true));
          // reverse
          assert(isOMLSTMTheSameAsNaiveImplFor(-1, s, b, i, h, true, true));
          // bidirectional
          assert(isOMLSTMTheSameAsNaiveImplFor(2, s, b, i, h, true, true));
        }
  return 0;
}
