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

static const llvm::StringRef SHARED_LIB_BASE("./TestGRU_main_graph");

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

// Returns whether onnx-mlir compiled GRU is producing the same results as a
// naive implementation of GRU for a specific set of GRU
// parameters/configuration.
bool isOMGRUTheSameAsNaiveImplFor(const int direction, const int S, const int B,
    const int I, const int H, const int LinearBeforeReset,
    bool isDynamicS = false, bool isDynamicB = false) {

  int D;
  SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt = nullptr;
  OMTensor *rOmt = nullptr;
  OMTensor *bOmt = nullptr;
  if (!genGRUModelAndCompile(
          /* compile option */
          SHARED_LIB_BASE.str(),
          /* GRU param in*/
          direction, S, B, I, H, LinearBeforeReset, isDynamicS, isDynamicB,
          /* GRU param out*/
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

  auto refY = omTensorCreateWithShape<float>({S, D, B, H});
  auto refYh = omTensorCreateWithShape<float>({D, B, H});
  // Naive GRU implementation.
  // Equations for GRU.
  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // when linear_before_reset = 0 (means not linear before reset)
  //  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default,
  // when linear_before_reset != 0
  //  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) #
  // Ht = (1 - zt) (.) ht + zt (.) Ht-1
  auto &input = inputs.at(0);
  auto &initialH = inputs.at(1);
  auto weight = OMTensorUniquePtr(wOmt, omTensorDestroy);
  auto recurr = OMTensorUniquePtr(rOmt, omTensorDestroy);
  auto bias = OMTensorUniquePtr(bOmt, omTensorDestroy);

  // Initialize refYh and refYc.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++)
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH.get(), {d, b, h});

  // Main computation.
  for (int64_t d = 0; d < D; ++d)
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      auto XtWz = omTensorCreateWithShape<float>({B, H});
      auto XtWr = omTensorCreateWithShape<float>({B, H});
      auto XtWh = omTensorCreateWithShape<float>({B, H});
      auto HtRz = omTensorCreateWithShape<float>({B, H});
      auto HtRr = omTensorCreateWithShape<float>({B, H});
      auto HtRh = omTensorCreateWithShape<float>({B, H});
      auto RtHt = omTensorCreateWithShape<float>({B, H});
      auto RtHtRh = omTensorCreateWithShape<float>({B, H});
      auto rt = omTensorCreateWithShape<float>({B, H});
      auto zt = omTensorCreateWithShape<float>({B, H});
      for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWz, {b, h}) = 0;
          omTensorGetElem<float>(XtWr, {b, h}) = 0;
          omTensorGetElem<float>(XtWh, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input.get(), {seq, b, k});
            omTensorGetElem<float>(XtWz, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h, k});
            omTensorGetElem<float>(XtWr, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h + 1 * H, k});
            omTensorGetElem<float>(XtWh, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h + 2 * H, k});
          }
          omTensorGetElem<float>(HtRz, {b, h}) = 0;
          omTensorGetElem<float>(HtRr, {b, h}) = 0;
          omTensorGetElem<float>(HtRh, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRz, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr.get(), {d, h, k});
            omTensorGetElem<float>(HtRr, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurr.get(), {d, h + 1 * H, k});
            if (LinearBeforeReset != 0) {
              omTensorGetElem<float>(HtRh, {b, h}) +=
                  previousHt *
                  omTensorGetElem<float>(recurr.get(), {d, h + 2 * H, k});
            }
          }
        }

      for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++) {
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          omTensorGetElem<float>(zt, {b, h}) =
              sigmoid(omTensorGetElem<float>(XtWz, {b, h}) +
                      omTensorGetElem<float>(HtRz, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 3 * H}));
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
          omTensorGetElem<float>(rt, {b, h}) =
              sigmoid(omTensorGetElem<float>(XtWr, {b, h}) +
                      omTensorGetElem<float>(HtRr, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 1 * H}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 4 * H}));
          if (LinearBeforeReset == 0) {
            // rt (.) Ht-1
            float previousHt = omTensorGetElem<float>(refYh, {d, b, h});
            omTensorGetElem<float>(RtHt, {b, h}) =
                previousHt * omTensorGetElem<float>(rt, {b, h});
          }
        }

      // (rt (.) Ht-1)*(Rh^T)
      if (LinearBeforeReset == 0)
        for (int64_t b = 0; b < B; b++)
          for (int64_t h = 0; h < H; h++) {
            omTensorGetElem<float>(RtHtRh, {b, h}) = 0;
            for (int64_t k = 0; k < H; k++)
              omTensorGetElem<float>(RtHtRh, {b, h}) +=
                  omTensorGetElem<float>(RtHt, {b, k}) *
                  omTensorGetElem<float>(recurr.get(), {d, h + 2 * H, k});
          }

      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          // when linear_before_reset = 0
          //  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default
          // when linear_before_reset != 0
          //  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          float ht;
          if (LinearBeforeReset == 0)
            ht = tanh(omTensorGetElem<float>(XtWh, {b, h}) +
                      omTensorGetElem<float>(RtHtRh, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 5 * H}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 2 * H}));
          else
            ht = tanh(
                omTensorGetElem<float>(XtWh, {b, h}) +
                omTensorGetElem<float>(rt, {b, h}) *
                    (omTensorGetElem<float>(HtRh, {b, h}) +
                        omTensorGetElem<float>(bias.get(), {d, h + 5 * H})) +
                omTensorGetElem<float>(bias.get(), {d, h + 2 * H}));
          float previousHt = omTensorGetElem<float>(refYh, {d, b, h});
          float Ht = (1 - omTensorGetElem<float>(zt, {b, h})) * ht +
                     omTensorGetElem<float>(zt, {b, h}) * previousHt;
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          omTensorGetElem<float>(refY, {seq, d, b, h}) = Ht;
        }
      }
    }

  // onnx-mlir implementation.
  auto outputs = sess.run(move(inputs));
  auto &gruY = outputs.at(0);
  auto &gruYh = outputs.at(1);

  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  return (omTensorAreTwoOmtsClose<float>(gruY.get(), refY, rtol, atol) &&
          omTensorAreTwoOmtsClose<float>(gruYh.get(), refYh, rtol, atol));
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE.str()));
  setCompilerOptions({{OptionKind::CompilerOptLevel, "3"}});
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestGRU\n", nullptr, "TEST_ARGS");

  // RapidCheck test case generation.
  bool success = rc::check("GRU implementation correctness", []() {
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
    // LinearBeforeReset.
    const auto L = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for sequence.
    const auto isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const auto isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(isOMGRUTheSameAsNaiveImplFor(
        D, S, B, I, H, L, isDynS == 0, isDynB == 0));
  });
  if (!success)
    return 1;

  // Exhaustive test case generation.
  for (int64_t s = 3; s < 4; s++)
    for (int64_t b = 3; b < 4; b++)
      for (int64_t i = 2; i < 5; i++)
        for (int64_t h = 2; h < 5; h++)
          for (int64_t l = 0; l < 2; l++) {
            // Static dimensions.
            // forward
            assert(isOMGRUTheSameAsNaiveImplFor(1, s, b, i, h, l));
            // reverse
            assert(isOMGRUTheSameAsNaiveImplFor(-1, s, b, i, h, l));
            // bidirectional
            assert(isOMGRUTheSameAsNaiveImplFor(2, s, b, i, h, l));

            // Dynamic dimensions for sequence, batch size.
            // forward
            assert(isOMGRUTheSameAsNaiveImplFor(1, s, b, i, h, l, true, true));
            // reverse
            assert(isOMGRUTheSameAsNaiveImplFor(-1, s, b, i, h, l, true, true));
            // bidirectional
            assert(isOMGRUTheSameAsNaiveImplFor(2, s, b, i, h, l, true, true));
          }
  return 0;
}
