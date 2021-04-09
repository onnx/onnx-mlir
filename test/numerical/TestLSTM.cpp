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

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/MainUtils.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#define SHARED_LIB_BASE string("./TestLSTM_main_graph")

using namespace std;

// Sigmoid
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

// Returns whether onnx-mlir compiled LSTM is producing the same results as a
// naive implementation of LSTM for a specific set of LSTM
// parameters/configuration.
bool isOMLSTMTheSameAsNaiveImplFor(const int direction, const int S,
    const int B, const int I, const int H, bool isDynamicS = false,
    bool isDynamicB = false) {
  MLIRContext ctx;
  registerDialects(ctx);

  int D = abs(direction);
  int S1 = S, B1 = B;
  if (isDynamicS)
    S1 = -1;
  if (isDynamicB)
    B1 = -1;

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 3> xShape = {S, B, I};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {S1, B1, I};
  llvm::SmallVector<int64_t, 3> wShape = {D, 4 * H, I};
  llvm::SmallVector<int64_t, 3> rShape = {D, 4 * H, H};
  llvm::SmallVector<int64_t, 2> bShape = {D, 8 * H};
  llvm::SmallVector<int64_t, 3> hShape = {D, B, H};
  llvm::SmallVector<int64_t, 3> hShapeSymbol = {D, B1, H};
  llvm::SmallVector<int64_t, 3> cShape = {D, B, H};
  llvm::SmallVector<int64_t, 3> cShapeSymbol = {D, B1, H};
  llvm::SmallVector<int64_t, 2> pShape = {D, 3 * H};

  auto xType = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShapeSymbol, builder.getF32Type());
  auto cType = RankedTensorType::get(cShapeSymbol, builder.getF32Type());
  auto pType = RankedTensorType::get(pShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());
  auto yCType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 7> inputsType{
      xType, wType, rType, bType, hType, cType, pType};
  llvm::SmallVector<Type, 3> outputsType{yType, yHType, yCType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto noneVal = builder
                     .create<mlir::ConstantOp>(
                         UnknownLoc::get(&ctx), builder.getUnitAttr())
                     .getResult();
  auto xVal = entryBlock->getArgument(0);
  auto wVal = entryBlock->getArgument(1);
  auto rVal = entryBlock->getArgument(2);
  auto bVal = entryBlock->getArgument(3);
  auto sVal = noneVal;
  auto hVal = entryBlock->getArgument(4);
  auto cVal = entryBlock->getArgument(5);
  auto pVal = entryBlock->getArgument(6);

  StringAttr directionAttr;
  if (direction == 1)
    directionAttr = builder.getStringAttr("forward");
  else if (direction == 2)
    directionAttr = builder.getStringAttr("bidirectional");
  else
    directionAttr = builder.getStringAttr("reverse");
  auto hiddenSizeAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, H, /*isSigned=*/true));
  auto inputForgetAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 0, /*isSigned=*/true));

  auto lstmOp = builder.create<ONNXLSTMOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*Y_h=*/yHType, /*Y_c=*/yCType,
      /*X=*/xVal, /*W=*/wVal, /*R=*/rVal, /*B=*/bVal,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*initial_c=*/cVal, /*P=*/pVal,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/ArrayAttr(), /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr,
      /*input_forget=*/inputForgetAttr);

  lstmOp.getResults()[0].setType(yType);
  lstmOp.getResults()[1].setType(yHType);
  lstmOp.getResults()[2].setType(yCType);

  builder.create<ReturnOp>(UnknownLoc::get(&ctx), lstmOp.getResults());
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/7,
      /*numOutputs=*/3,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);

  compileModule(moduleRef, ctx, SHARED_LIB_BASE, EmitLib);
  onnx_mlir::ExecutionSession sess(SHARED_LIB_BASE + ".so", "run_main_graph");

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto xOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(xShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(xOmt));
  auto wOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(wOmt));
  auto rOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(rOmt));
  auto bOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(bOmt));
  auto hOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(hShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(hOmt));
  auto cOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(cShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(cOmt));
  auto pOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(pShape), 0, 1),
      omTensorDestroy);
  inputs.emplace_back(move(pOmt));

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

  auto &input = inputs.at(0);
  auto &weight = inputs.at(1);
  auto &recurr = inputs.at(2);
  auto &bias = inputs.at(3);
  auto &initialH = inputs.at(4);
  auto &initialC = inputs.at(5);
  auto &peepholes = inputs.at(6);

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
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  // RapidCheck test case generation.
  rc::check("LSTM implementation correctness", []() {
    // The number of directions.
    // 1: forward, -1: reverse, 2: bidirectional
    const auto D = *rc::gen::element(1, -1, 2);
    // Sequence length.
    const auto S = *rc::gen::inRange(1, 5);
    // Batch size.
    const auto B = *rc::gen::inRange(5, 20);
    // Input size.
    const auto I = *rc::gen::inRange(20, 30);
    // Hidden size.
    const auto H = *rc::gen::inRange(30, 40);
    // Whether test dynamic dimension for sequence.
    const auto isDynS = *rc::gen::element(0, 1);
    // Whether test dynamic dimension for batch size.
    const auto isDynB = *rc::gen::element(0, 1);

    RC_ASSERT(
        isOMLSTMTheSameAsNaiveImplFor(D, S, B, I, H, isDynS == 0, isDynB == 0));
  });

  // Exhaustive test case generation.
  for (int64_t s = 2; s < 5; s++)
    for (int64_t b = 2; b < 5; b++)
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
