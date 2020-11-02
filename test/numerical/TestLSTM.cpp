#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
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
bool isOMLSTMTheSameAsNaiveImplFor(
    const int S, const int B, const int I, const int H, const int D) {
  MLIRContext ctx;
  registerDialects(ctx);

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> xShape = {S, B, I};
  llvm::SmallVector<int64_t, 1> wShape = {D, 4 * H, I};
  llvm::SmallVector<int64_t, 1> rShape = {D, 4 * H, H};
  llvm::SmallVector<int64_t, 1> bShape = {D, 8 * H};

  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());
  auto yCType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 4> inputsType{xType, wType, rType, bType};
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
  auto sequenceLength = noneVal;
  auto initialH = noneVal;
  auto initialC = noneVal;
  auto peepholes = noneVal;

  StringAttr directionAttr;
  if (D == 1)
    directionAttr = builder.getStringAttr("forward");
  else if (D == 2)
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
      /*sequence_lens=*/sequenceLength, /*initial_h=*/initialH,
      /*initial_c=*/initialC, /*P=*/peepholes,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/ArrayAttr(), /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr,
      /*input_forget=*/inputForgetAttr);

  // Use the lstmOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in an inconsistent state.
  lstmOp.inferShapes();
  auto yOutputShape =
      lstmOp.getResults()[0].getType().cast<ShapedType>().getShape();
  auto SOut = yOutputShape[0];
  auto DOut = yOutputShape[1];
  auto BOut = yOutputShape[2];
  auto HOut = yOutputShape[3];
  lstmOp.getResults()[0].setType(yType);
  lstmOp.getResults()[1].setType(yHType);
  lstmOp.getResults()[2].setType(yCType);

  builder.create<ReturnOp>(UnknownLoc::get(&ctx), lstmOp.getResults());
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/4,
      /*numOutputs=*/3);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);

  compileModule(moduleRef, ctx, SHARED_LIB_BASE, EmitLib);
  onnx_mlir::ExecutionSession sess(SHARED_LIB_BASE + ".so", "run_main_graph");

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto xOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(xShape)),
      omTensorDestroy);
  inputs.emplace_back(move(xOmt));
  auto wOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape)),
      omTensorDestroy);
  inputs.emplace_back(move(wOmt));
  auto rOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape)),
      omTensorDestroy);
  inputs.emplace_back(move(rOmt));
  auto bOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape)),
      omTensorDestroy);
  inputs.emplace_back(move(bOmt));

  auto refY = omTensorCreateWithShape<float>({SOut, DOut, BOut, HOut});
  auto refYh = omTensorCreateWithShape<float>({DOut, BOut, HOut});
  auto refYc = omTensorCreateWithShape<float>({DOut, BOut, HOut});
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

  // Initialize refYh and refYc.
  for (int64_t d = 0; d < DOut; d++)
    for (int64_t b = 0; b < BOut; b++)
      for (int64_t h = 0; h < HOut; h++) {
        omTensorGetElem<float>(refYh, {d, b, h}) = 0;
        omTensorGetElem<float>(refYc, {d, b, h}) = 0;
      }

  // Main computation. 
  for (int d = 0; d < D; ++d) {
    for (int s = 0; s < S; ++s) {
      auto XtWi = omTensorCreateWithShape<float>({BOut, HOut});
      auto XtWo = omTensorCreateWithShape<float>({BOut, HOut});
      auto XtWf = omTensorCreateWithShape<float>({BOut, HOut});
      auto XtWc = omTensorCreateWithShape<float>({BOut, HOut});
      auto HtRi = omTensorCreateWithShape<float>({BOut, HOut});
      auto HtRo = omTensorCreateWithShape<float>({BOut, HOut});
      auto HtRf = omTensorCreateWithShape<float>({BOut, HOut});
      auto HtRc = omTensorCreateWithShape<float>({BOut, HOut});
      for (int64_t b = 0; b < BOut; b++) {
        for (int64_t h = 0; h < HOut; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          omTensorGetElem<float>(XtWo, {b, h}) = 0;
          omTensorGetElem<float>(XtWf, {b, h}) = 0;
          omTensorGetElem<float>(XtWc, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input.get(), {d, b, k});
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
      for (int64_t b = 0; b < BOut; b++) {
        for (int64_t h = 0; h < HOut; h++) {
          float previousCt = omTensorGetElem<float>(refYc, {d, b, h});
          // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
          float it =
              sigmoid(omTensorGetElem<float>(XtWi, {b, h}) +
                      omTensorGetElem<float>(HtRi, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 4 * H}));
          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 +
          // Wbf + Rbf)
          float ft =
              sigmoid(omTensorGetElem<float>(XtWf, {b, h}) +
                      omTensorGetElem<float>(HtRf, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 1 * H}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 5 * H}));
          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          float ct = tanh(omTensorGetElem<float>(XtWc, {b, h}) +
                          omTensorGetElem<float>(HtRc, {b, h}) +
                          omTensorGetElem<float>(bias.get(), {d, h + 3 * H}) +
                          omTensorGetElem<float>(bias.get(), {d, h + 7 * H}));
          // Ct = ft (.) Ct-1 + it (.) ct
          float Ct = ft * previousCt + it * ct;
          omTensorGetElem<float>(refYc, {d, b, h}) = Ct;
          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo
          float ot =
              sigmoid(omTensorGetElem<float>(XtWo, {b, h}) +
                      omTensorGetElem<float>(HtRo, {b, h}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 2 * H}) +
                      omTensorGetElem<float>(bias.get(), {d, h + 6 * H}));
          // Ht = ot (.) h(Ct)
          float Ht = ot * tanh(Ct);
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          if (d == 1)
            // forward
            omTensorGetElem<float>(refY, {s, d, b, h}) = Ht;
          else
            // backward
            omTensorGetElem<float>(refY, {S - s - 1, d, b, h}) = Ht;
        }
      }
    }
  }

  // onnx-mlir implementation.
  auto outputs = sess.run(move(inputs));
  auto &lstmY = outputs.at(0);
  auto &lstmYh = outputs.at(1);
  auto &lstmYc = outputs.at(2);

  return omTensorAreTwoOmtsClose<float>(lstmY.get(), refY) &&
         omTensorAreTwoOmtsClose<float>(lstmYh.get(), refYh) &&
         omTensorAreTwoOmtsClose<float>(lstmYc.get(), refYc);
}

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  // RapidCheck test case generation.
  rc::check("LSTM implementation correctness", []() {
    // Sequence length.
    const auto S = *rc::gen::inRange(1, 5);
    // Batch size.
    const auto B = *rc::gen::inRange(1, 10);
    // Input size.
    const auto I = *rc::gen::inRange(1, 10);
    // Hidden size.
    const auto H = *rc::gen::inRange(1, 20);
    // Number of directions: 1 or 2.
    const auto D = *rc::gen::inRange(1, 3);

    // Make sure the number of directions is 1 or 2
    RC_PRE((D == 1) || (D == 2));

    RC_ASSERT(isOMLSTMTheSameAsNaiveImplFor(S, B, I, H, D));
  });

  return 0;
}
