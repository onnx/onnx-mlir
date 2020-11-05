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

#define SHARED_LIB_BASE string("./TestRNN_main_graph")

using namespace std;

// Returns whether onnx-mlir compiled RNN is producing the same results as a
// naive implementation of RNN for a specific set of RNN
// parameters/configuration.
bool isOMRNNTheSameAsNaiveImplFor(
    const int direction, const int S, const int B, const int I, const int H) {
  MLIRContext ctx;
  registerDialects(ctx);

  int D = abs(direction);

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 3> xShape = {S, B, I};
  llvm::SmallVector<int64_t, 3> wShape = {D, H, I};
  llvm::SmallVector<int64_t, 3> rShape = {D, H, H};
  llvm::SmallVector<int64_t, 2> bShape = {D, 2 * H};
  llvm::SmallVector<int64_t, 3> hShape = {D, B, H};

  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 5> inputsType{xType, wType, rType, bType, hType};
  llvm::SmallVector<Type, 2> outputsType{yType, yHType};

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
  auto activationsAttr = builder.getStrArrayAttr({"Tanh", "Tanh"});

  auto rnnOp = builder.create<ONNXRNNOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*Y_h=*/yHType,
      /*X=*/xVal, /*W=*/wVal, /*R=*/rVal, /*B=*/bVal,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/activationsAttr, /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr);

  // Use the rnnOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in an inconsistent state.
  rnnOp.inferShapes();
  auto yOutputShape =
      rnnOp.getResults()[0].getType().cast<ShapedType>().getShape();
  auto SOut = yOutputShape[0];
  auto DOut = yOutputShape[1];
  auto BOut = yOutputShape[2];
  auto HOut = yOutputShape[3];
  rnnOp.getResults()[0].setType(yType);
  rnnOp.getResults()[1].setType(yHType);

  builder.create<ReturnOp>(UnknownLoc::get(&ctx), rnnOp.getResults());
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/5,
      /*numOutputs=*/2);
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

  // Naive RNN implementation.
  // Equations for RNN.
  // - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  auto refY = omTensorCreateWithShape<float>({SOut, DOut, BOut, HOut});
  auto refYh = omTensorCreateWithShape<float>({DOut, BOut, HOut});
  auto &input = inputs.at(0);
  auto &weight = inputs.at(1);
  auto &recurr = inputs.at(2);
  auto &bias = inputs.at(3);
  auto &initialH = inputs.at(4);

  // Initialize refYh.
  for (int64_t d = 0; d < DOut; d++)
    for (int64_t b = 0; b < BOut; b++)
      for (int64_t h = 0; h < HOut; h++)
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH.get(), {d, b, h});

  // Main computation.
  for (int64_t d = 0; d < DOut; ++d) {
    for (int64_t s = 0; s < SOut; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      auto XtWi = omTensorCreateWithShape<float>({BOut, HOut});
      auto HtRi = omTensorCreateWithShape<float>({BOut, HOut});
      for (int64_t b = 0; b < BOut; b++) {
        for (int64_t h = 0; h < HOut; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input.get(), {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weight.get(), {d, h, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          for (int64_t k = 0; k < HOut; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr.get(), {d, h, k});
          }
        }
      }
      for (int64_t b = 0; b < BOut; b++) {
        for (int64_t h = 0; h < HOut; h++) {
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
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  // RapidCheck test case generation.
  rc::check("RNN implementation correctness", []() {
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

    RC_ASSERT(isOMRNNTheSameAsNaiveImplFor(D, S, B, I, H));
  });

  // Exhaustive test case generation.
  for (int64_t s = 1; s < 5; s++)
    for (int64_t b = 1; b < 5; b++)
      for (int64_t i = 1; i < 5; i++)
        for (int64_t h = 1; h < 5; h++) {
          // forward
          assert(isOMRNNTheSameAsNaiveImplFor(1, s, b, i, h));
          // reverse
          assert(isOMRNNTheSameAsNaiveImplFor(-1, s, b, i, h));
          // bidirectional
          assert(isOMRNNTheSameAsNaiveImplFor(2, s, b, i, h));
        }
  return 0;
}
