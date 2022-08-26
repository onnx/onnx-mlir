/*
 * SPDX-License-Identifier: Apache-2.0
 */

//=============-- LSTMModel.cpp - Building LSTM Models for tests -============//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds an LSTM model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace test {

/// Sigmoid
static float sigmoid(float x) { return 1 / (1 + exp(-x)); }

LSTMLibBuilder::LSTMLibBuilder(const std::string &modelName,
    const int direction, const int S, const int B, const int I, const int H,
    const bool isDynamicS, const bool isDynamicB, const bool isNoneH,
    const bool isNoneC, const bool isNoneP)
    : ModelLibBuilder(modelName), direction(direction), S(S), B(B), I(I), H(H),
      isDynamicS(isDynamicS), isDynamicB(isDynamicB), isNoneH(isNoneH),
      isNoneC(isNoneC), isNoneP(isNoneP), xShape(), hShape(), cShape(),
      wOmt(nullptr), rOmt(nullptr), bOmt(nullptr), pOmt(nullptr) {}

LSTMLibBuilder::~LSTMLibBuilder() {
  omTensorDestroy(wOmt);
  omTensorDestroy(rOmt);
  omTensorDestroy(bOmt);
  omTensorDestroy(pOmt);
}

bool LSTMLibBuilder::build() {
  D = abs(direction);
  int S1 = S, B1 = B;
  if (isDynamicS)
    S1 = -1;
  if (isDynamicB)
    B1 = -1;

  xShape = {S, B, I};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {S1, B1, I};
  llvm::SmallVector<int64_t, 3> wShape = {D, 4 * H, I};
  llvm::SmallVector<int64_t, 3> rShape = {D, 4 * H, H};
  llvm::SmallVector<int64_t, 2> bShape = {D, 8 * H};
  hShape = {D, B, H};
  llvm::SmallVector<int64_t, 3> hShapeSymbol = {D, B1, H};
  cShape = {D, B, H};
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

  llvm::SmallVector<Type, 3> inputsType{xType, hType, cType};
  llvm::SmallVector<Type, 3> outputsType{yType, yHType, yCType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();
  auto xVal = entryBlock.getArgument(0);
  auto hVal = (isNoneH) ? noneVal : entryBlock.getArgument(1);
  auto cVal = (isNoneC) ? noneVal : entryBlock.getArgument(2);
  auto sVal = noneVal;

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

  wOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape), 0, 1);
  rOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape), 0, 1);
  bOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape), 0, 1);
  pOmt = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(pShape), 0.0, (isNoneP) ? 0.0 : 1.0);
  auto wConstant = buildONNXConstantOp(wOmt, wType);
  auto rConstant = buildONNXConstantOp(rOmt, rType);
  auto bConstant = buildONNXConstantOp(bOmt, bType);
  auto pConstant = (isNoneP) ? noneVal : buildONNXConstantOp(pOmt, pType);

  auto lstmOp = builder.create<ONNXLSTMOp>(loc,
      /*Y=*/yType, /*Y_h=*/yHType, /*Y_c=*/yCType,
      /*X=*/xVal, /*W=*/wConstant, /*R=*/rConstant, /*B=*/bConstant,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*initial_c=*/cVal, /*P=*/pConstant,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/ArrayAttr(), /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr);

  lstmOp.getResults()[0].setType(yType);
  lstmOp.getResults()[1].setType(yHType);
  lstmOp.getResults()[2].setType(yCType);

  builder.create<func::ReturnOp>(loc, lstmOp.getResults());
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool LSTMLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 3;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  float dataRangeHLL = (isNoneH) ? 0.0 : dataRangeLB;
  float dataRangeHUL = (isNoneH) ? 0.0 : dataRangeUB;
  float dataRangeCLL = (isNoneC) ? 0.0 : dataRangeLB;
  float dataRangeCUL = (isNoneC) ? 0.0 : dataRangeUB;
  list[0] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(xShape), dataRangeLB, dataRangeUB);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(hShape), dataRangeHLL, dataRangeHUL);
  list[2] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(cShape), dataRangeCLL, dataRangeCUL);
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1] && list[2];
}

bool LSTMLibBuilder::prepareInputs() {
  return LSTMLibBuilder::prepareInputs(0.0, 1.0);
}

bool LSTMLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool LSTMLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;

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

  // Rename constant tensors with more explicit names.
  OMTensor *weight = wOmt;
  OMTensor *recurr = rOmt;
  OMTensor *bias = bOmt;
  OMTensor *peepholes = pOmt;
  // Get inputs and outputs.
  OMTensor *input = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *initialH = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *initialC = omTensorListGetOmtByIndex(inputs, 2);
  OMTensor *lstmY = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *lstmYh = omTensorListGetOmtByIndex(outputs, 1);
  OMTensor *lstmYc = omTensorListGetOmtByIndex(outputs, 2);

  // Initialize refYh and refYc.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++) {
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH, {d, b, h});
        omTensorGetElem<float>(refYc, {d, b, h}) =
            omTensorGetElem<float>(initialC, {d, b, h});
      }

  // Main computation.
  OMTensor *XtWi = omTensorCreateWithShape<float>({B, H});
  OMTensor *XtWo = omTensorCreateWithShape<float>({B, H});
  OMTensor *XtWf = omTensorCreateWithShape<float>({B, H});
  OMTensor *XtWc = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRi = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRo = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRf = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRc = omTensorCreateWithShape<float>({B, H});
  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          omTensorGetElem<float>(XtWo, {b, h}) = 0;
          omTensorGetElem<float>(XtWf, {b, h}) = 0;
          omTensorGetElem<float>(XtWc, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input, {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h, k});
            omTensorGetElem<float>(XtWo, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h + 1 * H, k});
            omTensorGetElem<float>(XtWf, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h + 2 * H, k});
            omTensorGetElem<float>(XtWc, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h + 3 * H, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          omTensorGetElem<float>(HtRo, {b, h}) = 0;
          omTensorGetElem<float>(HtRf, {b, h}) = 0;
          omTensorGetElem<float>(HtRc, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h, k});
            omTensorGetElem<float>(HtRo, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h + 1 * H, k});
            omTensorGetElem<float>(HtRf, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h + 2 * H, k});
            omTensorGetElem<float>(HtRc, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h + 3 * H, k});
          }
        }
      }
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          float previousCt = omTensorGetElem<float>(refYc, {d, b, h});
          // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
          float it =
              sigmoid(omTensorGetElem<float>(XtWi, {b, h}) +
                      omTensorGetElem<float>(HtRi, {b, h}) +
                      omTensorGetElem<float>(peepholes, {d, h}) * previousCt +
                      omTensorGetElem<float>(bias, {d, h}) +
                      omTensorGetElem<float>(bias, {d, h + 4 * H}));
          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
          float ft = sigmoid(
              omTensorGetElem<float>(XtWf, {b, h}) +
              omTensorGetElem<float>(HtRf, {b, h}) +
              omTensorGetElem<float>(peepholes, {d, h + 2 * H}) * previousCt +
              omTensorGetElem<float>(bias, {d, h + 2 * H}) +
              omTensorGetElem<float>(bias, {d, h + 6 * H}));
          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          float ct = tanh(omTensorGetElem<float>(XtWc, {b, h}) +
                          omTensorGetElem<float>(HtRc, {b, h}) +
                          omTensorGetElem<float>(bias, {d, h + 3 * H}) +
                          omTensorGetElem<float>(bias, {d, h + 7 * H}));
          // Ct = ft (.) Ct-1 + it (.) ct
          float Ct = ft * previousCt + it * ct;
          omTensorGetElem<float>(refYc, {d, b, h}) = Ct;
          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
          float ot =
              sigmoid(omTensorGetElem<float>(XtWo, {b, h}) +
                      omTensorGetElem<float>(HtRo, {b, h}) +
                      omTensorGetElem<float>(peepholes, {d, h + 1 * H}) * Ct +
                      omTensorGetElem<float>(bias, {d, h + 1 * H}) +
                      omTensorGetElem<float>(bias, {d, h + 5 * H}));
          // Ht = ot (.) h(Ct)
          float Ht = ot * tanh(Ct);
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          omTensorGetElem<float>(refY, {seq, d, b, h}) = Ht;
        }
      }
    }
  }
  omTensorDestroy(XtWi);
  omTensorDestroy(XtWo);
  omTensorDestroy(XtWf);
  omTensorDestroy(XtWc);
  omTensorDestroy(HtRi);
  omTensorDestroy(HtRo);
  omTensorDestroy(HtRf);
  omTensorDestroy(HtRc);

  bool ok = areCloseFloat(lstmY, refY) && areCloseFloat(lstmYh, refYh) &&
            areCloseFloat(lstmYc, refYc);
  omTensorDestroy(refY);
  omTensorDestroy(refYh);
  omTensorDestroy(refYc);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
