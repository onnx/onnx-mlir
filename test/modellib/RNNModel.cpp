/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- RNNModel.cpp - Building RNN Models for tests -=============//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a RNN model and compiles it.
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

RNNLibBuilder::RNNLibBuilder(const std::string &modelName, const int direction,
    const int S, const int B, const int I, const int H, const bool isDynamicS,
    const bool isDynamicB)
    : ModelLibBuilder(modelName), direction(direction), S(S), B(B), I(I), H(H),
      isDynamicS(isDynamicS), isDynamicB(isDynamicB), xShape(), hShape(),
      wOmt(nullptr), rOmt(nullptr), bOmt(nullptr) {}

RNNLibBuilder::~RNNLibBuilder() {
  omTensorDestroy(wOmt);
  omTensorDestroy(rOmt);
  omTensorDestroy(bOmt);
}

bool RNNLibBuilder::build() {
  D = abs(direction);
  int S1 = S, B1 = B;
  if (isDynamicS)
    S1 = -1;
  if (isDynamicB)
    B1 = -1;

  xShape = {S, B, I};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {S1, B1, I};
  llvm::SmallVector<int64_t, 3> wShape = {D, H, I};
  llvm::SmallVector<int64_t, 3> rShape = {D, H, H};
  llvm::SmallVector<int64_t, 2> bShape = {D, 2 * H};
  hShape = {D, B, H};
  llvm::SmallVector<int64_t, 3> hShapeSymbol = {D, B1, H};

  auto xType = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShapeSymbol, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 5> inputsType{xType, hType};
  llvm::SmallVector<Type, 2> outputsType{yType, yHType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();
  auto xVal = entryBlock.getArgument(0);
  auto sVal = noneVal;
  auto hVal = entryBlock.getArgument(1);

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

  wOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape), 0, 1);
  rOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape), 0, 1);
  bOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape), 0, 1);
  auto wConstant = buildONNXConstantOp(wOmt, wType);
  auto rConstant = buildONNXConstantOp(rOmt, rType);
  auto bConstant = buildONNXConstantOp(bOmt, bType);

  auto rnnOp = builder.create<ONNXRNNOp>(loc,
      /*Y=*/yType, /*Y_h=*/yHType,
      /*X=*/xVal, /*W=*/wConstant, /*R=*/rConstant, /*B=*/bConstant,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/activationsAttr, /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr);

  rnnOp.getResults()[0].setType(yType);
  rnnOp.getResults()[1].setType(yHType);

  builder.create<func::ReturnOp>(loc, rnnOp.getResults());
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool RNNLibBuilder::prepareInputs() {
  constexpr int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(xShape), 0.0, 1.0);
  list[1] =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(hShape), 0.0, 1.0);
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

bool RNNLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;

  // Naive RNN implementation.
  // Equations for RNN.
  // - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)

  // Rename constant tensors with more explicit names.

  OMTensor *weight = wOmt;
  OMTensor *recurr = rOmt;
  OMTensor *bias = bOmt;
  // Get inputs and outputs.
  OMTensor *refY = omTensorCreateWithShape<float>({S, D, B, H});
  OMTensor *refYh = omTensorCreateWithShape<float>({D, B, H});
  OMTensor *input = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *initialH = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *rnnY = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *rnnYh = omTensorListGetOmtByIndex(outputs, 1);

  // Initialize refYh.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++)
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH, {d, b, h});

  // Main computation.
  OMTensor *XtWi = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRi = omTensorCreateWithShape<float>({B, H});
  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            float xt = omTensorGetElem<float>(input, {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYh, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h, k});
          }
        }
      }
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          // - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
          float Ht = tanh(omTensorGetElem<float>(XtWi, {b, h}) +
                          omTensorGetElem<float>(HtRi, {b, h}) +
                          omTensorGetElem<float>(bias, {d, h}) +
                          omTensorGetElem<float>(bias, {d, h + H}));
          omTensorGetElem<float>(refYh, {d, b, h}) = Ht;
          omTensorGetElem<float>(refY, {seq, d, b, h}) = Ht;
        }
      }
    }
  }
  omTensorDestroy(XtWi);
  omTensorDestroy(HtRi);

  bool ok = areCloseFloat(rnnY, refY) && areCloseFloat(rnnYh, refYh);
  omTensorDestroy(refY);
  omTensorDestroy(refYh);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
