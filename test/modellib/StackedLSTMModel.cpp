/*
 * SPDX-License-Identifier: Apache-2.0
 */

//======-- StackedLSTMModel.cpp - Building StackedLSTM Models for tests -=====//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a Stacked LSTM model and compiles
// it.
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

StackedLSTMLibBuilder::StackedLSTMLibBuilder(const std::string &modelName,
    const int direction, const int S, const int B, const int I, const int H,
    const bool isDynamicS, const bool isDynamicB, const bool isNoneH,
    const bool isNoneC, const bool isNoneP)
    : ModelLibBuilder(modelName), direction(direction), S(S), B(B), I(I), H(H),
      isDynamicS(isDynamicS), isDynamicB(isDynamicB), isNoneH(isNoneH),
      isNoneC(isNoneC), isNoneP(isNoneP), xShape(), hShape(), cShape(),
      wOmt(nullptr), rOmt(nullptr), bOmt(nullptr), pOmt(nullptr),
      wFwdOmt(nullptr), rFwdOmt(nullptr), bFwdOmt(nullptr), pFwdOmt(nullptr) {}

StackedLSTMLibBuilder::~StackedLSTMLibBuilder() {
  omTensorDestroy(wOmt);
  omTensorDestroy(rOmt);
  omTensorDestroy(bOmt);
  omTensorDestroy(pOmt);
  omTensorDestroy(wFwdOmt);
  omTensorDestroy(rFwdOmt);
  omTensorDestroy(bFwdOmt);
  omTensorDestroy(pFwdOmt);
}

bool StackedLSTMLibBuilder::build() {
  D = abs(direction);
  DFwd = 1;
  int S1 = isDynamicS ? -1 : S;
  int B1 = isDynamicB ? -1 : B;

  // tensor shapes
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
  llvm::SmallVector<int64_t, 4> yShape = {S, D, B1, H};
  llvm::SmallVector<int64_t, 3> yHShape = {D, B1, H};
  llvm::SmallVector<int64_t, 3> yCShape = {D, B1, H};
  llvm::SmallVector<int64_t, 3> reshapedShape = {S, B, H * D};
  llvm::SmallVector<int64_t, 3> xFwdShape = {S1, B1, H * D};
  llvm::SmallVector<int64_t, 3> wFwdShape = {DFwd, 4 * H, H * D};
  llvm::SmallVector<int64_t, 3> rFwdShape = {DFwd, 4 * H, H};
  llvm::SmallVector<int64_t, 3> bFwdShape = {DFwd, 8 * H};
  llvm::SmallVector<int64_t, 2> pFwdShape = {DFwd, 3 * H};
  llvm::SmallVector<int64_t, 4> yFwdShape = {S, DFwd, B1, H};
  llvm::SmallVector<int64_t, 3> yHFwdShape = {DFwd, B1, H};
  llvm::SmallVector<int64_t, 3> yCFwdShape = {DFwd, B1, H};

  // input types
  auto xType = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShapeSymbol, builder.getF32Type());
  auto cType = RankedTensorType::get(cShapeSymbol, builder.getF32Type());
  auto pType = RankedTensorType::get(pShape, builder.getF32Type());
  auto wFwdType = RankedTensorType::get(wFwdShape, builder.getF32Type());

  // intermediate types
  auto yType = RankedTensorType::get(yShape, builder.getF32Type());
  auto yHType = RankedTensorType::get(yHShape, builder.getF32Type());
  auto yCType = RankedTensorType::get(yCShape, builder.getF32Type());
  auto reshapedType =
      RankedTensorType::get(reshapedShape, builder.getF32Type());
  auto rFwdType = RankedTensorType::get(rFwdShape, builder.getF32Type());
  auto bFwdType = RankedTensorType::get(bFwdShape, builder.getF32Type());
  auto pFwdType = RankedTensorType::get(pFwdShape, builder.getF32Type());

  // output types
  auto yFwdType = RankedTensorType::get(yFwdShape, builder.getF32Type());
  auto yHFwdType = RankedTensorType::get(yHFwdShape, builder.getF32Type());
  auto yCFwdType = RankedTensorType::get(yCFwdShape, builder.getF32Type());

  llvm::SmallVector<Type, 3> inputsType{xType, hType, cType};
  llvm::SmallVector<Type, 3> outputsType{yFwdType, yHFwdType, yCFwdType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  //
  // create the first ONNXLSTMOp
  //
  auto noneVal = builder.create<ONNXNoneOp>(loc).getResult();
  auto xVal = entryBlock.getArgument(0);
  auto sVal = noneVal;
  auto hVal = isNoneH ? noneVal : entryBlock.getArgument(1);
  auto cVal = isNoneC ? noneVal : entryBlock.getArgument(2);

  wOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape), 0, 1);
  rOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape), 0, 1);
  bOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape), 0, 1);
  pOmt = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(pShape), 0.0, (isNoneP) ? 0.0 : 1.0);
  auto wConstant = buildONNXConstantOp(wOmt, wType);
  auto rConstant = buildONNXConstantOp(rOmt, rType);
  auto bConstant = buildONNXConstantOp(bOmt, bType);
  auto pConstant = (isNoneP) ? noneVal : buildONNXConstantOp(pOmt, pType);

  const char *directionStr =
      (D == 1) ? "forward" : ((D == -1) ? "backward" : "bidirectional");
  StringAttr directionAttrFirst = builder.getStringAttr(directionStr);
  auto hiddenSizeAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, H, /*isSigned=*/true));
  auto inputForgetAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 0, /*isSigned=*/true));
  auto lstmOpFirst = builder.create<ONNXLSTMOp>(loc,
      /*Y=*/yType, /*Y_h=*/yHType, /*Y_c=*/yCType,
      /*X=*/xVal, /*W=*/wConstant, /*R=*/rConstant, /*B=*/bConstant,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*initial_c=*/cVal, /*P=*/pConstant,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/ArrayAttr(), /*clip=*/FloatAttr(),
      /*direction=*/directionAttrFirst, /*hidden_size=*/hiddenSizeAttr,
      /*input_forget=*/inputForgetAttr);
  lstmOpFirst.getResults()[0].setType(yType);
  lstmOpFirst.getResults()[1].setType(yHType);
  lstmOpFirst.getResults()[2].setType(yCType);
  //
  // craete ONNXConstantOp
  //
  SmallVector<int64_t, 1> constShape = {3};
  MemRefType memrefConst = MemRefType::get(constShape, builder.getF32Type());
  ShapedType constShapedType =
      RankedTensorType::get(constShape, builder.getI64Type());
  SmallVector<int64_t, 4> constVal = {S, B, H * D};
  ArrayRef<int64_t> constArray(constVal);
  DenseElementsAttr denseAttr = DenseElementsAttr::get<int64_t>(
      constShapedType, llvm::makeArrayRef(constVal));
  auto constantOp = builder.create<ONNXConstantOp>(loc,
      /*memRefTypeConst*/ memrefConst, nullptr, denseAttr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr);
  //
  // create ONNXReshapeOp
  //
  auto reshapeOp = builder.create<ONNXReshapeOp>(loc,
      /*reshaped=*/reshapedType, /*data=*/lstmOpFirst.getResults()[0],
      /*shape=*/constantOp.getResult());
  reshapeOp.getResult().setType(reshapedType);

  //
  // create the second ONNXLSTMOp (forward)
  //
  wFwdOmt =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wFwdShape), 0, 1);
  rFwdOmt =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rFwdShape), 0, 1);
  bFwdOmt =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bFwdShape), 0, 1);
  pFwdOmt = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(pFwdShape), 0.0, (isNoneP) ? 0.0 : 1.0);
  auto wFwdConstant = buildONNXConstantOp(wFwdOmt, wFwdType);
  auto rFwdConstant = buildONNXConstantOp(rFwdOmt, rFwdType);
  auto bFwdConstant = buildONNXConstantOp(bFwdOmt, bFwdType);
  auto pFwdConstant =
      (isNoneP) ? noneVal : buildONNXConstantOp(pFwdOmt, pFwdType);
  StringAttr directionAttrSecond = builder.getStringAttr("forward");
  auto lstmOpSecond = builder.create<ONNXLSTMOp>(loc,
      /*Y=*/yFwdType, /*Y_h=*/yHFwdType, /*Y_c=*/yCFwdType,
      /*X=*/reshapeOp.getResult(), /*W=*/wFwdConstant, /*R=*/rFwdConstant,
      /*B=*/bFwdConstant, /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*initial_c=*/cVal, /*P=*/pFwdConstant, /*activation_alpha=*/ArrayAttr(),
      /*activation_beta=*/ArrayAttr(), /*activations=*/ArrayAttr(),
      /*clip=*/FloatAttr(), /*direction=*/directionAttrSecond,
      /*hidden_size=*/hiddenSizeAttr, /*input_forget=*/inputForgetAttr);
  lstmOpSecond.getResults()[0].setType(yFwdType);
  lstmOpSecond.getResults()[1].setType(yHFwdType);
  lstmOpSecond.getResults()[2].setType(yCFwdType);

  builder.create<func::ReturnOp>(loc, lstmOpSecond.getResults());
  module.push_back(funcOp);
  createEntryPoint(funcOp);
  return true;
}

bool StackedLSTMLibBuilder::prepareInputs() {
  constexpr int num = 3;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] =
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(xShape), 0.0, 1.0);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(hShape), 0.0, (isNoneH) ? 0.0 : 1.0);
  list[2] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(cShape), 0.0, (isNoneC) ? 0.0 : 1.0);
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1] && list[2];
}

bool StackedLSTMLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;

  OMTensor *refY = omTensorCreateWithShape<float>({S, D, B, H});
  OMTensor *refYh = omTensorCreateWithShape<float>({D, B, H});
  OMTensor *refYc = omTensorCreateWithShape<float>({D, B, H});

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
  OMTensor *weightFwd = wFwdOmt;
  OMTensor *recurrFwd = rFwdOmt;
  OMTensor *biasFwd = bFwdOmt;
  OMTensor *peepholesFwd = pFwdOmt;

  OMTensor *input = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *initialH = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *initialC = omTensorListGetOmtByIndex(inputs, 2);
  OMTensor *lstmY = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *lstmYh = omTensorListGetOmtByIndex(outputs, 1);
  OMTensor *lstmYc = omTensorListGetOmtByIndex(outputs, 2);

  //
  // Main computation.
  //

#if 1
  // Initialize refYh and refYc.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++) {
        omTensorGetElem<float>(refYh, {d, b, h}) =
            omTensorGetElem<float>(initialH, {d, b, h});
        omTensorGetElem<float>(refYc, {d, b, h}) =
            omTensorGetElem<float>(initialC, {d, b, h});
      }
#endif

  auto XtWi = omTensorCreateWithShape<float>({B, H});
  auto XtWo = omTensorCreateWithShape<float>({B, H});
  auto XtWf = omTensorCreateWithShape<float>({B, H});
  auto XtWc = omTensorCreateWithShape<float>({B, H});
  auto HtRi = omTensorCreateWithShape<float>({B, H});
  auto HtRo = omTensorCreateWithShape<float>({B, H});
  auto HtRf = omTensorCreateWithShape<float>({B, H});
  auto HtRc = omTensorCreateWithShape<float>({B, H});

  // first LSTM
  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1)
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

  // second LSTM (forward)
  // new feature size = h(hidden_size) * D(num_directions)
  auto IFwd = H * D;
  auto inputFwd = omTensorCreateWithShape<float>({S, B, IFwd});
  auto refYFwd = omTensorCreateWithShape<float>({S, DFwd, B, H});
  auto refYhFwd = omTensorCreateWithShape<float>({DFwd, B, H});
  auto refYcFwd = omTensorCreateWithShape<float>({DFwd, B, H});

  // Initialize inputFwd
  for (int64_t s = 0; s < S; ++s) {
    for (int64_t d = 0; d < D; d++) {
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          int64_t idx = d * (B * H) + b * H + h;
          int64_t newB = idx / IFwd;
          int64_t newI = idx % IFwd;
          omTensorGetElem<float>(inputFwd, {s, newB, newI}) =
              omTensorGetElem<float>(refY, {s, d, b, h});
        }
      }
    }
  }

#if 1
  // Initialize refYhFwd and refYcFwd.
  for (int64_t d = 0; d < DFwd; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++) {
        omTensorGetElem<float>(refYhFwd, {d, b, h}) =
            omTensorGetElem<float>(initialH, {d, b, h});
        omTensorGetElem<float>(refYcFwd, {d, b, h}) =
            omTensorGetElem<float>(initialC, {d, b, h});
      }
#endif

  // forward LSTM
  for (int64_t d = 0; d < DFwd; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1)
        // reverse
        seq = S - s - 1;
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWi, {b, h}) = 0;
          omTensorGetElem<float>(XtWo, {b, h}) = 0;
          omTensorGetElem<float>(XtWf, {b, h}) = 0;
          omTensorGetElem<float>(XtWc, {b, h}) = 0;
          for (int64_t k = 0; k < IFwd; k++) {
            float xt = omTensorGetElem<float>(inputFwd, {seq, b, k});
            omTensorGetElem<float>(XtWi, {b, h}) +=
                xt * omTensorGetElem<float>(weightFwd, {d, h, k});
            omTensorGetElem<float>(XtWo, {b, h}) +=
                xt * omTensorGetElem<float>(weightFwd, {d, h + 1 * H, k});
            omTensorGetElem<float>(XtWf, {b, h}) +=
                xt * omTensorGetElem<float>(weightFwd, {d, h + 2 * H, k});
            omTensorGetElem<float>(XtWc, {b, h}) +=
                xt * omTensorGetElem<float>(weightFwd, {d, h + 3 * H, k});
          }
          omTensorGetElem<float>(HtRi, {b, h}) = 0;
          omTensorGetElem<float>(HtRo, {b, h}) = 0;
          omTensorGetElem<float>(HtRf, {b, h}) = 0;
          omTensorGetElem<float>(HtRc, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            float previousHt = omTensorGetElem<float>(refYhFwd, {d, b, k});
            omTensorGetElem<float>(HtRi, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurrFwd, {d, h, k});
            omTensorGetElem<float>(HtRo, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurrFwd, {d, h + 1 * H, k});
            omTensorGetElem<float>(HtRf, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurrFwd, {d, h + 2 * H, k});
            omTensorGetElem<float>(HtRc, {b, h}) +=
                previousHt *
                omTensorGetElem<float>(recurrFwd, {d, h + 3 * H, k});
          }
        }
      }
      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          float previousCt = omTensorGetElem<float>(refYcFwd, {d, b, h});
          // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
          float it = sigmoid(
              omTensorGetElem<float>(XtWi, {b, h}) +
              omTensorGetElem<float>(HtRi, {b, h}) +
              omTensorGetElem<float>(peepholesFwd, {d, h}) * previousCt +
              omTensorGetElem<float>(biasFwd, {d, h}) +
              omTensorGetElem<float>(biasFwd, {d, h + 4 * H}));
          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
          float ft =
              sigmoid(omTensorGetElem<float>(XtWf, {b, h}) +
                      omTensorGetElem<float>(HtRf, {b, h}) +
                      omTensorGetElem<float>(peepholesFwd, {d, h + 2 * H}) *
                          previousCt +
                      omTensorGetElem<float>(biasFwd, {d, h + 2 * H}) +
                      omTensorGetElem<float>(biasFwd, {d, h + 6 * H}));
          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          float ct = tanh(omTensorGetElem<float>(XtWc, {b, h}) +
                          omTensorGetElem<float>(HtRc, {b, h}) +
                          omTensorGetElem<float>(biasFwd, {d, h + 3 * H}) +
                          omTensorGetElem<float>(biasFwd, {d, h + 7 * H}));
          // Ct = ft (.) Ct-1 + it (.) ct
          float Ct = ft * previousCt + it * ct;
          omTensorGetElem<float>(refYcFwd, {d, b, h}) = Ct;
          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
          float ot = sigmoid(
              omTensorGetElem<float>(XtWo, {b, h}) +
              omTensorGetElem<float>(HtRo, {b, h}) +
              omTensorGetElem<float>(peepholesFwd, {d, h + 1 * H}) * Ct +
              omTensorGetElem<float>(biasFwd, {d, h + 1 * H}) +
              omTensorGetElem<float>(biasFwd, {d, h + 5 * H}));
          // Ht = ot (.) h(Ct)
          float Ht = ot * tanh(Ct);
          omTensorGetElem<float>(refYhFwd, {d, b, h}) = Ht;
          omTensorGetElem<float>(refYFwd, {seq, d, b, h}) = Ht;
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

  bool ok = areCloseFloat(lstmY, refYFwd) && areCloseFloat(lstmYh, refYhFwd) &&
            areCloseFloat(lstmYc, refYcFwd);

  omTensorDestroy(inputFwd);
  omTensorDestroy(refY);
  omTensorDestroy(refYh);
  omTensorDestroy(refYc);

  return ok;
}

} // namespace test
} // namespace onnx_mlir
