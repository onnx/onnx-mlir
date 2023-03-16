/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- GRUModel.cpp - Building GRU Models for tests -=============//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a GRU model and compiles it.
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

GRULibBuilder::GRULibBuilder(const std::string &modelName, const int direction,
    const int S, const int B, const int I, const int H,
    const int linearBeforeReset, const bool isDynamicS, const bool isDynamicB,
    const int layout)
    : RNNModelLibBuilder(modelName, layout), direction(direction), S(S), B(B),
      I(I), H(H), linearBeforeReset(linearBeforeReset), isDynamicS(isDynamicS),
      isDynamicB(isDynamicB), xShape(), hShape(), wOmt(nullptr), rOmt(nullptr),
      bOmt(nullptr) {}

GRULibBuilder::~GRULibBuilder() {
  omTensorDestroy(wOmt);
  omTensorDestroy(rOmt);
  omTensorDestroy(bOmt);
}

bool GRULibBuilder::build() {
  D = abs(direction);

  int64_t S1 = S, B1 = B;
  if (isDynamicS)
    S1 = ShapedType::kDynamic;
  if (isDynamicB)
    B1 = ShapedType::kDynamic;

  xShape = perm3(S, B, I);
  SmallVector<int64_t, 3> xShapeSymbol = perm3(S1, B1, I);
  SmallVector<int64_t, 3> wShape = {D, 3 * H, I};
  SmallVector<int64_t, 3> rShape = {D, 3 * H, H};
  SmallVector<int64_t, 2> bShape = {D, 6 * H};
  hShape = perm3(D, B, H);
  SmallVector<int64_t, 3> hShapeSymbol = perm3(D, B1, H);

  auto xType = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShapeSymbol, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());

  SmallVector<Type, 2> inputsType{xType, hType};
  SmallVector<Type, 2> outputsType{yType, yHType};

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

  int64_t linearBeforeResetArg = linearBeforeReset;

  wOmt = omTensorCreateWithRandomData<float>(llvm::ArrayRef(wShape), 0, 1);
  rOmt = omTensorCreateWithRandomData<float>(llvm::ArrayRef(rShape), 0, 1);
  bOmt = omTensorCreateWithRandomData<float>(llvm::ArrayRef(bShape), 0, 1);
  auto wConstant = buildONNXConstantOp(wOmt, wType);
  auto rConstant = buildONNXConstantOp(rOmt, rType);
  auto bConstant = buildONNXConstantOp(bOmt, bType);

  auto gruOp = builder.create<ONNXGRUOp>(loc,
      /*Y=*/yType, /*Y_h=*/yHType,
      /*X=*/xVal, /*W=*/wConstant, /*R=*/rConstant, /*B=*/bConstant,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/ArrayAttr(), /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr,
      /*layout=*/layout, /*linear_before_reset=*/linearBeforeResetArg);

  gruOp.getResults()[0].setType(yType);
  gruOp.getResults()[1].setType(yHType);

  builder.create<func::ReturnOp>(loc, gruOp.getResults());
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool GRULibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 2;
  OMTensor* list[num];
  list[0] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(xShape), dataRangeLB, dataRangeUB);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(hShape), dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1];
}

bool GRULibBuilder::prepareInputs() {
  return GRULibBuilder::prepareInputs(0.0, 1.0);
}

bool GRULibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool GRULibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;

  auto refY = omTensorCreateWithShape<float>(llvm::ArrayRef(perm4(S, D, B, H)));
  auto refYh = omTensorCreateWithShape<float>(llvm::ArrayRef(perm3(D, B, H)));
  // Naive GRU implementation.
  // Equations for GRU.
  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // when linear_before_reset = 0 (means not linear before reset)
  //  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default,
  // when linear_before_reset != 0
  //  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) #
  // Ht = (1 - zt) (.) ht + zt (.) Ht-1

  // Rename constant tensors with more explicit names.
  OMTensor *weight = wOmt;
  OMTensor *recurr = rOmt;
  OMTensor *bias = bOmt;
  // Get inputs and outputs.
  OMTensor *input = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *initialH = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *gruY = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *gruYh = omTensorListGetOmtByIndex(outputs, 1);

  // Initialize refYh.
  for (int64_t d = 0; d < D; d++)
    for (int64_t b = 0; b < B; b++)
      for (int64_t h = 0; h < H; h++) {
        std::vector<int64_t> p3 = llvm::ArrayRef(perm3(d, b, h));
        omTensorGetElem<float>(refYh, p3) =
            omTensorGetElem<float>(initialH, p3);
      }

  // Main computation.
  OMTensor *XtWz = omTensorCreateWithShape<float>({B, H});
  OMTensor *XtWr = omTensorCreateWithShape<float>({B, H});
  OMTensor *XtWh = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRz = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRr = omTensorCreateWithShape<float>({B, H});
  OMTensor *HtRh = omTensorCreateWithShape<float>({B, H});
  OMTensor *RtHt = omTensorCreateWithShape<float>({B, H});
  OMTensor *RtHtRh = omTensorCreateWithShape<float>({B, H});
  OMTensor *rt = omTensorCreateWithShape<float>({B, H});
  OMTensor *zt = omTensorCreateWithShape<float>({B, H});

  for (int64_t d = 0; d < D; ++d) {
    for (int64_t s = 0; s < S; ++s) {
      int64_t seq = s;
      if (d == 1 || direction == -1)
        // reverse
        seq = S - s - 1;
      for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++) {
          omTensorGetElem<float>(XtWz, {b, h}) = 0;
          omTensorGetElem<float>(XtWr, {b, h}) = 0;
          omTensorGetElem<float>(XtWh, {b, h}) = 0;
          for (int64_t k = 0; k < I; k++) {
            std::vector<int64_t> p3 = llvm::ArrayRef(perm3(seq, b, k));
            float xt = omTensorGetElem<float>(input, p3);
            omTensorGetElem<float>(XtWz, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h, k});
            omTensorGetElem<float>(XtWr, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h + 1 * H, k});
            omTensorGetElem<float>(XtWh, {b, h}) +=
                xt * omTensorGetElem<float>(weight, {d, h + 2 * H, k});
          }
          omTensorGetElem<float>(HtRz, {b, h}) = 0;
          omTensorGetElem<float>(HtRr, {b, h}) = 0;
          omTensorGetElem<float>(HtRh, {b, h}) = 0;
          for (int64_t k = 0; k < H; k++) {
            std::vector<int64_t> p3 = llvm::ArrayRef(perm3(d, b, k));
            float previousHt = omTensorGetElem<float>(refYh, p3);
            omTensorGetElem<float>(HtRz, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h, k});
            omTensorGetElem<float>(HtRr, {b, h}) +=
                previousHt * omTensorGetElem<float>(recurr, {d, h + 1 * H, k});
            if (linearBeforeReset != 0) {
              omTensorGetElem<float>(HtRh, {b, h}) +=
                  previousHt *
                  omTensorGetElem<float>(recurr, {d, h + 2 * H, k});
            }
          }
        }

      for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < H; h++) {
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          omTensorGetElem<float>(zt, {b, h}) =
              sigmoid(omTensorGetElem<float>(XtWz, {b, h}) +
                      omTensorGetElem<float>(HtRz, {b, h}) +
                      omTensorGetElem<float>(bias, {d, h}) +
                      omTensorGetElem<float>(bias, {d, h + 3 * H}));
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
          omTensorGetElem<float>(rt, {b, h}) =
              sigmoid(omTensorGetElem<float>(XtWr, {b, h}) +
                      omTensorGetElem<float>(HtRr, {b, h}) +
                      omTensorGetElem<float>(bias, {d, h + 1 * H}) +
                      omTensorGetElem<float>(bias, {d, h + 4 * H}));
          if (linearBeforeReset == 0) {
            // rt (.) Ht-1
            std::vector<int64_t> p3 = llvm::ArrayRef(perm3(d, b, h));
            float previousHt = omTensorGetElem<float>(refYh, p3);
            omTensorGetElem<float>(RtHt, {b, h}) =
                previousHt * omTensorGetElem<float>(rt, {b, h});
          }
        }

      // (rt (.) Ht-1)*(Rh^T)
      if (linearBeforeReset == 0)
        for (int64_t b = 0; b < B; b++)
          for (int64_t h = 0; h < H; h++) {
            omTensorGetElem<float>(RtHtRh, {b, h}) = 0;
            for (int64_t k = 0; k < H; k++)
              omTensorGetElem<float>(RtHtRh, {b, h}) +=
                  omTensorGetElem<float>(RtHt, {b, k}) *
                  omTensorGetElem<float>(recurr, {d, h + 2 * H, k});
          }

      for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
          // when linear_before_reset = 0
          //  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default
          // when linear_before_reset != 0
          //  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          float ht;
          if (linearBeforeReset == 0)
            ht = tanh(omTensorGetElem<float>(XtWh, {b, h}) +
                      omTensorGetElem<float>(RtHtRh, {b, h}) +
                      omTensorGetElem<float>(bias, {d, h + 5 * H}) +
                      omTensorGetElem<float>(bias, {d, h + 2 * H}));
          else
            ht = tanh(omTensorGetElem<float>(XtWh, {b, h}) +
                      omTensorGetElem<float>(rt, {b, h}) *
                          (omTensorGetElem<float>(HtRh, {b, h}) +
                              omTensorGetElem<float>(bias, {d, h + 5 * H})) +
                      omTensorGetElem<float>(bias, {d, h + 2 * H}));
          std::vector<int64_t> p3 = llvm::ArrayRef(perm3(d, b, h));
          float previousHt = omTensorGetElem<float>(refYh, p3);
          float Ht = (1 - omTensorGetElem<float>(zt, {b, h})) * ht +
                     omTensorGetElem<float>(zt, {b, h}) * previousHt;
          omTensorGetElem<float>(refYh, p3) = Ht;
          std::vector<int64_t> p4 = llvm::ArrayRef(perm4(seq, d, b, h));
          omTensorGetElem<float>(refY, p4) = Ht;
        }
      }
    }
  }
  omTensorDestroy(XtWz);
  omTensorDestroy(XtWr);
  omTensorDestroy(XtWh);
  omTensorDestroy(HtRz);
  omTensorDestroy(HtRr);
  omTensorDestroy(HtRh);
  omTensorDestroy(RtHt);
  omTensorDestroy(RtHtRh);
  omTensorDestroy(rt);
  omTensorDestroy(zt);

  bool ok = areCloseFloat(gruY, refY) && areCloseFloat(gruYh, refYh);
  omTensorDestroy(refY);
  omTensorDestroy(refYh);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
