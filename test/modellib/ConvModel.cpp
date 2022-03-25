/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ConvModel.cpp - Building Conv Models for tests -===========//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a convolution model and compiles
// it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace test {

static int myCeil(int a, int b) { return ceil((1.0 * a) / (1.0 * b)); }
static int myFloor(int a, int b) { return floor((1.0 * a) / (1.0 * b)); }

Conv2DLibBuilder::Conv2DLibBuilder(const std::string &modelName, const int N,
    const int C, const int H, const int W, const int kH, const int kW,
    const ConvAutoPad autoPad, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd, const int stride, const int dilation,
    const int isDynamic)
    : ModelLibBuilder(modelName), N(N), C(C), H(H), W(W), kH(kH), kW(kW),
      autoPad(autoPad), pHBegin(pHBegin), pHEnd(pHEnd), pWBegin(pWBegin),
      pWEnd(pWEnd), stride(stride), dilation(dilation), isDynamic(isDynamic) {}

const std::string Conv2DLibBuilder::getAutoPadName(const ConvAutoPad autoPad) {
  static const std::string autoPadName[] = {
      "NOTSET", "VALID", "SAME_LOWER", "SAME_UPPER"};
  return autoPadName[autoPad];
}

bool Conv2DLibBuilder::build() {
  if (autoPad != ConvAutoPad::NOTSET) {
    // Make sure all pads are initially zero, only value tolarated.
    assert(pHBegin == 0 && pHEnd == 0 && pWBegin == 0 && pWEnd == 0);
  }

  // We use the Ns for the shape of the input, and the N1s for the construction
  // of the model. That way, when the shape is dynamic, we set the N1s to "-1"
  // (dynamic value) so that the compiler may not infer the size of the model,
  // and instead generate code to figure the sizes at run time.
  int N1 = N;
  int C1 = C;
  int H1 = H;
  int W1 = W;
  if (isDynamic)
    N1 = C1 = H1 = W1 = -1;

  llvm::SmallVector<int64_t, 4> xShape = {N, C, H, W};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {N1, C1, H1, W1};
  llvm::SmallVector<int64_t, 1> bShape = {C};
  llvm::SmallVector<int64_t, 4> wShape = {C, C, kH, kW};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto xTypeSymbol = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{xTypeSymbol, wType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  auto xVal = entryBlock.getArgument(0);
  auto wVal = entryBlock.getArgument(1);
  auto bVal = builder.create<ONNXNoneOp>(loc).getResult();

  auto dilations = builder.getI64ArrayAttr({dilation, dilation});
  auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
  auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
  auto strides = builder.getI64ArrayAttr({stride, stride});

  auto convOp = builder.create<ONNXConvOp>(loc,
      /*Y=*/yType,
      /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
      /*auto_pad=*/builder.getStringAttr(getAutoPadName(autoPad)),
      /*dilations=*/dilations,
      /*group=*/
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 1, /*isSigned=*/true)),
      /*kernel_shape=*/kernel_shape, /*pads=*/pads,
      /*strides=*/strides);

  // Use the convOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in a inconsistent state.
  convOp.X().setType(xType); // Use static dims to infer shape.
  LogicalResult res = convOp.inferShapes([](mlir::Region &) {});
  if (failed(res))
    return false;

  auto outputShape = convOp.getResult().getType().cast<ShapedType>().getShape();
  NOut = outputShape[0];
  COut = outputShape[1];
  HOut = outputShape[2];
  WOut = outputShape[3];
  convOp.getResult().setType(yType);
  convOp.X().setType(xTypeSymbol);

  llvm::SmallVector<Value, 1> results = {convOp.getResult()};
  builder.create<ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool Conv2DLibBuilder::prepareInputs() {
  const int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>({N, C, H, W});
  list[1] = omTensorCreateWithRandomData<float>({C, C, kH, kW});
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

bool Conv2DLibBuilder::verifyShapeAndComputeBeginEnd() {
  // Check first params.
  if (N != NOut) {
    std::cerr << "N mismatch: in " << N << ", out " << NOut << std::endl;
    return false;
  }
  if (C != COut) {
    std::cerr << "C mismatch: in " << C << ", out " << COut << std::endl;
    return false;
  }

  // Gather variables in arrays to match ONNX descriptions.
  int I[] = {H, W};
  int K[] = {kH, kW};
  int pBegin[] = {pHBegin, pWBegin};
  int pEnd[] = {pHEnd, pWEnd};
  int p[] = {pHBegin + pHEnd, pWBegin + pWEnd};
  int s[] = {stride, stride};
  int d[] = {dilation, dilation};
  int O[] = {HOut, WOut};

  // Check dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
  int myO[2], myPBegin[2], myPEnd[2];
  for (int i = 0; i < 2; ++i) {
    if (autoPad == ConvAutoPad::NOTSET) {
      // NOSET:
      //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
      myO[i] = myFloor((I[i] + p[i] - ((K[i] - 1) * d[i] + 1)), s[i]) + 1;
      myPBegin[i] = pBegin[i];
      myPEnd[i] = pEnd[i];
    } else if (autoPad == ConvAutoPad::VALID) {
      // VALID:
      // * O[i] = ceil((I[i] - ((K[i] - 1) * d[i] + 1) + 1) / s[i])
      // * P = 0
      myO[i] = myCeil((I[i] - ((K[i] - 1) * d[i] + 1) + 1), s[i]);
      myPBegin[i] = myPEnd[i] = 0;
    } else {
      // SAME_LOWER or SAME_UPPER:
      // * O[i] = ceil(I[i] / s[i])
      // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
      // * P[i] = p' / 2, if odd, first or second are increased by one.
      myO[i] = myCeil(I[i], s[i]);
      int pSum = (myO[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i];
      pSum = pSum >= 0 ? pSum : 0;
      myPBegin[i] = myPEnd[i] = pSum / 2;
      if (pSum % 2 != 0) {
        if (autoPad == ConvAutoPad::UPPER)
          myPEnd[i] += 1;
        else
          myPBegin[i] += 1;
      }
    }
    if (myO[i] != O[i]) {
      std::cerr << "output sizes mismatch: computed " << myO[i] << ", got "
                << O[i] << std::endl;
      return false;
    }
  }
  // Test all good, set padding values for computed ones.
  pHBegin = myPBegin[0];
  pWBegin = myPBegin[1];
  pHEnd = myPEnd[0];
  pWEnd = myPEnd[1];
  return true;
}

bool Conv2DLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *img = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *filter = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({NOut, COut, HOut, WOut});
  if (!img || !filter || !res || !ref)
    return false;
  if (!verifyShapeAndComputeBeginEnd())
    return false;
  // Compute reference.
  for (int64_t n = 0; n < NOut; n++)
    for (int64_t c = 0; c < COut; c++)
      for (int64_t h = 0; h < HOut; h++)
        for (int64_t w = 0; w < WOut; w++) {
          omTensorGetElem<float>(ref, {n, c, h, w}) = 0;
          for (int64_t ci = 0; ci < C; ci++)
            for (int64_t kh = 0; kh < kH; kh++)
              for (int64_t kw = 0; kw < kW; kw++)
                if ((h * stride + kh * dilation - pHBegin >= 0 &&
                        h * stride + kh * dilation - pHBegin < H) &&
                    (w * stride + kw * dilation - pWBegin >= 0 &&
                        w * stride + kw * dilation - pWBegin < W))
                  omTensorGetElem<float>(ref, {n, c, h, w}) +=
                      omTensorGetElem<float>(
                          img, {n, ci, h * stride + kh * dilation - pHBegin,
                                   w * stride + kw * dilation - pWBegin}) *
                      omTensorGetElem<float>(filter, {c, ci, kh, kw});
        }
  return areCloseFloat(res, ref);
}

} // namespace test
} // namespace onnx_mlir
