/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ConvModel.cpp - Building Conv Models for tests -===========//
//
// Copyright 2019-2023 The IBM Research Authors.
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
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace test {

static int myCeil(int a, int b) { return ceil((1.0 * a) / (1.0 * b)); }
static int myFloor(int a, int b) { return floor((1.0 * a) / (1.0 * b)); }

Conv2DLibBuilder::Conv2DLibBuilder(const std::string &modelName, const int N,
    const int CIn, const int COut, const int H, const int W, const int kH,
    const int kW, const ConvAutoPad autoPad, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd, const int stride, const int dilation,
    const int isDynamic)
    : ModelLibBuilder(modelName), N(N), CIn(CIn), COut(COut), H(H), W(W),
      kH(kH), kW(kW), autoPad(autoPad), pHBegin(pHBegin), pHEnd(pHEnd),
      pWBegin(pWBegin), pWEnd(pWEnd), stride(stride), dilation(dilation),
      isDynamic(isDynamic) {}

const std::string Conv2DLibBuilder::getAutoPadName(const ConvAutoPad autoPad) {
  static const std::string autoPadName[] = {
      "NOTSET", "VALID", "SAME_LOWER", "SAME_UPPER"};
  return autoPadName[autoPad];
}

bool Conv2DLibBuilder::build() {
  if (autoPad != ConvAutoPad::NOTSET) {
    // Make sure all pads are initially zero, only value tolerated.
    assert(pHBegin == 0 && pHEnd == 0 && pWBegin == 0 && pWEnd == 0);
  }

  // We use the Ns for the shape of the input, and the N1s for the construction
  // of the model. That way, when the shape is dynamic, we set the N1s to
  // "ShapedType::kDynamic" (dynamic value) so that the compiler may not infer
  // the size of the model, and instead generate code to figure the sizes at run
  // time.
  int64_t N1 = N;
  int64_t CIn1 = CIn;
  int64_t COut1 = COut;
  int64_t H1 = H;
  int64_t W1 = W;
  if (isDynamic)
    N1 = CIn1 = COut1 = H1 = W1 = ShapedType::kDynamic;

  llvm::SmallVector<int64_t, 4> xShape = {N, CIn, H, W};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {N1, CIn1, H1, W1};
  llvm::SmallVector<int64_t, 1> bShape = {COut};
  llvm::SmallVector<int64_t, 4> wShape = {COut, CIn, kH, kW};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto xTypeSymbol = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{xTypeSymbol, wType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  auto xVal = entryBlock.getArgument(0);
  auto wVal = entryBlock.getArgument(1);
  auto bVal = builder.create<ONNXNoneOp>(loc).getResult();

  auto dilations = builder.getI64ArrayAttr({dilation, dilation});
  auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
  auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
  auto strides = builder.getI64ArrayAttr({stride, stride});
  auto group = IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
      APInt(64, 1, /*isSigned=*/true));
  auto convOp = builder.create<ONNXConvOp>(loc,
      /*Y=*/yType,
      /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
      /*auto_pad=*/builder.getStringAttr(getAutoPadName(autoPad)),
      /*dilations=*/dilations,
      /*group=*/group,
      /*kernel_shape=*/kernel_shape, /*pads=*/pads,
      /*strides=*/strides);

  // Use the convOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in a inconsistent state.
  convOp.getX().setType(xType); // Use static dims to infer shape.
  LogicalResult res = convOp.inferShapes([](Region &) {});
  if (failed(res))
    return false;

  auto outputShape =
      mlir::cast<ShapedType>(convOp.getResult().getType()).getShape();
  modelNOut = outputShape[0];
  modelCOut = outputShape[1];
  modelHOut = outputShape[2];
  modelWOut = outputShape[3];
  convOp.getResult().setType(yType);
  convOp.getX().setType(xTypeSymbol);

  llvm::SmallVector<Value, 1> results = {convOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool Conv2DLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 2;
  OMTensor *list[num];
  list[0] = omTensorCreateWithRandomData<float>(
      {N, CIn, H, W}, dataRangeLB, dataRangeUB);
  list[1] = omTensorCreateWithRandomData<float>(
      {COut, CIn, kH, kW}, dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1];
}

bool Conv2DLibBuilder::prepareInputs() {
  return Conv2DLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool Conv2DLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool Conv2DLibBuilder::verifyShapeAndComputeBeginEnd() {
  // Check first params.
  if (N != modelNOut) {
    std::cerr << "N mismatch: in " << N << ", out " << modelNOut << std::endl;
    return false;
  }
  if (COut != modelCOut) {
    std::cerr << "C mismatch: in " << COut << ", out " << modelCOut
              << std::endl;
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
  int O[] = {modelHOut, modelWOut};

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
  OMTensor *ref = omTensorCreateWithShape<float>(
      {modelNOut, modelCOut, modelHOut, modelWOut});
  if (!img || !filter || !res || !ref)
    return false;
  if (!verifyShapeAndComputeBeginEnd())
    return false;
  // Compute reference.
  for (int64_t n = 0; n < modelNOut; n++)
    for (int64_t co = 0; co < modelCOut; co++)
      for (int64_t h = 0; h < modelHOut; h++)
        for (int64_t w = 0; w < modelWOut; w++) {
          omTensorGetElem<float>(ref, {n, co, h, w}) = 0;
          for (int64_t ci = 0; ci < CIn; ci++)
            for (int64_t kh = 0; kh < kH; kh++)
              for (int64_t kw = 0; kw < kW; kw++)
                if ((h * stride + kh * dilation - pHBegin >= 0 &&
                        h * stride + kh * dilation - pHBegin < H) &&
                    (w * stride + kw * dilation - pWBegin >= 0 &&
                        w * stride + kw * dilation - pWBegin < W))
                  omTensorGetElem<float>(ref, {n, co, h, w}) +=
                      omTensorGetElem<float>(
                          img, {n, ci, h * stride + kh * dilation - pHBegin,
                                   w * stride + kw * dilation - pWBegin}) *
                      omTensorGetElem<float>(filter, {co, ci, kh, kw});
        }
  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
