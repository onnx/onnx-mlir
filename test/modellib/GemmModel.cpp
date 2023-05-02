/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- GemmModel.cpp - Building GEMM Models for tests -===========//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a GEMM model and compiles it.
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

GemmLibBuilder::GemmLibBuilder(const std::string &modelName, const int I,
    const int J, const int K, const int aTrans, const int bTrans,
    const int cRank, const float alphaVal, const float betaVal)
    : ModelLibBuilder(modelName), I(I), J(J), K(K), aTrans(aTrans),
      bTrans(bTrans), cRank(cRank), alphaVal(alphaVal), betaVal(betaVal) {}

bool GemmLibBuilder::build() {
  aShape = {I, K};
  if (aTrans)
    aShape = {K, I};
  bShape = {K, J};
  if (bTrans)
    bShape = {J, K};
  cShape = {J};
  if (cRank == 2)
    cShape = {I, J};
  else
    assert(cRank == 1 && "cRank == 1 or 2");

  llvm::SmallVector<int64_t, 2> yShape = {I, J};
  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto cType = RankedTensorType::get(cShape, builder.getF32Type());
  auto yType = RankedTensorType::get(yShape, builder.getF32Type());

  llvm::SmallVector<Type, 3> inputsType{aType, bType, cType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  auto aVal = entryBlock.getArgument(0);
  auto bVal = entryBlock.getArgument(1);
  auto cVal = entryBlock.getArgument(2);

  FloatAttr alphaAttr = FloatAttr::get(builder.getF32Type(), alphaVal);
  FloatAttr betaAttr = FloatAttr::get(builder.getF32Type(), betaVal);
  IntegerAttr aTransAttr =
      IntegerAttr::get(builder.getIntegerType(64, true), aTrans);
  IntegerAttr bTransAttr =
      IntegerAttr::get(builder.getIntegerType(64, true), bTrans);
  auto gemmOp = builder.create<ONNXGemmOp>(loc,
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal, /*C=*/cVal, alphaAttr, betaAttr,
      aTransAttr, bTransAttr);
  gemmOp.getResult().setType(yType);

  llvm::SmallVector<Value, 1> results = {gemmOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool GemmLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 3;
  OMTensor* list[num];
  list[0] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(aShape), dataRangeLB, dataRangeUB);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(bShape), dataRangeLB, dataRangeUB);
  list[2] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(cShape), dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1] && list[2];
}

bool GemmLibBuilder::prepareInputs() {
  return GemmLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool GemmLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool GemmLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *b = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *c = omTensorListGetOmtByIndex(inputs, 2);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({I, J});
  if (!a || !b || !c || !res || !ref)
    return false;
  // Compute reference.
  // Matmul A * B.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(ref, {i, j}) = 0;
      for (int64_t k = 0; k < K; k++) {
        float aVal, bVal;
        if (aTrans == 0)
          aVal = omTensorGetElem<float>(a, {i, k});
        else
          aVal = omTensorGetElem<float>(a, {k, i});
        if (bTrans == 0)
          bVal = omTensorGetElem<float>(b, {k, j});
        else
          bVal = omTensorGetElem<float>(b, {j, k});
        omTensorGetElem<float>(ref, {i, j}) += aVal * bVal;
      }
    }
  }
  // Add C.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      float cVal;
      if (cRank == 1)
        cVal = omTensorGetElem<float>(c, {j});
      else if (cRank == 2)
        cVal = omTensorGetElem<float>(c, {i, j});
      else
        assert(false);
      omTensorGetElem<float>(ref, {i, j}) =
          alphaVal * omTensorGetElem<float>(ref, {i, j}) + betaVal * cVal;
    }
  }
  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
