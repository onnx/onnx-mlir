/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==========-- LeakyReluModel.cpp - Building LeakyRelu Models for tests -=====//
//
// Copyright 2022,2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a model consisting of onnx.Add,
// onnx.LeakyRelu and onnx.Sub ops, and compiles it to check if the second
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

// =============================================================================
// Model consisting of onnx.Add, onnx.LeakyRelu and onnx.Sub ops

LeakyReluLibBuilder::LeakyReluLibBuilder(
    const std::string &modelName, const int N, const float alphaVal)
    : ModelLibBuilder(modelName), N(N), alphaVal(alphaVal) {}

bool LeakyReluLibBuilder::build() {
  llvm::SmallVector<int64_t, 1> xShape = {N};
  llvm::SmallVector<int64_t, 1> yShape = {N};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto yType = RankedTensorType::get(yShape, builder.getF32Type());

  llvm::SmallVector<Type, 1> inputsType{xType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();
  auto xVal = entryBlock.getArgument(0);

  FloatAttr alphaAttr = FloatAttr::get(builder.getF32Type(), alphaVal);
  auto addOp = builder.create<ONNXAddOp>(loc,
      /*Y=*/yType, /*X=*/xVal, /*X=*/xVal);
  auto leakyReluOp = builder.create<ONNXLeakyReluOp>(loc,
      /*Y=*/yType, /*X=*/addOp, /*alpha=*/alphaAttr);
  auto subOp = builder.create<ONNXSubOp>(loc,
      /*Y=*/yType, /*X=*/leakyReluOp, /*X=*/xVal);

  llvm::SmallVector<Value, 1> results = {subOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool LeakyReluLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 1;
  OMTensor* list[num];
  list[0] = omTensorCreateWithRandomData<float>({N}, dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0];
}

bool LeakyReluLibBuilder::prepareInputs() {
  return LeakyReluLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool LeakyReluLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool LeakyReluLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *x = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({N});
  if (!x || !res || !ref)
    return false;
  for (int64_t i = 0; i < N; ++i) {
    float val1 = omTensorGetElem<float>(x, {i}) * 2;
    float val2 = (val1 > 0.0) ? val1 : (val1 * alphaVal);
    float val3 = val2 - omTensorGetElem<float>(x, {i});
    omTensorGetElem<float>(ref, {i}) = val3;
  }
  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
