/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- MatMulModel.cpp - Building MatMul Models for tests -=======//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a MatMul model and compiles it.
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

MatMul2DLibBuilder::MatMul2DLibBuilder(
    const std::string &modelName, const int I, const int J, const int K)
    : ModelLibBuilder(modelName), I(I), J(J), K(K) {}

bool MatMul2DLibBuilder::build() {
  llvm::SmallVector<int64_t, 4> aShape = {I, K};
  llvm::SmallVector<int64_t, 1> bShape = {K, J};
  llvm::SmallVector<int64_t, 4> cShape = {I, J};
  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto yType = RankedTensorType::get(cShape, builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{aType, bType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();
  auto aVal = entryBlock.getArgument(0);
  auto bVal = entryBlock.getArgument(1);

  auto MatmulOp = builder.create<ONNXMatMulOp>(loc,
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal);

  llvm::SmallVector<Value, 1> results = {MatmulOp.getResult()};
  builder.create<ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool MatMul2DLibBuilder::prepareInputs() {
  const int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>({I, K});
  list[1] = omTensorCreateWithRandomData<float>({K, J});
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

bool MatMul2DLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *b = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({I, J});
  if (!a || !b || !res || !ref)
    return false;
  // Compute reference, Matmul A * B.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(ref, {i, j}) = 0;
      for (int64_t k = 0; k < K; k++) {
        omTensorGetElem<float>(ref, {i, j}) +=
            omTensorGetElem<float>(a, {i, k}) *
            omTensorGetElem<float>(b, {k, j});
      }
    }
  }
  return areCloseFloat(res, ref);
}

} // namespace test
} // namespace onnx_mlir
