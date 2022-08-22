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
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace test {

// =============================================================================
// 2D matmul without broadcast

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

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();
  auto aVal = entryBlock.getArgument(0);
  auto bVal = entryBlock.getArgument(1);

  auto MatmulOp = builder.create<ONNXMatMulOp>(loc,
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal);

  llvm::SmallVector<Value, 1> results = {MatmulOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool MatMul2DLibBuilder::prepareInputs() {
  constexpr int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>({I, K});
  list[1] = omTensorCreateWithRandomData<float>({K, J});
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

bool MatMul2DLibBuilder::prepareInputs(float dataRange) {
  constexpr int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>({I, K}, -dataRange, dataRange);
  list[1] = omTensorCreateWithRandomData<float>({K, J}, -dataRange, dataRange);
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
  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

// =============================================================================
// Matmul with broadcast in A or B but not both

MatMulSingleBroadcastLibBuilder::MatMulSingleBroadcastLibBuilder(
    const std::string &modelName, bool broadcastB,
    std::vector<int64_t> broadcastDims, const int I, const int J, const int K)
    : ModelLibBuilder(modelName), broadcastB(broadcastB),
      broadcastDims(broadcastDims), I(I), J(J), K(K) {}

bool MatMulSingleBroadcastLibBuilder::build() {
  aShape.clear();
  bShape.clear();
  cShape.clear();
  for (long int s : broadcastDims) {
    cShape.emplace_back(s);
    if (broadcastB)
      // B is being broadcasted, so A has a higher rank.
      aShape.emplace_back(s);
    else
      // A is being broadcasted, so B has a higher rank.
      bShape.emplace_back(s);
  }
  // Add I, K for A.
  aShape.emplace_back(I);
  aShape.emplace_back(K);
  // Add K, J for B.
  bShape.emplace_back(K);
  bShape.emplace_back(J);
  // Add I, J for C.
  cShape.emplace_back(I);
  cShape.emplace_back(J);

  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto yType = RankedTensorType::get(cShape, builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{aType, bType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();
  auto aVal = entryBlock.getArgument(0);
  auto bVal = entryBlock.getArgument(1);

  auto MatmulOp = builder.create<ONNXMatMulOp>(loc,
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal);

  llvm::SmallVector<Value, 1> results = {MatmulOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool MatMulSingleBroadcastLibBuilder::prepareInputs() {
  constexpr int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>(aShape);
  list[1] = omTensorCreateWithRandomData<float>(bShape);
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

// Vectors are copied as they will be modified in the function.
void MatMulSingleBroadcastLibBuilder::computeOneMatMul(OMTensor *a, OMTensor *b,
    OMTensor *c, std::vector<int64_t> aBroadcast,
    std::vector<int64_t> bBroadcast, std::vector<int64_t> cBroadcast) {
  int64_t aIndex = aBroadcast.size();
  int64_t bIndex = bBroadcast.size();
  int64_t cIndex = bBroadcast.size();
  // Do we have to recurse? Remove the last 2 dims that belong to matmul (i,j).
  int broadcastRank = cShape.size() - 2;
  if (cIndex < broadcastRank) {
    int64_t num = cShape[cIndex]; // Size that we need to iterate over.
    if (broadcastB) {
      // A has higher dim.
      for (int64_t i = 0; i < num; i++) {
        aBroadcast.emplace_back(i);
        cBroadcast.emplace_back(i);
        computeOneMatMul(a, b, c, aBroadcast, bBroadcast, cBroadcast);
      }
    } else {
      // B has higher dim.
      for (int64_t i = 0; i < num; i++) {
        bBroadcast.emplace_back(i);
        cBroadcast.emplace_back(i);
        computeOneMatMul(a, b, c, aBroadcast, bBroadcast, cBroadcast);
      }
    }
    // Done with recursion at this level.
    return;
  }
  // We have reached the recursion level where we can compute the matmul.
  aBroadcast.emplace_back(0); // aIndex+0: i
  aBroadcast.emplace_back(0); // aIndex+1: k
  bBroadcast.emplace_back(0); // bIndex+0: k
  bBroadcast.emplace_back(0); // bIndex+1: j
  cBroadcast.emplace_back(0); // cIndex+0: i
  cBroadcast.emplace_back(0); // cIndex+1: j
  // Compute reference, Matmul A * B.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      cBroadcast[cIndex + 0] = i;
      cBroadcast[cIndex + 1] = j;
      aBroadcast[aIndex + 0] = i;
      bBroadcast[bIndex + 1] = j;
      omTensorGetElem<float>(c, cBroadcast) = 0;
      for (int64_t k = 0; k < K; k++) {
        aBroadcast[aIndex + 1] = bBroadcast[bIndex + 0] = k;
        omTensorGetElem<float>(c, cBroadcast) +=
            omTensorGetElem<float>(a, aBroadcast) *
            omTensorGetElem<float>(b, bBroadcast);
      }
    }
  }
}

bool MatMulSingleBroadcastLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *b = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>(cShape);
  if (!a || !b || !res || !ref)
    return false;
  // Compute reference, Matmul A * B.
  computeOneMatMul(a, b, ref, {}, {}, {});

  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
