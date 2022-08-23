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
// Matmul with broadcast in A or B but not both.

MatMulSingleBroadcastLibBuilder::MatMulSingleBroadcastLibBuilder(
    const std::string &modelName, bool broadcastingB,
    std::vector<int64_t> broadcastDims, const int I, const int J, const int K)
    : ModelLibBuilder(modelName), broadcastingB(broadcastingB),
      broadcastDims(broadcastDims), I(I), J(J), K(K) {}

bool MatMulSingleBroadcastLibBuilder::build() {
  aShape.clear();
  bShape.clear();
  cShape.clear();
  for (long int s : broadcastDims) {
    cShape.emplace_back(s);
    if (broadcastingB)
      // B is being broadcasted, so B has a higher rank.
      bShape.emplace_back(s);
    else
      // A is being broadcasted, so A has a higher rank.
      aShape.emplace_back(s);
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
    OMTensor *c, std::vector<int64_t> &aIndexValues,
    std::vector<int64_t> &bIndexValues, std::vector<int64_t> &cIndexValues) {
  int64_t aIndex = aIndexValues.size();
  int64_t bIndex = bIndexValues.size();
  int64_t cIndex = cIndexValues.size();
  // Do we have to recurse? Remove the last 2 dims that belong to matmul (i,j).
  int broadcastRank = cShape.size() - 2;
  if (cIndex < broadcastRank) {
    int64_t num = cShape[cIndex]; // Size that we need to iterate over.
    cIndexValues.emplace_back(0); // Add broadcast index value.
    if (broadcastingB) {
      // B has higher dim.
      bIndexValues.emplace_back(0); // Add broadcast index value.
      for (int64_t i = 0; i < num; i++) {
        // Set the index of the matrix we are computing right now for the
        // index values b & c and recurse.
        bIndexValues[bIndex] = cIndexValues[bIndex] = i;
        computeOneMatMul(a, b, c, aIndexValues, bIndexValues, cIndexValues);
      }
      bIndexValues.pop_back(); // Remove broadcast index value.
    } else {
      // A has higher dim.
      aIndexValues.emplace_back(0); // Add broadcast index value.
      for (int64_t i = 0; i < num; i++) {
        // Set the index of the matrix we are computing right now for the
        // index values a & c and recurse.
        aIndexValues[bIndex] = cIndexValues[bIndex] = i;
        computeOneMatMul(a, b, c, aIndexValues, bIndexValues, cIndexValues);
      }
      aIndexValues.pop_back(); // Remove broadcast index value.
    }
    // Done with recursion at this level.
    cIndexValues.pop_back(); // Remove broadcast index value.
    return;
  }
  // We have reached the recursion level where we can compute the matmul.
  aIndexValues.emplace_back(0); // aIndex+0: i
  aIndexValues.emplace_back(0); // aIndex+1: k
  bIndexValues.emplace_back(0); // bIndex+0: k
  bIndexValues.emplace_back(0); // bIndex+1: j
  cIndexValues.emplace_back(0); // cIndex+0: i
  cIndexValues.emplace_back(0); // cIndex+1: j
  // Compute reference, Matmul A * B.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      cIndexValues[cIndex + 0] = i;
      cIndexValues[cIndex + 1] = j;
      aIndexValues[aIndex + 0] = i;
      bIndexValues[bIndex + 1] = j;
      omTensorGetElem<float>(c, cIndexValues) = 0;
      for (int64_t k = 0; k < K; k++) {
        aIndexValues[aIndex + 1] = bIndexValues[bIndex + 0] = k;
        omTensorGetElem<float>(c, cIndexValues) +=
            omTensorGetElem<float>(a, aIndexValues) *
            omTensorGetElem<float>(b, bIndexValues);
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
  std::vector<int64_t> aIndexValues, bIndexValues, cIndexValues;
  computeOneMatMul(a, b, ref, aIndexValues, bIndexValues, cIndexValues);

  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
