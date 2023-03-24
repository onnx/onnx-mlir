/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- MatMulModel.cpp - Building MatMul Models for tests -=======//
//
// Copyright 2019-2023 The IBM Research Authors.
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

#define DEBUG 0

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

bool MatMul2DLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 2;
  OMTensor *list[num];
  list[0] =
      omTensorCreateWithRandomData<float>({I, K}, dataRangeLB, dataRangeUB);
  list[1] =
      omTensorCreateWithRandomData<float>({K, J}, dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1];
}

bool MatMul2DLibBuilder::prepareInputs() {
  return MatMul2DLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool MatMul2DLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
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
// Matmul with broadcast in A or B but not both, or both have same static
// broadcast dims.

MatMulSingleBroadcastLibBuilder::MatMulSingleBroadcastLibBuilder(
    const std::string &modelName, bool broadcastingB, bool sameStaticBroadcast,
    std::vector<int64_t> broadcastDims, const int I, const int J, const int K)
    : ModelLibBuilder(modelName), broadcastingB(broadcastingB),
      sameStaticBroadcast(sameStaticBroadcast), broadcastDims(broadcastDims),
      I(I), J(J), K(K) {}

bool MatMulSingleBroadcastLibBuilder::build() {
  // Create shapes for a, b, and result y with broadcast.
  aShape.clear();
  bShape.clear();
  yShape.clear();
  // Init shapes of A, B, and Y.
  for (long int s : broadcastDims) {
    if (sameStaticBroadcast || !broadcastingB)
      // A is being broadcasted, so A has a higher rank.
      aShape.emplace_back(s);
    if (sameStaticBroadcast || broadcastingB)
      // B is being broadcasted, so B has a higher rank.
      bShape.emplace_back(s);
    yShape.emplace_back(s);
  }
  // Add I, K for A.
  aShape.emplace_back(I);
  aShape.emplace_back(K);
  // Add K, J for B.
  bShape.emplace_back(K);
  bShape.emplace_back(J);
  // Add I, J for C.
  yShape.emplace_back(I);
  yShape.emplace_back(J);
  // Create types.
  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto yType = RankedTensorType::get(yShape, builder.getF32Type());
  // Create function.
  llvm::SmallVector<Type, 2> inputsType{aType, bType};
  llvm::SmallVector<Type, 1> outputsType{yType};
  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();
  // Create op.
  auto aVal = entryBlock.getArgument(0);
  auto bVal = entryBlock.getArgument(1);
  auto MatmulOp = builder.create<ONNXMatMulOp>(loc,
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal);
  // Create function return.
  llvm::SmallVector<Value, 1> results = {MatmulOp.getResult()};
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);

  createEntryPoint(funcOp);
  return true;
}

bool MatMulSingleBroadcastLibBuilder::prepareInputs() {
  constexpr int num = 2;
  OMTensor *list[num];
  list[0] = omTensorCreateWithRandomData<float>(aShape);
  list[1] = omTensorCreateWithRandomData<float>(bShape);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1];
}

// Vectors are copied as they will be modified in the function.
void MatMulSingleBroadcastLibBuilder::computeOneMatMul(OMTensor *a, OMTensor *b,
    OMTensor *y, std::vector<int64_t> &aIndexValues,
    std::vector<int64_t> &bIndexValues, std::vector<int64_t> &yIndexValues) {
  int64_t aIndex = aIndexValues.size();
  int64_t bIndex = bIndexValues.size();
  int64_t yIndex = yIndexValues.size();
  // Do we have to recurse? Remove the last 2 dims that belong to matmul (i,j).
  int broadcastRank = yShape.size() - 2;
  if (yIndex < broadcastRank) {
    int64_t num = yShape[yIndex]; // Size that we need to iterate over.
    yIndexValues.emplace_back(0); // Add broadcast index value.
    if (sameStaticBroadcast) {
      // A & B have higher dim.
      aIndexValues.emplace_back(0); // Add broadcast index value.
      bIndexValues.emplace_back(0); // Add broadcast index value.
      for (int64_t i = 0; i < num; i++) {
        // Set the index of the matrix we are computing right now for the
        // index values a, b, & c and recurse.
        aIndexValues[aIndex] = bIndexValues[bIndex] = yIndexValues[bIndex] = i;
        computeOneMatMul(a, b, y, aIndexValues, bIndexValues, yIndexValues);
      }
      aIndexValues.pop_back(); // Remove broadcast index value.
      bIndexValues.pop_back(); // Remove broadcast index value.
    } else if (broadcastingB) {
      // B has higher dim.
      bIndexValues.emplace_back(0); // Add broadcast index value.
      for (int64_t i = 0; i < num; i++) {
        // Set the index of the matrix we are computing right now for the
        // index values b & c and recurse.
        bIndexValues[bIndex] = yIndexValues[bIndex] = i;
        computeOneMatMul(a, b, y, aIndexValues, bIndexValues, yIndexValues);
      }
      bIndexValues.pop_back(); // Remove broadcast index value.
    } else {
      // A has higher dim.
      aIndexValues.emplace_back(0); // Add broadcast index value.
      for (int64_t i = 0; i < num; i++) {
        // Set the index of the matrix we are computing right now for the
        // index values a & c and recurse.
        aIndexValues[aIndex] = yIndexValues[aIndex] = i;
        computeOneMatMul(a, b, y, aIndexValues, bIndexValues, yIndexValues);
      }
      aIndexValues.pop_back(); // Remove broadcast index value.
    }
    // Done with recursion at this level.
    yIndexValues.pop_back(); // Remove broadcast index value.
    return;
  }
  // We have reached the recursion level where we can compute the matmul.
  aIndexValues.emplace_back(0); // aIndex+0: i
  aIndexValues.emplace_back(0); // aIndex+1: k
  bIndexValues.emplace_back(0); // bIndex+0: k
  bIndexValues.emplace_back(0); // bIndex+1: j
  yIndexValues.emplace_back(0); // cIndex+0: i
  yIndexValues.emplace_back(0); // cIndex+1: j
  // Compute reference, Matmul A * B.
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      yIndexValues[yIndex + 0] = i;
      yIndexValues[yIndex + 1] = j;
      aIndexValues[aIndex + 0] = i;
      bIndexValues[bIndex + 1] = j;
      omTensorGetElem<float>(y, yIndexValues) = 0;
      for (int64_t k = 0; k < K; k++) {
        aIndexValues[aIndex + 1] = bIndexValues[bIndex + 0] = k;
        omTensorGetElem<float>(y, yIndexValues) +=
            omTensorGetElem<float>(a, aIndexValues) *
            omTensorGetElem<float>(b, bIndexValues);
      }
    }
  }
  aIndexValues.pop_back();
  aIndexValues.pop_back();
  bIndexValues.pop_back();
  bIndexValues.pop_back();
  yIndexValues.pop_back();
  yIndexValues.pop_back();
}

bool MatMulSingleBroadcastLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *b = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>(yShape);
  if (!a || !b || !res || !ref)
    return false;
  // Compute reference, Matmul A * B.
  std::vector<int64_t> aIndexValues, bIndexValues, cIndexValues;
  computeOneMatMul(a, b, ref, aIndexValues, bIndexValues, cIndexValues);

  bool ok = areCloseFloat(res, ref);
  if (DEBUG && !ok) {
    printf("After compute with errors\n");
    printTensor("A", a);
    printTensor("B", b);
    printTensor("Res", res);
    printTensor("Ref", ref);
  }
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
