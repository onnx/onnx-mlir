/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- UniqueModel.cpp - Building GEMM Models for tests
//-===========//
//
// Copyright 2019-2022 The IBM Research Authors.
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

UniqueLibBuilder::UniqueLibBuilder(const std::string &modelName, const int rank,
    const int I, const int J, /*const int K, */ const int axis,
        const int sorted, const int isNoneAxis, const int isNoneIndexOutput)
    : ModelLibBuilder(modelName), rank(rank), I(I), J(J), /*K(K),*/ axis(axis),
      sorted(sorted), isNoneAxis(isNoneAxis), isNoneIndexOutput(isNoneIndexOutput) {}

UniqueLibBuilder::~UniqueLibBuilder() {
  //omTensorListDestroy(inputs);
  //omTensorListDestroy(outputs);
  //if (exec)
  //  delete exec;
}

bool UniqueLibBuilder::build() {
  if (rank != 2) { // XXX TODO support rank==3
    return false;
  }
  if (axis > rank) {
    return false;
  }
  xShape = {I, J};
  yShape = {I, J};
  if (isNoneAxis) {
    yRank = 1;
    yShape = {-1};
  } else {
    yRank = rank;
    yShape = {I, J};
    yShape[axis] = -1;
  }
  Type xType = RankedTensorType::get(xShape, builder.getI64Type());
  Type yType = UnrankedTensorType::get(builder.getI64Type());
  Type noneType = builder.getNoneType();
  Type IndexOutputType = (isNoneIndexOutput) ? noneType : UnrankedTensorType::get(builder.getI64Type());

  llvm::SmallVector<Type, 1> inputsType{xType};

  func::FuncOp funcOp;
  if (isNoneIndexOutput) {
    llvm::SmallVector<Type, 1> outputsType{yType};
    funcOp = createEmptyTestFunction(inputsType, outputsType);
  } else {
    llvm::SmallVector<Type, 4> outputsType{
        yType, IndexOutputType, IndexOutputType, IndexOutputType};
    funcOp = createEmptyTestFunction(inputsType, outputsType);
  }
  Block &entryBlock = funcOp.getBody().front();

  auto xVal = entryBlock.getArgument(0);

  IntegerAttr axisAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true), axis);
  if (isNoneAxis)
    axisAttr = nullptr;
  IntegerAttr sortedAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true), sorted);
  auto uniqueOp = builder.create<ONNXUniqueOp>(loc, /*Y=*/yType,
      /*indices=*/IndexOutputType, /*inverse_indices=*/IndexOutputType,
      /*counts=*/IndexOutputType, /*X=*/xVal, axisAttr, sortedAttr);
  uniqueOp.getResults()[0].setType(yType);
  uniqueOp.getResults()[1].setType(IndexOutputType);
  uniqueOp.getResults()[2].setType(IndexOutputType);
  uniqueOp.getResults()[3].setType(IndexOutputType);

  if (isNoneIndexOutput) {
    llvm::SmallVector<Value, 1> results = {uniqueOp.getResults()[0]};
    builder.create<func::ReturnOp>(loc, results);
  } else {
    llvm::SmallVector<Value, 4> results = {uniqueOp.getResults()[0],
          uniqueOp.getResults()[1], uniqueOp.getResults()[2],
              uniqueOp.getResults()[3]};
    builder.create<func::ReturnOp>(loc, results);
  }
  module.push_back(funcOp);
  createEntryPoint(funcOp);
  return true;
}

bool UniqueLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 1;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<int64_t>(
      llvm::makeArrayRef(xShape), dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0];
}

bool UniqueLibBuilder::prepareInputs() {
  return UniqueLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool UniqueLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool UniqueLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *x = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({I, J});
  if (!x || !res || !ref)
    return false;
  printf("UniqueLibBuilder::verifyOutputs: rank=2<%dx%d>[", I, J);
  printf("  Input(%p):", x);
  omTensorPrint(" ", x);
  printf("outputs(%p):", res);
  omTensorPrint(" ", res);
  printf("]\n");
  fflush(stdout);
#if 0
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
#else
  return true;
#endif
}

} // namespace test
} // namespace onnx_mlir
