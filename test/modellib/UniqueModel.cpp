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
#ifdef __cplusplus
#include "src/Runtime/OMUnique.hpp"
#endif

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
  if (!x || !res)
    return false;
  printf("UniqueLibBuilder::verifyOutputs: rank=%d, I=%d, J=%d, axis=%d, "
         "sorted=%ld, ", rank, I, J, axis, sorted);
  omTensorPrint("INPUT=[ ", x);
  omTensorPrint("], OUTPUT=[ ", res);
  printf("]\n");
  fflush(stdout);
  // Count Unique elements
  OMTensor *total = omTensorCreateWithShape<float>({1});
  omTensorUnique(total, x, (int64_t) axis, (int64_t) sorted, NULL, NULL, NULL,
      NULL);
  OMTensor *ref;
  if (axis < 0) {
    ref = omTensorCreateWithShape<float>({I, J});
  } else {
    ref = omTensorCreateWithShape<float>({I, J});
  }
  if (!ref)
    return false;
  // Compute reference.
  omTensorUnique(total, x, (int64_t) axis, (int64_t) sorted, res, NULL,
      NULL, NULL);
  bool ok = areCloseFloat(res, ref);
  omTensorDestroy(ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
