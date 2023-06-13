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
      sorted(sorted), isNoneAxis(isNoneAxis),
      isNoneIndexOutput(isNoneIndexOutput) {}

UniqueLibBuilder::~UniqueLibBuilder() {
  // omTensorListDestroy(inputs);
  // omTensorListDestroy(outputs);
  // if (exec)
  //  delete exec;
}

bool UniqueLibBuilder::build() {
  if (rank != 2) { // XXX TODO support rank==3
    return false;
  }
  if ((axis < -rank) || (rank <= axis)) {
    return false;
  }
  xShape = {I, J};
  yShape = {I, J};
  int64_t int64_axis = -1;
  if (isNoneAxis) {
    yRank = 1;
    yShape = {-1};
  } else {
    int64_axis = (axis < 0) ? (rank + axis) : axis;
    yRank = rank;
    yShape = {I, J};
    yShape[int64_axis] = -1;
  }
  printf("UniqueLibBuilder::build: rank=%d, I=%d, J=%d, axis=%ld, "
         "sorted=%d, isNoneIndexOutput=%d\n",
      rank, I, J, int64_axis, sorted, isNoneIndexOutput);
  fflush(stdout);
  Type xType = RankedTensorType::get(xShape, builder.getI64Type());
  Type yType = UnrankedTensorType::get(builder.getI64Type());
  Type noneType = builder.getNoneType();
  Type IndexOutputType = (isNoneIndexOutput)
                             ? noneType
                             : UnrankedTensorType::get(builder.getI64Type());

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
  if ((I == 3) && (J == 3)) {
    list[0] = omTensorCreateWithShape<int64_t>({I, J});
    omTensorGetElem<int64_t>(list[0], {0, 0}) = 1;
    omTensorGetElem<int64_t>(list[0], {0, 1}) = 0;
    omTensorGetElem<int64_t>(list[0], {0, 2}) = 0;
    omTensorGetElem<int64_t>(list[0], {1, 0}) = 1;
    omTensorGetElem<int64_t>(list[0], {1, 1}) = 0;
    omTensorGetElem<int64_t>(list[0], {1, 2}) = 0;
    omTensorGetElem<int64_t>(list[0], {2, 0}) = 2;
    omTensorGetElem<int64_t>(list[0], {2, 1}) = 3;
    omTensorGetElem<int64_t>(list[0], {2, 2}) = 4;
  } else {
    list[0] = omTensorCreateWithRandomData<int64_t>(
        llvm::makeArrayRef(xShape), dataRangeLB, dataRangeUB);
  }
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
  OMTensor *y_res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ind_res = omTensorListGetOmtByIndex(outputs, 1);
  OMTensor *inv_ind_res = omTensorListGetOmtByIndex(outputs, 2);
  OMTensor *cnt_res = omTensorListGetOmtByIndex(outputs, 3);
  if (!x || !y_res || !ind_res)
    return false;
  int64_t int64_axis = isNoneAxis ? -1 : ((axis < 0) ? (rank + axis) : axis);
  // Count Unique elements
  OMTensor *total = omTensorCreateWithShape<int64_t>({1});
  omTensorUnique(total, x, int64_axis, (int64_t)sorted, NULL, NULL, NULL, NULL);
  int64_t int64_total = ((int64_t *)omTensorGetDataPtr(total))[0];
  OMTensor *y_ref, *ind_ref, *inv_ind_ref, *cnt_ref;
  if (int64_axis < 0) {
    y_ref = omTensorCreateWithShape<int64_t>({int64_total});
    ind_ref = omTensorCreateWithShape<int64_t>({int64_total});
    inv_ind_ref = omTensorCreateWithShape<int64_t>({I * J});
    cnt_ref = omTensorCreateWithShape<int64_t>({int64_total});
  } else if (int64_axis == 0) {
    y_ref = omTensorCreateWithShape<int64_t>({int64_total, J});
    ind_ref = omTensorCreateWithShape<int64_t>({int64_total, J});
    inv_ind_ref = omTensorCreateWithShape<int64_t>({I});
    cnt_ref = omTensorCreateWithShape<int64_t>({int64_total, J});
  } else if (int64_axis == 1) {
    y_ref = omTensorCreateWithShape<int64_t>({I, int64_total});
    ind_ref = omTensorCreateWithShape<int64_t>({I, int64_total});
    inv_ind_ref = omTensorCreateWithShape<int64_t>({J});
    cnt_ref = omTensorCreateWithShape<int64_t>({I, int64_total});
  } else {
    printf("UniqueLibBuilder::verifyOutputs: invalid axis==%ld\n", int64_axis);
    return false;
  }
  if (!y_ref)
    return false;
  // Compute reference.
  omTensorUnique(total, x, int64_axis, (int64_t)sorted, y_ref, ind_ref,
      inv_ind_ref, cnt_ref);
  printf("UniqueLibBuilder::verifyOutputs: rank=%d, I=%d, J=%d, axis=%ld, "
         "sorted=%d, isNoneIndexOutput=%d\n",
      rank, I, J, int64_axis, sorted, isNoneIndexOutput);
  omTensorPrint("INPUT=[\n", x);
  omTensorPrint("], Y_REF=[\n", y_ref);
  omTensorPrint("], Y_OUT=[\n", y_res);
#if 0
  omTensorPrint("], IND_REF=[\n", ind_ref);
  omTensorPrint("], IND_OUT=[\n", ind_res);
  omTensorPrint("], INV_IND_REF=[\n", inv_ind_ref);
  omTensorPrint("], INV_IND_OUT=[\n", inv_ind_res);
  omTensorPrint("], CNT_REF=[\n", cnt_ref);
  omTensorPrint("], CNT_OUT=[\n", cnt_res);
#endif
  printf("]\n");
  fflush(stdout);
  bool ok = areCloseFloat(y_res, y_ref);
#if 0
  ok &= areCloseFloat(ind_res, ind_ref);
  ok &= areCloseFloat(inv_ind_res, inv_ind_ref);
  ok &= areCloseFloat(cnt_res, cnt_ref);
#endif
  //omTensorDestroy(y_ref);
  //omTensorDestroy(ind_ref);
  //omTensorDestroy(inv_ind_ref);
  //omTensorDestroy(cnt_ref);
  return ok;
}

} // namespace test
} // namespace onnx_mlir
