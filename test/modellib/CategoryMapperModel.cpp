/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- CategoryMapperModel.cpp - CategoryMapperLibBuilder implementation -===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the CategoryMapperLibBuilder class.
//
//===----------------------------------------------------------------------===//

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "category-mapper-test"

using namespace mlir;

namespace onnx_mlir {
namespace test {

// Utility function used to create an OMTensor.
template <typename T>
static OMTensor *createOMTensor(const ArrayRef<T> array, int64_t shape[],
    int64_t rank, OM_DATA_TYPE dtype) {
  return omTensorCreate(
      static_cast<void *>(const_cast<T *>(array.data())), shape, rank, dtype);
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::build() {
  int64_t shape[1] = {static_cast<int64_t>(input.size())};

  RankedTensorType inputType, outputType;

  if (std::is_same<T1, int64_t>::value) {
    inputType = RankedTensorType::get(shape, builder.getI64Type());
    outputType = RankedTensorType::get(shape, ONNXStringType::get(&ctx));
  } else {
    inputType = RankedTensorType::get(shape, ONNXStringType::get(&ctx));
    outputType = RankedTensorType::get(shape, builder.getI64Type());
  }

  createTestFunction(inputType, outputType, attributes);
  return true;
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::prepareInputs() {
  OMTensor **list = (OMTensor **)malloc(sizeof(OMTensor *));
  if (!list)
    return false;

  int64_t shape[1] = {static_cast<int64_t>(input.size())};
  list[0] = createOMTensor<T1>(input, shape, 1,
      (std::is_same<T1, int64_t>::value) ? ONNX_TYPE_INT64 : ONNX_TYPE_STRING);
  inputs = omTensorListCreateWithOwnership(list, 1, true);

  return inputs && list[0];
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::verifyOutputs() {
  if (!inputs || !outputs)
    return false;

  int64_t shape[1] = {static_cast<int64_t>(expOutput.size())};
  auto expOutputOMT = onnx_mlir::OMTensorUniquePtr(
      createOMTensor<T2>(expOutput, shape, 1 /*rank*/,
          (std::is_same<T2, int64_t>::value) ? ONNX_TYPE_INT64
                                             : ONNX_TYPE_STRING),
      omTensorDestroy);

  // Verify the result(s).
  OMTensor *output = omTensorListGetOmtByIndex(outputs, 0);
  if (!verifyResults(output, expOutputOMT.get()))
    return false;

  return true;
}

template <typename T1, typename T2>
void CategoryMapperLibBuilder<T1, T2>::createTestFunction(
    Type inputType, Type outputType, const CMAttributes &attributes) {
  SmallVector<Type, 1> inputsType{inputType}, outputsType{outputType};
  FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  createCategoryMapper(outputType, attributes, funcOp);
  createEntryPoint(funcOp);
}

template <typename T1, typename T2>
void CategoryMapperLibBuilder<T1, T2>::createCategoryMapper(
    Type outputType, const CMAttributes &attributes, FuncOp &funcOp) {
  Block &entryBlock = funcOp.getBody().front();
  BlockArgument input = entryBlock.getArgument(0);
  auto categoryMapperOp = builder.create<ONNXCategoryMapperOp>(loc, outputType,
      input, builder.getI64ArrayAttr(attributes.cat_int64s),
      builder.getStrArrayAttr(attributes.cat_strings), attributes.default_int,
      builder.getStringAttr(attributes.default_string));

  SmallVector<Value, 1> results = {categoryMapperOp.getResult()};
  builder.create<ReturnOp>(loc, results);
  module.push_back(funcOp);
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::verifyRank(
    const OMTensor &out, int64_t rank) const {
  if (omTensorGetRank(&out) == rank)
    return true;

  llvm::errs() << "Output tensor has rank " << omTensorGetRank(&out)
               << ", expecting " << rank << "\n";
  return false;
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::verifyNumElements(
    const OMTensor &out, int64_t numElems) const {
  if (omTensorGetNumElems(&out) == numElems)
    return true;

  llvm::errs() << "Output tensor has " << omTensorGetNumElems(&out)
               << " elements, expecting " << numElems << "\n";
  return false;
}

template <typename T>
static bool compareEqual(T val, T expectedVal) {
  return val == expectedVal;
}

template <>
bool compareEqual(const char *val, const char *expectedVal) {
  assert(val != nullptr && "Illegal val");
  assert(expectedVal != nullptr && "Illegal expectedVal");
  return strcmp(expectedVal, val) == 0;
}

template <typename T1, typename T2>
bool CategoryMapperLibBuilder<T1, T2>::verifyResults(
    const OMTensor *out, const OMTensor *expected) const {
  if (!verifyRank(*out, omTensorGetRank(expected)))
    return false;
  if (!verifyNumElements(*out, omTensorGetNumElems(expected)))
    return false;

  // Verify that the output tensor contains the expected result.
  const auto *outDataPtr = static_cast<T2 *>(omTensorGetDataPtr(out));
  const auto *expDataPtr = static_cast<T2 *>(omTensorGetDataPtr(expected));

  LLVM_DEBUG(llvm::dbgs() << "Result Verification:\n");
  for (int64_t i = 0; i < omTensorGetNumElems(out); ++i) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Got: " << outDataPtr[i] << ", expected: " << expDataPtr[i]
               << "\n");

    if (!compareEqual(outDataPtr[i], expDataPtr[i])) {
      llvm::errs() << "Output tensor contains \"" << outDataPtr[i]
                   << "\" at index = " << i << ", expecting \"" << expDataPtr[i]
                   << "\"\n";
      return false;
    }
  }

  return true;
}

template class CategoryMapperLibBuilder<int64_t, const char *>;
template class CategoryMapperLibBuilder<const char *, int64_t>;

} // namespace test
} // namespace onnx_mlir