/*
 * SPDX-License-Identifier: Apache-2.0
 */

//========-- Elementwise.cpp - Building Elementwise Models for tests -========//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds some elementwise model and compiles
// it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

#include <functional>

using namespace mlir;

namespace onnx_mlir {
namespace test {

// =============================================================================
// Elementwise without broadcast
// Support following ops:
//    ONNXAddOp
//    ONNXDivOp
//    ONNXHardSigmoidOp
//    ONNXErFop

const float alphaVal = 2.0;
const float betaVal = 0.5;

Elementwise2DLibBuilder::Elementwise2DLibBuilder(const std::string &modelName,
    const std::string &onnxOpName, const int I, const int J)
    : ModelLibBuilder(modelName), onnxOpName(onnxOpName), I(I), J(J),
      inputNum(((onnxOpName.compare("ONNXHardSigmoidOp") == 0) ||
                   (onnxOpName.compare("ONNXErfOp") == 0))
                   ? 1
                   : 2) {}

bool Elementwise2DLibBuilder::build() {
  llvm::SmallVector<int64_t, 4> shape = {I, J};
  Type elementType = builder.getF32Type();
  auto aType = RankedTensorType::get(shape, elementType);
  auto bType = RankedTensorType::get(shape, elementType);
  auto yType = RankedTensorType::get(shape, elementType);

  llvm::SmallVector<Type, 2> inputsType{aType};
  if (inputNum > 1)
    inputsType.emplace_back(bType);
  llvm::SmallVector<Type, 1> outputsType{yType};

  func::FuncOp funcOp = createEmptyTestFunction(inputsType, outputsType);
  Block &entryBlock = funcOp.getBody().front();

  // Create operation.
  llvm::SmallVector<Value, 1> results;
  if (inputNum == 1) {
    // Unary operations.
    auto aVal = entryBlock.getArgument(0);
    if (onnxOpName.compare("ONNXHardSigmoidOp") == 0) {
      // Hard Sigmoid.
      FloatAttr alpha = builder.getFloatAttr(elementType, alphaVal);
      FloatAttr beta = builder.getFloatAttr(elementType, betaVal);
      auto op =
          builder.create<ONNXHardSigmoidOp>(loc, yType, aVal, alpha, beta);
      results.emplace_back(op.getResult());
    } else if (onnxOpName.compare("ONNXErfOp") == 0) {
      // Erf.
      auto op = builder.create<ONNXErfOp>(loc, yType, aVal);
      results.emplace_back(op.getResult());
    } else
      llvm_unreachable("unsupported unary elementwise op");

  } else if (inputNum == 2) {
    // Binary operations.
    auto aVal = entryBlock.getArgument(0);
    auto bVal = entryBlock.getArgument(1);
    if (onnxOpName.compare("ONNXAddOp") == 0) {
      // Add.
      auto op = builder.create<ONNXAddOp>(loc, yType, aVal, bVal);
      results.emplace_back(op.getResult());
    } else if (onnxOpName.compare("ONNXDivOp") == 0) {
      // Div.
      auto op = builder.create<ONNXDivOp>(loc, yType, aVal, bVal);
      results.emplace_back(op.getResult());
    } else
      llvm_unreachable("unsupported binary elementwise op");
  } else
    llvm_unreachable("support only unary and binary op");

  // Create function.
  builder.create<func::ReturnOp>(loc, results);
  module.push_back(funcOp);
  createEntryPoint(funcOp);
  return true;
}

bool Elementwise2DLibBuilder::prepareInputs(
    float dataRangeLB, float dataRangeUB) {
  constexpr int num = 2;
  assert(inputNum <= num && "bad constant");
  OMTensor *list[num];

  // Create elements in the list.
  if (inputNum == 1) {
    // Unary operation.
    list[0] =
        omTensorCreateWithRandomData<float>({I, J}, dataRangeLB, dataRangeUB);
    if (!list[0])
      return false;

  } else if (inputNum == 2) {
    // Binary operation.
    list[0] =
        omTensorCreateWithRandomData<float>({I, J}, dataRangeLB, dataRangeUB);
    list[1] =
        omTensorCreateWithRandomData<float>({I, J}, dataRangeLB, dataRangeUB);
    if (!list[0] || !list[1])
      return false;
  } else
    llvm_unreachable("support only unary and binary op");

  // Create actual list.
  inputs = omTensorListCreate(list, inputNum);
  return inputs;
}

bool Elementwise2DLibBuilder::prepareInputs() {
  return Elementwise2DLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool Elementwise2DLibBuilder::prepareInputsFromEnv(
    const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

using F1 = std::function<float(float)>;
using F2 = std::function<float(float, float)>;

bool Elementwise2DLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;

  OMTensor *res = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *ref = omTensorCreateWithShape<float>({I, J});
  if (!res || !ref)
    return false;

  if (inputNum == 1) {
    // Unary operations.
    OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
    if (!a)
      return false;
    // Compute reference
    F1 fct;
    if (onnxOpName.compare("ONNXHardSigmoidOp") == 0)
      fct = [](float a) -> float {
        float val = a * alphaVal + betaVal;
        val = (val > 0.0) ? val : 0.0;
        val = (val < 1.0) ? val : 1.0;
        return val;
      };
    else if (onnxOpName.compare("ONNXErfOp") == 0)
      fct = [](float a) -> float {
        float val = erf(a);
        return val;
      };
    else
      llvm_unreachable("unsupported binary elementwise op");
    for (int64_t i = 0; i < I; ++i) {
      for (int64_t j = 0; j < J; ++j) {
        float aa = omTensorGetElem<float>(a, {i, j});
        omTensorGetElem<float>(ref, {i, j}) = fct(aa);
      }
    }

  } else if (inputNum == 2) {
    // Binary operation.
    OMTensor *a = omTensorListGetOmtByIndex(inputs, 0);
    OMTensor *b = omTensorListGetOmtByIndex(inputs, 1);
    if (!a || !b)
      return false;
    // Compute reference
    F2 fct;
    if (onnxOpName.compare("ONNXAddOp") == 0)
      fct = [](float a, float b) -> float { return a + b; };
    else if (onnxOpName.compare("ONNXDivOp") == 0)
      fct = [](float a, float b) -> float { return a / b; };
    else
      llvm_unreachable("unsupported binary elementwise op");
    for (int64_t i = 0; i < I; ++i) {
      for (int64_t j = 0; j < J; ++j) {
        float aa = omTensorGetElem<float>(a, {i, j});
        float bb = omTensorGetElem<float>(b, {i, j});
        omTensorGetElem<float>(ref, {i, j}) = fct(aa, bb);
      }
    }
  } else
    llvm_unreachable("support only unary and binary op");

  // Check similarities.
  return areCloseFloat(res, ref);
}

} // namespace test
} // namespace onnx_mlir
