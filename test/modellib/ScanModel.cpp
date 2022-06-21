/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ScanModel.cpp - Building Scan Models for tests -===========//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a Scan model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

#undef PRINT_TENSORS

using namespace mlir;

namespace onnx_mlir {
namespace test {

//
// Functions to build onnx.Scan and onnx.Scan9 op for testing.
//
// Operation:
//   onnx.Scan9
// Attribute:
//   Body: {onnx.Add}
//   num_scan_inputs: 1
//   scan_input_axis:
//   scan_input_directions: []
//   scan_output_axis:
//   scan_output_directions: []
// Inputs:
//   initial_state_and_scan_inputs: V
// Outputs:
//   final_state_and_scan_outputs: V
//
// Example Parameter:
//   B(=batch-size):1 x S(=sequence-length):3 x I(=inner-dim):2
// Variables:
//   initial_state_and_scan_outputs<BxI>
//   x<BxSxI>(Inputs) : [[1, 3, 5], [2, 4, 6]]<1x3x2>
//   y<BxI>(=last-state) : [9, 12]
//   z<BxSxI>(scan-output) : [[1, 4, 9], [2, 6, 12]]
// : [0, 0]<1x2>

//
// Similar to the onnx.Loop numerical test code, we use a model builder
// based on MLIR-representation in order to define onnx.Scan's body part
// in the simplest way.
// 

std::string testScanIdentityAdd = R"(
module {
  func @main_graph(%arg0: tensor<%Ixf32>, %arg1: tensor<%Sx%Ixf32>) ->
                   (tensor<%Ixf32>, tensor<%Sx%Ixf32>) {
    %1:2 = "onnx.Scan"(%arg0, %arg1) ({
    ^bb0(%body_arg0: tensor<%Ixf32>, %body_arg1: tensor<%Ixf32>):
      %2 = "onnx.Add"(%body_arg0, %body_arg1) :
           (tensor<%Ixf32>, tensor<%Ixf32>) -> tensor<%Ixf32>
      %3 = "onnx.Identity"(%2) : (tensor<%Ixf32>) -> tensor<%Ixf32>
      "onnx.Return"(%2, %3) : (tensor<%Ixf32>, tensor<%Ixf32>) -> ()
    }) {num_scan_inputs = 1 : si64} :
        (tensor<%Ixf32>, tensor<%Sx%Ixf32>)
        -> (tensor<%Ixf32>, tensor<%Sx%Ixf32>)
    "func.return"(%1#0, %1#1) : (tensor<%Ixf32>, tensor<%Sx%Ixf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
})";

ScanLibBuilder::ScanLibBuilder(const std::string &modelName,
    const int /*batch=*/B, const int /*seq=*/S, const int /*inner-dim=*/I)
    : ModelLibBuilder(modelName), B(B), S(S), I(I) {}

bool ScanLibBuilder::build() {
  initialShape = {B, I};
  xShape = {B, S, I};

  moduleIR = std::regex_replace(
      testScanIdentityAdd, std::regex("%B"), std::to_string(B));
  moduleIR = std::regex_replace(moduleIR, std::regex("%S"), std::to_string(S));
  moduleIR = std::regex_replace(moduleIR, std::regex("%I"), std::to_string(I));
  return true;
}

bool ScanLibBuilder::compileAndLoad() {
  OwningOpRef<ModuleOp> moduleOp =
      mlir::parseSourceString<ModuleOp>(moduleIR, &ctx);
  OwningOpRef<ModuleOp> module(std::move(moduleOp));
  if (compileModule(module, ctx, sharedLibBaseName, onnx_mlir::EmitLib) != 0)
    return false;
  exec = new ExecutionSession(
      onnx_mlir::getTargetFilename(sharedLibBaseName, onnx_mlir::EmitLib));
  return exec != nullptr;
}

bool ScanLibBuilder::compileAndLoad(const onnx_mlir::CompilerOptionList &list) {
  if (setCompilerOptions(list) != 0)
    return false;
  return compileAndLoad();
}

const static float omDefaultRangeBound = 1.0;
bool ScanLibBuilder::prepareInputs() {
  return ScanLibBuilder::prepareInputs(omDefaultRangeBound);
}

bool ScanLibBuilder::prepareInputs(float dataRange) {
  constexpr int num = 2;
  OMTensor **list = (OMTensor **)malloc(num * sizeof(OMTensor *));
  if (!list)
    return false;
  list[0] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(initialShape), -dataRange, dataRange);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::makeArrayRef(xShape), -dataRange, dataRange);
#ifdef SET_KNOWN_INPUT_VALUE
  // Compute reference. Scan with onnx.Add
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < I; ++i) {
      omTensorGetElem<float>(list[0], {b, i}) = 0;
    }
  }
  int n = 1;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t i = 0; i < I; ++i) {
        omTensorGetElem<float>(list[1], {b, s, i}) = (float) n++;
      }
    }
  }
#endif
  inputs = omTensorListCreateWithOwnership(list, num, true);
  return inputs && list[0] && list[1];
}

bool ScanLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *init = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *x = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *resy = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *resz = omTensorListGetOmtByIndex(outputs, 1);
  OMTensor *refy = omTensorCreateWithShape<float>({I});
  OMTensor *refz = omTensorCreateWithShape<float>({S, I});
  if (!init || !x || !resy || !refy || !resz || !refz)
    return false;
#ifdef PRINT_TENSORS
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < I; ++i) {
      printf("init<b=%ld, i=%ld>: %f\n", b, i,
          omTensorGetElem<float>(init, {b, i}));
    }
  }
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t i = 0; i < I; ++i) {
        printf("x<b=%ld, s=%ld, i=%ld>: %f\n", b, s, i,
            omTensorGetElem<float>(x, {b, s, i}));
      }
    }
  }
#endif
  // Compute reference. Scan with onnx.Add
  for (int64_t i = 0; i < I; ++i) {
    float refVal = 0.0;
    for (int64_t b = 0; b < B; ++b) {
      refVal = omTensorGetElem<float>(init, {b, i});
      for (int64_t s = 0; s < S; s++) {
        refVal += omTensorGetElem<float>(x, {b, s, i});
        omTensorGetElem<float>(refz, {s, i}) = refVal;
      }
    }
    omTensorGetElem<float>(refy, {i}) = refVal;
  }
#ifdef PRINT_TENSORS
  for (int64_t i = 0; i < I; ++i) {
    printf("resy/refy<i=%ld>: %f %f\n", i,
        omTensorGetElem<float>(resy, {i}),
        omTensorGetElem<float>(refy, {i}));
  }
  for (int64_t s = 0; s < S; ++s) {
    for (int64_t i = 0; i < I; ++i) {
      printf("resz/refz<s=%ld, i=%ld>: %f %f\n", s, i,
          omTensorGetElem<float>(resz, {s, i}),
          omTensorGetElem<float>(refz, {s, i}));
    }
  }
#endif
  bool ok = areCloseFloat(resy, refy) && areCloseFloat(resz, refz);
  omTensorDestroy(refy);
  omTensorDestroy(refz);
  return ok;
}
} // namespace test
} // namespace onnx_mlir
