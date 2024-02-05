/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ScanModel.cpp - Building Scan Models for tests -===========//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a Scan model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.hpp"
#include "test/modellib/ModelLib.hpp"

#include <regex>

#undef PRINT_TENSORS

using namespace mlir;

namespace onnx_mlir {
namespace test {

//
// Functions to build onnx.ScanV8 and onnx.Scan op for testing.
//
// Operation:
//   onnx.ScanV8
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

std::string testScanIdentityAddV8 = R"(
module {
  func.func @main_graph(%arg0: tensor<%Bx%Ixf32>, %arg1: tensor<%Bx%Sx%Ixf32>) ->
                   (tensor<%Bx%Ixf32>, tensor<%Bx%Sx%Ixf32>) {
    %1:2 = "onnx.ScanV8"(%arg0, %arg1) ({
    ^bb0(%body_arg0: tensor<%Bx%Ixf32>, %body_arg1: tensor<%Bx%Ixf32>):
      %2 = "onnx.Add"(%body_arg0, %body_arg1) :
           (tensor<%Bx%Ixf32>, tensor<%Bx%Ixf32>) -> tensor<%Bx%Ixf32>
      %3 = "onnx.Identity"(%2) : (tensor<%Bx%Ixf32>) -> tensor<%Bx%Ixf32>
      "onnx.Yield"(%2, %3) : (tensor<%Bx%Ixf32>, tensor<%Bx%Ixf32>) -> ()
    }) {num_scan_inputs = 1 : si64} :
        (tensor<%Bx%Ixf32>, tensor<%Bx%Sx%Ixf32>)
        -> (tensor<%Bx%Ixf32>, tensor<%Bx%Sx%Ixf32>)
    "func.return"(%1#0, %1#1) : (tensor<%Bx%Ixf32>, tensor<%Bx%Sx%Ixf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
})";

std::string testScanIdentityAdd = R"(
module {
  func.func @main_graph(%arg0: tensor<%Ixf32>, %arg1: tensor<%Sx%Ixf32>) ->
                   (tensor<%Ixf32>, tensor<%Sx%Ixf32>) {
    %1:2 = "onnx.Scan"(%arg0, %arg1) ({
    ^bb0(%body_arg0: tensor<%Ixf32>, %body_arg1: tensor<%Ixf32>):
      %2 = "onnx.Add"(%body_arg0, %body_arg1) :
           (tensor<%Ixf32>, tensor<%Ixf32>) -> tensor<%Ixf32>
      %3 = "onnx.Identity"(%2) : (tensor<%Ixf32>) -> tensor<%Ixf32>
      "onnx.Yield"(%2, %3) : (tensor<%Ixf32>, tensor<%Ixf32>) -> ()
    }) {num_scan_inputs = 1 : si64} :
        (tensor<%Ixf32>, tensor<%Sx%Ixf32>)
        -> (tensor<%Ixf32>, tensor<%Sx%Ixf32>)
    "func.return"(%1#0, %1#1) : (tensor<%Ixf32>, tensor<%Sx%Ixf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
})";

ScanLibBuilder::ScanLibBuilder(const std::string &modelName,
    const int /*seq=*/S, const int /*inner-dim=*/I, const int /*batch=*/B,
    const bool is_v8)
    : ModelLibBuilder(modelName), S(S), I(I), B(B), is_v8(is_v8) {}

bool ScanLibBuilder::build() {
  if (is_v8) {
    initialShape = {B, I};
    xShape = {B, S, I};
    moduleIR = testScanIdentityAddV8;
  } else {
    initialShape = {I};
    xShape = {S, I};
    moduleIR = testScanIdentityAdd;
  }

  moduleIR = std::regex_replace(moduleIR, std::regex("%B"), std::to_string(B));
  moduleIR = std::regex_replace(moduleIR, std::regex("%S"), std::to_string(S));
  moduleIR = std::regex_replace(moduleIR, std::regex("%I"), std::to_string(I));
  OwningOpRef<ModuleOp> moduleRef =
      mlir::parseSourceString<ModuleOp>(moduleIR, &ctx);
  module = moduleRef.get(); // XXXXXX
  moduleRef.release();
  return true;
}

bool ScanLibBuilder::prepareInputs() {
  return ScanLibBuilder::prepareInputs(
      -omDefaultRangeBound, omDefaultRangeBound);
}

bool ScanLibBuilder::prepareInputs(float dataRangeLB, float dataRangeUB) {
  constexpr int num = 2;
  OMTensor *list[num];
  list[0] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(initialShape), dataRangeLB, dataRangeUB);
  list[1] = omTensorCreateWithRandomData<float>(
      llvm::ArrayRef(xShape), dataRangeLB, dataRangeUB);
  inputs = omTensorListCreate(list, num);
  return inputs && list[0] && list[1];
}

bool ScanLibBuilder::prepareInputsFromEnv(const std::string envDataRange) {
  std::vector<float> range = ModelLibBuilder::getDataRangeFromEnv(envDataRange);
  return range.size() == 2 ? prepareInputs(range[0], range[1])
                           : prepareInputs();
}

bool ScanLibBuilder::verifyOutputs() {
  // Get inputs and outputs.
  if (!inputs || !outputs)
    return false;
  OMTensor *init = omTensorListGetOmtByIndex(inputs, 0);
  OMTensor *x = omTensorListGetOmtByIndex(inputs, 1);
  OMTensor *resy = omTensorListGetOmtByIndex(outputs, 0);
  OMTensor *resz = omTensorListGetOmtByIndex(outputs, 1);
  OMTensor *refy = is_v8 ? omTensorCreateWithShape<float>({B, I})
                         : omTensorCreateWithShape<float>({I});
  OMTensor *refz = is_v8 ? omTensorCreateWithShape<float>({B, S, I})
                         : omTensorCreateWithShape<float>({S, I});
  if (!init || !x || !resy || !refy || !resz || !refz)
    return false;
#ifdef PRINT_TENSORS
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < I; ++i) {
      if (is_v8)
        printf("init<b=%ld, i=%ld>: %f\n", b, i,
            omTensorGetElem<float>(init, {b, i}));
      else
        printf("init<i=%ld>: %f\n", i, omTensorGetElem<float>(init, {i}));
    }
  }
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t i = 0; i < I; ++i) {
        if (is_v8)
          printf("x<b=%ld, s=%ld, i=%ld>: %f\n", b, s, i,
              omTensorGetElem<float>(x, {b, s, i}));
        else
          printf(
              "x<s=%ld, i=%ld>: %f\n", s, i, omTensorGetElem<float>(x, {s, i}));
      }
    }
  }
#endif
  // Compute reference. Scan with onnx.Add
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < I; ++i) {
      float refVal = 0.0;
      refVal = is_v8 ? omTensorGetElem<float>(init, {b, i})
                     : omTensorGetElem<float>(init, {i});
      for (int64_t s = 0; s < S; s++) {
        if (is_v8) {
          refVal += omTensorGetElem<float>(x, {b, s, i});
          omTensorGetElem<float>(refz, {b, s, i}) = refVal;
        } else {
          refVal += omTensorGetElem<float>(x, {s, i});
          omTensorGetElem<float>(refz, {s, i}) = refVal;
        }
      }
      if (is_v8)
        omTensorGetElem<float>(refy, {b, i}) = refVal;
      else
        omTensorGetElem<float>(refy, {i}) = refVal;
    }
  }
#ifdef PRINT_TENSORS
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < I; ++i) {
      if (is_v8)
        printf("resy/refy<b=%ld, i=%ld>: %f %f\n", b, i,
            omTensorGetElem<float>(resy, {b, i}),
            omTensorGetElem<float>(refy, {b, i}));
      else
        printf("resy/refy<i=%ld>: %f %f\n", i,
            omTensorGetElem<float>(resy, {i}),
            omTensorGetElem<float>(refy, {i}));
    }
  }
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t i = 0; i < I; ++i) {
        if (is_v8)
          printf("resz/refz<b=%ld, s=%ld, i=%ld>: %f %f\n", b, s, i,
              omTensorGetElem<float>(resz, {b, s, i}),
              omTensorGetElem<float>(refz, {b, s, i}));
        else
          printf("resz/refz<s=%ld, i=%ld>: %f %f\n", s, i,
              omTensorGetElem<float>(resz, {s, i}),
              omTensorGetElem<float>(refz, {s, i}));
      }
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
