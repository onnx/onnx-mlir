/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------------- TestHelper.hpp ------------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines a set of helper functions for creating, compiling and
// running an end-to-end test.
//
//====---------------------------------------------------------------------===//

#ifndef ONNX_MLIR_TEST_HELPER_H
#define ONNX_MLIR_TEST_HELPER_H

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#include <string>

using namespace std;
using namespace mlir;

namespace Tests {

// Helper class containing useful function for creating, compiling and running a
// test.
class TestHelper {
  MLIRContext &ctx;
  Location loc;
  ModuleOp module;
  OpBuilder builder;

public:
  TestHelper(MLIRContext &ctx)
      : ctx(ctx), loc(UnknownLoc::get(&ctx)), module(createEmptyModule()),
        builder(OpBuilder(&ctx)) {}

  MLIRContext &getContext() { return ctx; }
  ModuleOp &getModule() { return module; }
  OpBuilder &getBuilder() { return builder; }
  Location &getLocation() { return loc; }

  // Create a function with a single input/output and an empty body.
  // This function will contain the operator to be tested.
  FuncOp createEmptyTestFunction(Type inputType, Type outputType);

  // Create the entry point function (used to call the test function).
  void createEntryPoint(FuncOp &funcOp, int numInputs = 1, int numOutputs = 1);

  // Compile the module.
  bool compileTest(const string &sharedLibBase);

  // Run the test and verify the correctness of the result.
  bool runAndVerifyTest(
      std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> &inputs,
      std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>>
          &expectedOutputs,
      const string &sharedLibBase);

  // Prepare for a new test.
  void reset();

  static string getSharedLibName(string sharedLibBaseName) {
#ifdef _WIN32
    return sharedLibBaseName + ".dll";
#else
    return sharedLibBaseName + ".so";
#endif
  }

private:
  ModuleOp createEmptyModule() const;
};

} // namespace Tests

#endif // ONNX_MLIR_TEST_HELPER_H