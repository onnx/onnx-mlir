/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------------- ModelBuilder.hpp -----------------------------===//
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

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace mlir;
namespace BackendCppTests {

// Helper class containing useful functions for creating, compiling and running
// a test.
class ModelBuilder {
  MLIRContext &ctx;
  const string sharedLibBaseName;
  Location loc;
  ModuleOp module;
  OpBuilder builder;

public:
  ModelBuilder(MLIRContext &ctx, const string sharedLibBaseName)
      : ctx(ctx), sharedLibBaseName(sharedLibBaseName),
        loc(UnknownLoc::get(&ctx)), module(createEmptyModule()),
        builder(OpBuilder(&ctx)) {}

  MLIRContext &getContext() { return ctx; }
  ModuleOp &getModule() { return module; }
  OpBuilder &getBuilder() { return builder; }
  Location &getLocation() { return loc; }

  // Create a function with an empty body.
  // This function will contain the model to be tested.
  FuncOp createEmptyTestFunction(const llvm::SmallVectorImpl<Type> &inputsTypes,
      const llvm::SmallVectorImpl<Type> &outputsTypes);

  // Create the entry point function (used to call the model test function).
  void createEntryPoint(FuncOp &funcOp);

  // Compile the model. Compiler options are passed in \p compileOptions.
  bool compileTest(const onnx_mlir::CompilerOptionList &compileOptions);

  // Run the model and verify the result(s). The \p verifyFunction parameter
  // is used to pass in the function object used to verify the correctness of
  // the test result.
  bool runAndVerifyTest(std::vector<onnx_mlir::OMTensorUniquePtr> &inputs,
      std::vector<onnx_mlir::OMTensorUniquePtr> &expectedOutputs,
      std::function<bool(OMTensor *, OMTensor *)> verifyFunction);

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

} // namespace BackendCppTests

#endif // ONNX_MLIR_TEST_HELPER_H
