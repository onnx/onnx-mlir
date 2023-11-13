/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- TestLoop.cpp - test Loop code -======================================//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the code to test Loop code.
//
//===----------------------------------------------------------------------===//

// Common.hpp needs to be included first to correctly suppress the rapidcheck.h
// warnings.
#include "Common.hpp"

#include "src/Runtime/OMTensorHelper.hpp"

#include "mlir/Parser/Parser.h"

static const llvm::StringRef SHARED_LIB_BASE("./TestLoop_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

std::string testLoopSimpleIR = R"(
module {
  func.func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
      %body_0 = "onnx.Identity"(%body_arg1) : (tensor<i1>) -> tensor<i1>
      %body_1 = "onnx.Add"(%body_arg0, %body_arg2) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
      onnx.Yield %body_0, %body_1, %body_1 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
})";

// Scan output for early termination is not supported yet.
// Previous implementation may fail if the maximum iteration count is too large.
// Save this model for future
std::string testLoopWithEarlyTermination_Orig = R"(
module {
  func.func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
      %0 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
      %1 = "onnx.Less"(%body_arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %2 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    onnx.Yield %1, %2, %2 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
})";

std::string testLoopWithEarlyTermination = R"(
module {
  func.func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<1xi64> {
    %0 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
      %0 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
      %1 = "onnx.Less"(%body_arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %2 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    onnx.Yield %1, %2 : tensor<i1>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
})";

std::string testLoopWithParentScopeVariable = R"(
module {
  func.func @main_graph(%max_trip_count: tensor<i64>, %cond: tensor<i1>, %y_initial: tensor<1xi64>) ->
        (tensor<1xi64>, tensor<?x1xi64>) {
    %const_offset = "onnx.Constant"() {value = dense<7> : tensor<i64>} : () -> tensor<i64>
    %y_final, %y_scan = "onnx.Loop"(%max_trip_count, %cond, %y_initial) ({
    ^bb0(%i: tensor<i64>, %body_cond: tensor<i1>, %y_prev: tensor<1xi64>):
      %2 = "onnx.Add"(%y_prev, %i) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
      %3 = "onnx.Add"(%2, %const_offset) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
      onnx.Yield %body_cond, %3, %3 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %y_final, %y_scan : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
})";

// Returns whether onnx-mlir compiled loop operation is producing the same
// results as a naive implementation of loop operation for a specific set of
// convolution parameters/configuration.
bool isOMLoopTheSameAsNaiveImplFor(std::string moduleIR,
    const int64_t tripCount, const int64_t yInit,
    const int64_t earlyTerminationTripCount =
        std::numeric_limits<int64_t>::max(),
    const int64_t constOffset = 0) {
  MLIRContext ctx;
  loadDialects(ctx);

  auto module = mlir::parseSourceString<ModuleOp>(moduleIR, &ctx);
  OwningOpRef<ModuleOp> moduleRef(std::move(module));
  if (compileModule(
          moduleRef, ctx, SHARED_LIB_BASE.str(), onnx_mlir::EmitLib) != 0)
    return false;

  std::vector<OMTensorUniquePtr> inputs;
  auto tripCountTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(tripCountTensor.get(), {}) = tripCount;
  inputs.emplace_back(std::move(tripCountTensor));

  auto condTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_BOOL),
      omTensorDestroy);
  omTensorGetElem<bool>(condTensor.get(), {}) = true;
  inputs.emplace_back(std::move(condTensor));

  int64_t yInitShape[1] = {1};
  auto yInitTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(&yInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(yInitTensor.get(), {0}) = yInit;
  inputs.emplace_back(std::move(yInitTensor));

  std::string modelTag = getCompilerOption(OptionKind::ModelTag);
  onnx_mlir::ExecutionSession sess(
      onnx_mlir::getTargetFilename(SHARED_LIB_BASE.str(), onnx_mlir::EmitLib),
      modelTag);
  std::vector<onnx_mlir::OMTensorUniquePtr> outputs;
  try {
    outputs = sess.run(std::move(inputs));
  } catch (const std::runtime_error &error) {
    std::cerr << "error while running: " << error.what() << std::endl;
    return false;
  }

  int64_t yRefInitShape[1] = {1};
  auto vFinalRef = OMTensorUniquePtr(
      omTensorCreateEmpty(&yRefInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);

  omTensorGetElem<int64_t>(vFinalRef.get(), {0}) = yInit;
  for (int64_t i = 0;
       i <= std::min<int64_t>(earlyTerminationTripCount, tripCount - 1); i++)
    omTensorGetElem<int64_t>(vFinalRef.get(), {0}) += (i + constOffset);

  auto &vFinal = outputs.at(0);
  return omTensorAreTwoOmtsClose<int64_t>(vFinal.get(), vFinalRef.get());
}

} // namespace test
} // namespace onnx_mlir

int main(int argc, char *argv[]) {
  using namespace onnx_mlir;
  using namespace onnx_mlir::test;

  llvm::FileRemover remover(
      onnx_mlir::getTargetFilename(SHARED_LIB_BASE.str(), onnx_mlir::EmitLib));

  ModelLibBuilder::setRandomNumberGeneratorSeed("TEST_SEED");
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptions});
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestLoop\n", nullptr, "TEST_ARGS");
  initCompilerConfig();
  std::cout << "Target options: \""
            << getCompilerOption(OptionKind::TargetAccel) << "\"\n";

  // Loop tests, simple.
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 0, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 1, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 10, 42));

  // Loop tests, with early termination. The early termination trip count is
  // hard-coded in the IR as a constant operation as 3.

  // Early termination for scan output is temporally disabled
#if 0
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 0, 42, /*earlyTerminationTripCount=*/3));
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 1, 42, /*earlyTerminationTripCount=*/3));
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 10, 42, /*earlyTerminationTripCount=*/3));
#endif

  // Loop tests, in which loop body makes reference to values defined in the
  // parent scope.
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopWithParentScopeVariable, 0, 42,
      /*earlyTerminationTripCount=*/std::numeric_limits<int64_t>::max(),
      /*constOffset=*/7));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopWithParentScopeVariable, 1, 42,
      /*earlyTerminationTripCount=*/std::numeric_limits<int64_t>::max(),
      /*constOffset=*/7));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopWithParentScopeVariable, 10, 42,
      /*earlyTerminationTripCount=*/std::numeric_limits<int64_t>::max(),
      /*constOffset=*/7));
  return 0;
}
