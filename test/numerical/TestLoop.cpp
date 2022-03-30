/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

static const llvm::StringRef SHARED_LIB_BASE("./TestLoop_main_graph");

using namespace mlir;

namespace onnx_mlir {
namespace test {

std::string testLoopSimpleIR = R"(
module {
  func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
      %body_0 = "onnx.Identity"(%body_arg1) : (tensor<i1>) -> tensor<i1>
      %body_1 = "onnx.Add"(%body_arg0, %body_arg2) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
      onnx.Return %body_0, %body_1, %body_1 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
})";

std::string testLoopWithEarlyTermination = R"(
module {
  func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) attributes {input_names = ["trip_count", "cond", "y"], output_names = ["res_y", "res_scan"]} {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
      %0 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
      %1 = "onnx.Less"(%body_arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %2 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    onnx.Return %1, %2, %2 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
})";

std::string testLoopWithParentScopeVariable = R"(
module {
  func @main_graph(%max_trip_count: tensor<i64>, %cond: tensor<i1>, %y_initial: tensor<1xi64>) ->
        (tensor<1xi64>, tensor<?x1xi64>) {
    %const_offset = "onnx.Constant"() {value = dense<7> : tensor<i64>} : () -> tensor<i64>
    %y_final, %y_scan = "onnx.Loop"(%max_trip_count, %cond, %y_initial) ({
    ^bb0(%i: tensor<i64>, %body_cond: tensor<i1>, %y_prev: tensor<1xi64>):
      %2 = "onnx.Add"(%y_prev, %i) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
      %3 = "onnx.Add"(%2, %const_offset) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
      onnx.Return %body_cond, %3, %3 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %y_final, %y_scan : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32, signature = "[    ]"} : () -> ()
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
  registerDialects(ctx);

  auto module = mlir::parseSourceString(moduleIR, &ctx);
  OwningOpRef<ModuleOp> moduleRef(std::move(module));
  compileModule(moduleRef, ctx, SHARED_LIB_BASE.str(), onnx_mlir::EmitLib);
  onnx_mlir::ExecutionSession sess(
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  std::vector<OMTensorUniquePtr> inputs;
  auto tripCountTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(tripCountTensor.get(), {}) = tripCount;
  inputs.emplace_back(move(tripCountTensor));

  auto condTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_BOOL),
      omTensorDestroy);
  omTensorGetElem<bool>(condTensor.get(), {}) = true;
  inputs.emplace_back(move(condTensor));

  auto *yInitShape = new int64_t[1]{1};
  auto yInitTensor = OMTensorUniquePtr(
      omTensorCreateEmpty(&yInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(yInitTensor.get(), {0}) = yInit;
  inputs.emplace_back(move(yInitTensor));

  auto outputs = sess.run(move(inputs));

  auto *yRefInitShape = new int64_t[1]{1};
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
      ModelLibBuilder::getSharedLibName(SHARED_LIB_BASE.str()));

  setCompilerOption(OptionKind::CompilerOptLevel, "3");
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestLoop\n", nullptr, "TEST_ARGS");

  // Loop tests, simple.
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 0, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 1, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(testLoopSimpleIR, 10, 42));

  // Loop tests, with early termination. The early termination trip count is
  // hard-coded in the IR as a constant operation as 3.
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 0, 42, /*earlyTerminationTripCount=*/3));
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 1, 42, /*earlyTerminationTripCount=*/3));
  assert(isOMLoopTheSameAsNaiveImplFor(
      testLoopWithEarlyTermination, 10, 42, /*earlyTerminationTripCount=*/3));

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
