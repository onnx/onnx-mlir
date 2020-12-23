#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/MainUtils.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#define SHARED_LIB_BASE string("./TestLoop_main_graph")

using namespace std;

std::string testLoopSimpleIR = R"(
module {
  func @loop_body(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<i1>, tensor<1xi64>, tensor<1xi64>) {
    %0 = "onnx.Identity"(%arg1) : (tensor<i1>) -> tensor<i1>
    %1 = "onnx.Add"(%arg0, %arg2) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    return %0, %1, %1 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
  }
  func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) {body = @loop_body} : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32} : () -> ()
})";

std::string testLoopWithEarlyTermination = R"(
module {
  func @loop_body(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<i1>, tensor<1xi64>, tensor<1xi64>) attributes {input_names = ["iter_count", "cond_in", "y_in"], output_names = ["cond_out", "y_out", "scan_out"]} {
    %0 = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
    %1 = "onnx.Less"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = "onnx.Add"(%arg2, %arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    return %1, %2, %2 : tensor<i1>, tensor<1xi64>, tensor<1xi64>
  }
  func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) attributes {input_names = ["trip_count", "cond", "y"], output_names = ["res_y", "res_scan"]} {
    %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) {body = @loop_body} : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
    return %0#0, %0#1 : tensor<1xi64>, tensor<?x1xi64>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32} : () -> ()
})";

// Returns whether onnx-mlir compiled loop operation is producing the same
// results as a naive implementation of loop operation for a specific set of
// convolution parameters/configuration.
bool isOMLoopTheSameAsNaiveImplFor(std::string moduleIR,
    const int64_t tripCount, const int64_t yInit,
    const int64_t earlyTerminationTripCount =
        std::numeric_limits<int64_t>::max()) {
  MLIRContext ctx;
  registerDialects(ctx);

  auto module = mlir::parseSourceString(moduleIR, &ctx);
  OwningModuleRef moduleRef(std::move(module));
  compileModule(moduleRef, ctx, SHARED_LIB_BASE, EmitLib);
  onnx_mlir::ExecutionSession sess(SHARED_LIB_BASE + ".so", "run_main_graph");

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto tripCountTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(tripCountTensor.get(), {}) = tripCount;
  inputs.emplace_back(move(tripCountTensor));

  auto condTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(nullptr, 0, OM_DATA_TYPE::ONNX_TYPE_BOOL),
      omTensorDestroy);
  omTensorGetElem<bool>(condTensor.get(), {}) = true;
  inputs.emplace_back(move(condTensor));

  auto *yInitShape = new int64_t[1]{1};
  auto yInitTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(&yInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(yInitTensor.get(), {0}) = yInit;
  inputs.emplace_back(move(yInitTensor));

  auto outputs = sess.run(move(inputs));

  auto *yRefInitShape = new int64_t[1]{1};
  auto vFinalRef = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(&yRefInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);

  omTensorGetElem<int64_t>(vFinalRef.get(), {0}) = yInit;
  for (int64_t i = 0;
       i <= std::min<int64_t>(earlyTerminationTripCount, tripCount - 1); i++)
    omTensorGetElem<int64_t>(vFinalRef.get(), {0}) += i;

  auto &vFinal = outputs.at(0);
  return omTensorAreTwoOmtsClose<int64_t>(vFinal.get(), vFinalRef.get());
}

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

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

  return 0;
}
