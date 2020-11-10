#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/MainUtils.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#define SHARED_LIB_BASE string("./TestLoop_main_graph")

using namespace std;

// Returns whether onnx-mlir compiled convolution is producing the same results
// as a naive implementation of convolution for a specific set of convolution
// parameters/configuration.
bool isOMLoopTheSameAsNaiveImplFor(
    const int64_t tripCount, const int64_t yInit) {
  MLIRContext ctx;
  registerDialects(ctx);

  std::string moduleIR = R"(
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

  auto module = mlir::parseSourceString(moduleIR, &ctx);
  OwningModuleRef moduleRef(std::move(module));
  compileModule(moduleRef, ctx, SHARED_LIB_BASE, EmitLib);
  onnx_mlir::ExecutionSession sess(SHARED_LIB_BASE + ".so", "run_main_graph");

  int64_t *yInitShape = new int64_t[1]{1};
  int64_t *yRefInitShape = new int64_t[1]{1};
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

  auto yInitTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(&yInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  omTensorGetElem<int64_t>(yInitTensor.get(), {0}) = yInit;
  inputs.emplace_back(move(yInitTensor));

  auto outputs = sess.run(move(inputs));

  auto vFinalRef = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateEmpty(&yRefInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);

  omTensorGetElem<int64_t>(vFinalRef.get(), {0}) = yInit;
  for (int i = 0; i < tripCount; i++)
    omTensorGetElem<int64_t>(vFinalRef.get(), {0}) += i;

  auto &vFinal = outputs.at(0);
  return omTensorAreTwoOmtsClose<int64_t>(vFinal.get(), vFinalRef.get());
}

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  assert(isOMLoopTheSameAsNaiveImplFor(0, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(1, 42));
  assert(isOMLoopTheSameAsNaiveImplFor(10, 42));

  return 0;
}
