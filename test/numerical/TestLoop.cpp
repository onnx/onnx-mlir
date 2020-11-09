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
bool isOMLoopTheSameAsNaiveImplFor(const int tripCount) {
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

  int64_t tripCountLiteral = 10;
  char cond = 1;
  int64_t yInit = 0;
  int64_t yInitShape[1] = {1};

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto tripCountTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreate(&tripCountLiteral, NULL, 0, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  inputs.emplace_back(move(tripCountTensor));

  auto condTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreate(&cond, NULL, 0, OM_DATA_TYPE::ONNX_TYPE_BOOL),
      omTensorDestroy);
  inputs.emplace_back(move(condTensor));

  auto yInitTensor = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreate(&yInit, &yInitShape[0], 1, OM_DATA_TYPE::ONNX_TYPE_INT64),
      omTensorDestroy);
  inputs.emplace_back(move(yInitTensor));

  auto outputs = sess.run(move(inputs));
  auto &conv = outputs.at(0);
  auto &scan = outputs.at(1);
  auto *data = (int64_t *)omTensorGetDataPtr(conv.get());
  auto *scanData = (int64_t *)omTensorGetDataPtr(scan.get());

  printf("out=%lld\n", data[0]);
  for (int i = 0; i < 10; i++)
    printf("scan_out=%lld\n", scanData[i]);

  //
  //    return omTensorAreTwoOmtsClose<float>(conv.get(), ref);
  return true;
}

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  isOMLoopTheSameAsNaiveImplFor(10);

  return 0;
}
