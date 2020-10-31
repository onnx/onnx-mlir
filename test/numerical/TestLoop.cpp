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

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  auto loc = UnknownLoc::get(&ctx);

  llvm::SmallVector<Type, 3> inputsType{
      RankedTensorType::get({}, builder.getI64Type()),
      RankedTensorType::get({}, builder.getI1Type()),
      RankedTensorType::get({1}, builder.getI64Type())};
  llvm::SmallVector<Type, 1> outputsType{
      RankedTensorType::get({1}, builder.getI64Type())};
  llvm::SmallVector<Type, 2> bodyOutputTypes{
      RankedTensorType::get({}, builder.getI1Type()),
      RankedTensorType::get({1}, builder.getI64Type())};
  auto bodyFuncTy = builder.getFunctionType(inputsType, bodyOutputTypes);
  auto bodyFuncOp = builder.create<FuncOp>(loc, "loop_body", bodyFuncTy);

  {
    OpBuilder::InsertionGuard guard(builder);
    auto bodyEntryBlock = bodyFuncOp.addEntryBlock();
    Value iv = bodyEntryBlock->getArgument(0);
    Value cond = bodyEntryBlock->getArgument(1);
    Value yInit = bodyEntryBlock->getArgument(2);

    builder.setInsertionPointToStart(bodyEntryBlock);
    cond =
        builder.create<ONNXIdentityOp>(loc, cond.getType(), cond).getResult();
    auto y = builder.create<ONNXAddOp>(loc, iv, yInit);
    builder.create<ReturnOp>(loc, ValueRange({cond, y}));
  }
  module.push_back(bodyFuncOp);

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  auto funcOp = builder.create<FuncOp>(loc, funcName, funcType);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  {
    OpBuilder::InsertionGuard guard(builder);
    auto maxTripCount = entryBlock->getArgument(0);
    auto cond = entryBlock->getArgument(1);
    auto yInit = entryBlock->getArgument(2);

    auto vFinalsTy =
        SmallVector<Type, 4>{RankedTensorType::get({1}, builder.getI64Type())};
    auto loopOutput =
        builder
            .create<ONNXLoopOp>(loc, vFinalsTy, maxTripCount, cond,
                ValueRange{yInit}, builder.getSymbolRefAttr(bodyFuncOp))
            .getResults();
    builder.create<ReturnOp>(loc, loopOutput);
  }
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  auto entryPoint = ONNXEntryPointOp::create(loc, funcOp,
      /*numInputs=*/3,
      /*numOutputs=*/1);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);
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
  int64_t *data = (int64_t *)omTensorGetDataPtr(conv.get());
  printf("out=%d\n", data[0]);

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
