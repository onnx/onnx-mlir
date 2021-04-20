/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
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

#define SHARED_LIB_BASE string("./TestGemm_main_graph")

using namespace std;

template <typename TYPE>
void omPrintAsPython(OMTensor *tensor, string name) {
  int rank = omTensorGetRank(tensor);
  int64_t *shape = omTensorGetShape(tensor);

  if (rank == 2) {
    cout << name << " = np.array([";
    for (int64_t i = 0; i < shape[0]; ++i) {
      if (i)
        cout << ", ";
      cout << "[";
      for (int64_t j = 0; j < shape[1]; ++j) {
        if (j)
          cout << ", ";
        cout << omTensorGetElem<TYPE>(tensor, {i, j});
      }
      cout << "]";
    }
    cout << "])\n";
  }
}

// Returns whether onnx-mlir compiled Gemm is producing the same results
// as a naive implementation of Gemm for a specific set of Gemm
// parameters/configuration. Gemm: A[IxK] * B[KxJ] = C[IxJ]
bool isOMGemmTheSameAsNaiveImplFor(const int I, const int J, const int K,
    const int aTrans, const int bTrans, const float alphaVal,
    const float betaVal) {
  MLIRContext ctx;
  registerDialects(ctx);
  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d %s %s\n", ++testNum, I, J, K,
      (aTrans ? ", aTrans" : ""), (bTrans ? ", bTrans" : ""));

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);

  llvm::SmallVector<int64_t, 4> aShape({I, K}), bShape({K, J});
  if (aTrans) {
    aShape = {K, I};
  }
  if (bTrans) {
    bShape = {J, K};
  }
  llvm::SmallVector<int64_t, 4> cShape = {I, J};
  llvm::SmallVector<int64_t, 4> yShape = {I, J};
  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto cType = RankedTensorType::get(cShape, builder.getF32Type());
  auto yType = RankedTensorType::get(yShape, builder.getF32Type());

  llvm::SmallVector<Type, 3> inputsType{aType, bType, cType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto aVal = entryBlock->getArgument(0);
  auto bVal = entryBlock->getArgument(1);
  auto cVal = entryBlock->getArgument(2);

  FloatAttr alphaAttr = FloatAttr::get(builder.getF32Type(), alphaVal);
  FloatAttr betaAttr = FloatAttr::get(builder.getF32Type(), betaVal);
  IntegerAttr aTransAttr =
      IntegerAttr::get(builder.getIntegerType(64, true), aTrans);
  IntegerAttr bTransAttr =
      IntegerAttr::get(builder.getIntegerType(64, true), bTrans);
  auto GemmOp = builder.create<ONNXGemmOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal, /*C=*/cVal, alphaAttr, betaAttr,
      aTransAttr, bTransAttr);

  llvm::SmallVector<Value, 1> results = {GemmOp.getResult()};
  builder.create<ReturnOp>(UnknownLoc::get(&ctx), results);
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/3,
      /*numOutputs=*/1,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);

  compileModule(moduleRef, ctx, SHARED_LIB_BASE, EmitLib);
  onnx_mlir::ExecutionSession sess(SHARED_LIB_BASE + ".so", "run_main_graph");

  std::vector<unique_ptr<OMTensor, decltype(&omTensorDestroy)>> inputs;
  auto aOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(aShape)),
      omTensorDestroy);
  inputs.emplace_back(move(aOmt));
  auto bOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape)),
      omTensorDestroy);
  inputs.emplace_back(move(bOmt));
  auto cOmt = unique_ptr<OMTensor, decltype(&omTensorDestroy)>(
      omTensorCreateWithRandomData<float>({I, J}), omTensorDestroy);
  inputs.emplace_back(move(cOmt));

  auto ref = omTensorCreateWithShape<float>({I, J});
  auto &a = inputs.at(0);
  auto &b = inputs.at(1);
  auto &c = inputs.at(2);

  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(ref, {i, j}) = 0;
      for (int64_t k = 0; k < K; k++) {
        float aVal, bVal;
        if (aTrans == 0)
          aVal = omTensorGetElem<float>(a.get(), {i, k});
        else
          aVal = omTensorGetElem<float>(a.get(), {k, i});
        if (bTrans == 0)
          bVal = omTensorGetElem<float>(b.get(), {k, j});
        else
          bVal = omTensorGetElem<float>(b.get(), {j, k});
        omTensorGetElem<float>(ref, {i, j}) += aVal * bVal;
      }
    }
  }
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(ref, {i, j}) =
          alphaVal * omTensorGetElem<float>(ref, {i, j}) +
          betaVal * omTensorGetElem<float>(c.get(), {i, j});
    }
  }

  auto outputs = sess.run(move(inputs));
  auto &Gemm = outputs.at(0);
  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  return omTensorAreTwoOmtsClose<float>(Gemm.get(), ref, rtol, atol);
}

int main(int argc, char *argv[]) {
  setExecPath(argv[0], (void *)main);
  llvm::FileRemover remover(SHARED_LIB_BASE + ".so");

  printf("RapidCheck test case generation.\n");
  rc::check("Gemm implementation correctness", []() {
    const int maxRange = 50;
    const auto I = *rc::gen::inRange(1, maxRange);
    const auto J = *rc::gen::inRange(1, maxRange);
    const auto K = *rc::gen::inRange(1, maxRange);
    const auto aTrans = *rc::gen::inRange(0, 2);
    const auto bTrans = *rc::gen::inRange(0, 2);
    const auto hasAlpha = *rc::gen::inRange(0, 2);
    const auto hasBeta = *rc::gen::inRange(0, 2);
    float alpha = hasAlpha ? 1.2 : 1.0;
    float beta = hasBeta ? 0.8 : 1.0;
    RC_ASSERT(
        isOMGemmTheSameAsNaiveImplFor(I, J, K, aTrans, bTrans, alpha, beta));
  });

  if (false) {
    // Was too slow on some machines, disable test.
    printf("\n\nIndividual test case generation (benchmarks).\n");
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 1024, 0, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 2048, 0, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 25088, 0, 1, 1.0, 1.0));
    // vcg 19
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 25088, 0, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 4096, 0, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 4096, 0, 1, 1.0, 1.0));
  }
  return 0;
}
