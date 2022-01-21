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

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"

#define SHARED_LIB_BASE string("./TestGemm_main_graph")

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

// Include some helper functions.
#include "Helper.hpp"

void *omTensorGetAllocatedPtr(OMTensor *tensor);
template <typename TYPE>
void omPrintAsPython(OMTensor *tensor, string name) {
  int rank = omTensorGetRank(tensor);
  int64_t *shape = omTensorGetShape(tensor);
  if (false) {
    printf("# tensor 0x%llx, allocated addr 0x%llx, data addr 0x%llx\n",
        (long long)tensor, (long long)omTensorGetAllocatedPtr(tensor),
        (long long)omTensorGetDataPtr(tensor));
  }
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
    const int aTrans, const int bTrans, const int cRank, const float alphaVal,
    const float betaVal) {
  MLIRContext ctx;
  setCompileContext(ctx, {{OptionKind::CompilerOptLevel, "3"}});

  static int testNum = 0;
  printf("attempt %d with i %d, j %d, k %d%s%s, cRank %d, alpha %7.3f, beta "
         "%7.3f\n",
      ++testNum, I, J, K, (aTrans ? ", aTrans" : ""),
      (bTrans ? ", bTrans" : ""), cRank, (double)alphaVal, (double)betaVal);

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);

  llvm::SmallVector<int64_t, 2> aShape = {I, K};
  llvm::SmallVector<int64_t, 2> aShapeT = {K, I};
  llvm::SmallVector<int64_t, 2> bShape = {K, J};
  llvm::SmallVector<int64_t, 2> bShapeT = {J, K};
  if (aTrans)
    aShape = aShapeT;
  if (bTrans)
    bShape = bShapeT;
  llvm::SmallVector<int64_t, 2> cShape = {J};
  llvm::SmallVector<int64_t, 2> cShape2 = {I, J};
  if (cRank == 2)
    cShape = cShape2;
  else
    assert(cRank == 1 && "cRank == 1 or 2");

  llvm::SmallVector<int64_t, 2> yShape = {I, J};
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
  auto gemmOp = builder.create<ONNXGemmOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal, /*C=*/cVal, alphaAttr, betaAttr,
      aTransAttr, bTransAttr);
  gemmOp.getResult().setType(yType);

  llvm::SmallVector<Value, 1> results = {gemmOp.getResult()};
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

  compileModule(moduleRef, ctx, SHARED_LIB_BASE, onnx_mlir::EmitLib);
  onnx_mlir::ExecutionSession sess(
      getSharedLibName(SHARED_LIB_BASE), "run_main_graph");

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
      omTensorCreateWithRandomData<float>(llvm::makeArrayRef(cShape)),
      omTensorDestroy);
  inputs.emplace_back(move(cOmt));

  auto ref = omTensorCreateWithShape<float>({I, J});
  auto &a = inputs.at(0);
  auto &b = inputs.at(1);
  auto &c = inputs.at(2);

  if (false) {
    printf("Initializes using defined values, better for debugging\n");
    assert(cRank == 1);
    // init A
    for (int64_t i = 0; i < I; ++i)
      for (int64_t k = 0; k < K; k++)
        omTensorGetElem<float>(a.get(), {i, k}) = 100.0 * i + 1.0 * k;
    // init B
    for (int64_t k = 0; k < K; k++)
      for (int64_t j = 0; j < J; ++j)
        omTensorGetElem<float>(b.get(), {k, j}) = 10 * j + 1.0 * k;
    // init C
    for (int64_t j = 0; j < J; ++j) {
      omTensorGetElem<float>(c.get(), {j}) = 0.0;
    }
  }

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
      float cVal;
      if (cRank == 1)
        cVal = omTensorGetElem<float>(c.get(), {j});
      else if (cRank == 2)
        cVal = omTensorGetElem<float>(c.get(), {i, j});
      else
        assert(false);
      omTensorGetElem<float>(ref, {i, j}) =
          alphaVal * omTensorGetElem<float>(ref, {i, j}) + betaVal * cVal;
    }
  }

  auto outputs = sess.run(move(inputs));

  auto &gemm = outputs.at(0);
  float rtol = getenv("TEST_RTOL") ? atof(getenv("TEST_RTOL")) : 1e-5;
  float atol = getenv("TEST_ATOL") ? atof(getenv("TEST_ATOL")) : 1e-5;

  bool success = omTensorAreTwoOmtsClose<float>(gemm.get(), ref, rtol, atol);
  return success;
}

int main(int argc, char *argv[]) {
  llvm::FileRemover remover(getSharedLibName(SHARED_LIB_BASE));

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TestGemm\n", nullptr, "TEST_ARGS");

  if (true) {
    printf("RapidCheck test case generation.\n");
    bool success = rc::check("Gemm implementation correctness", []() {
      const int maxRange = 50;
      const auto I = *rc::gen::inRange(1, maxRange);
      const auto J = *rc::gen::inRange(1, maxRange);
      const auto K = *rc::gen::inRange(1, maxRange);
      const auto aTrans = *rc::gen::inRange(0, 2);
      const auto bTrans = *rc::gen::inRange(0, 2);
      const auto cRank = *rc::gen::inRange(1, 3);
      const auto hasAlpha = *rc::gen::inRange(0, 2);
      const auto hasBeta = *rc::gen::inRange(0, 2);
      float alpha = hasAlpha ? 1.2 : 1.0;
      float beta = hasBeta ? 0.8 : 1.0;
      RC_ASSERT(isOMGemmTheSameAsNaiveImplFor(
          I, J, K, aTrans, bTrans, cRank, alpha, beta));
    });
    if (!success)
      return 1;
  }

  if (false) {
    // Was too slow on some machines, disable test.
    printf("\n\nIndividual test case generation (benchmarks).\n");
    assert(isOMGemmTheSameAsNaiveImplFor(3, 5, 4, 0, 0, 2, 0.25, 0.35));

    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 1024, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 2048, 0, 1, 2, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 25088, 0, 1, 1, 1.0, 1.0));
    // vcg 19
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 25088, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 4096, 4096, 0, 1, 1, 1.0, 1.0));
    assert(isOMGemmTheSameAsNaiveImplFor(1, 1000, 4096, 0, 1, 1, 1.0, 1.0));
  }
  return 0;
}
