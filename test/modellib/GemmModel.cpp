/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- GemmModel.cpp - Building GEMM Models for tests -===========//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a GEMM model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Generate and compile a GEMM.
//===----------------------------------------------------------------------===//

bool genGemmAndCompileModel(
    /* compile option */
    const string &modelName,
    /* conv param in*/
    const int I, const int J, const int K, const int aTrans, const int bTrans,
    const int cRank, const float alphaVal, const float betaVal,
    /* GEMM param out*/
    SmallVector<int64_t, 2> &aShape, SmallVector<int64_t, 2> &bShape,
    SmallVector<int64_t, 2> &cShape) {
  MLIRContext ctx;
  registerDialects(ctx);

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);

  aShape = {I, K};
  if (aTrans)
    aShape = {K, I};
  bShape = {K, J};
  if (bTrans)
    bShape = {J, K};
  cShape = {J};
  if (cRank == 2)
    cShape = {I, J};
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
  compileModule(moduleRef, ctx, modelName, onnx_mlir::EmitLib);
  return true;
}
