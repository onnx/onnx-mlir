/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- MatMulModel.cpp - Building MatMul Models for tests -=======//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a MatMul model and compiles it.
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
// Generate and compile a MatMul 2D.
//===----------------------------------------------------------------------===//

bool genMatMul2DModelAndCompile(
    /* compile option */
    const string &modelName,
    /* conv param in*/
    const int I, const int J, const int K) {

  MLIRContext ctx;
  registerDialects(ctx);

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> aShape = {I, K};
  llvm::SmallVector<int64_t, 1> bShape = {K, J};
  llvm::SmallVector<int64_t, 4> cShape = {I, J};
  auto aType = RankedTensorType::get(aShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto yType = RankedTensorType::get(cShape, builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{aType, bType};
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

  auto MatmulOp = builder.create<ONNXMatMulOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*A=*/aVal, /*B=*/bVal);

  llvm::SmallVector<Value, 1> results = {MatmulOp.getResult()};
  builder.create<ReturnOp>(UnknownLoc::get(&ctx), results);
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/2,
      /*numOutputs=*/1,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);
  compileModule(moduleRef, ctx, modelName, onnx_mlir::EmitLib);
  return true;
}
