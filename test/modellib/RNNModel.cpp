/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- RNNModel.cpp - Building RNN Models for tests -=============//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a RNN model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "include/OnnxMlirRuntime.h"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelHelper.hpp"
#include "test/modellib/ModelLib.hpp"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Generate and compile a RNN.
//===----------------------------------------------------------------------===//

bool genRNNModelAndCompile(
    /* compile option */
    const string &modelName,
    /* RNN param in*/
    const int direction, const int S, const int B, const int I, const int H,
    const bool isDynamicS, const bool isDynamicB,
    /* RNN param out*/
    int &D, SmallVector<int64_t, 3> &xShape, SmallVector<int64_t, 3> &hShape,
    OMTensor *&wOmt, OMTensor *&rOmt, OMTensor *&bOmt) {

  MLIRContext ctx;
  registerDialects(ctx);

  D = abs(direction);
  int S1 = S, B1 = B;
  if (isDynamicS)
    S1 = -1;
  if (isDynamicB)
    B1 = -1;

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  xShape = {S, B, I};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {S1, B1, I};
  llvm::SmallVector<int64_t, 3> wShape = {D, H, I};
  llvm::SmallVector<int64_t, 3> rShape = {D, H, H};
  llvm::SmallVector<int64_t, 2> bShape = {D, 2 * H};
  hShape = {D, B, H};
  llvm::SmallVector<int64_t, 3> hShapeSymbol = {D, B1, H};

  auto xType = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto rType = RankedTensorType::get(rShape, builder.getF32Type());
  auto bType = RankedTensorType::get(bShape, builder.getF32Type());
  auto hType = RankedTensorType::get(hShapeSymbol, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());
  auto yHType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 5> inputsType{xType, hType};
  llvm::SmallVector<Type, 2> outputsType{yType, yHType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto noneVal = builder.create<ONNXNoneOp>(UnknownLoc::get(&ctx)).getResult();
  auto xVal = entryBlock->getArgument(0);
  auto sVal = noneVal;
  auto hVal = entryBlock->getArgument(1);

  StringAttr directionAttr;
  if (direction == 1)
    directionAttr = builder.getStringAttr("forward");
  else if (direction == 2)
    directionAttr = builder.getStringAttr("bidirectional");
  else
    directionAttr = builder.getStringAttr("reverse");
  auto hiddenSizeAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, H, /*isSigned=*/true));
  auto activationsAttr = builder.getStrArrayAttr({"Tanh", "Tanh"});

  wOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(wShape), 0, 1);
  rOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(rShape), 0, 1);
  bOmt = omTensorCreateWithRandomData<float>(llvm::makeArrayRef(bShape), 0, 1);
  auto wConstant = buildONNXConstantOp(&ctx, builder, wOmt, wType);
  auto rConstant = buildONNXConstantOp(&ctx, builder, rOmt, rType);
  auto bConstant = buildONNXConstantOp(&ctx, builder, bOmt, bType);

  auto rnnOp = builder.create<ONNXRNNOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType, /*Y_h=*/yHType,
      /*X=*/xVal, /*W=*/wConstant, /*R=*/rConstant, /*B=*/bConstant,
      /*sequence_lens=*/sVal, /*initial_h=*/hVal,
      /*activation_alpha=*/ArrayAttr(), /*activation_beta=*/ArrayAttr(),
      /*activations=*/activationsAttr, /*clip=*/FloatAttr(),
      /*direction=*/directionAttr, /*hidden_size=*/hiddenSizeAttr);

  rnnOp.getResults()[0].setType(yType);
  rnnOp.getResults()[1].setType(yHType);

  builder.create<ReturnOp>(UnknownLoc::get(&ctx), rnnOp.getResults());
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/2,
      /*numOutputs=*/2,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);
  compileModule(moduleRef, ctx, modelName, onnx_mlir::EmitLib);
  return true;
}
