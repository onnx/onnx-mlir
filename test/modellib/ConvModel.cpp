/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ConvModel.cpp - Building Conv Models for tests -===========//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that builds a convolution model and compiles
// it.
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

const string getAutoPadName(const int autoPad) {
  static const string autoPadName[] = {
      "NOTSET", "VALID", "SAME_LOWER", "SAME_UPPER"};
  assert(autoPad >= 0 && autoPad < AUTO_PAD_UB && "out of bound autopad");
  return autoPadName[autoPad];
}

//===----------------------------------------------------------------------===//
// Generate and compile a convolution.
//===----------------------------------------------------------------------===//

bool genConv2DModelAndCompile(
    /* compile option */
    const string &modelName,
    /* conv param in*/
    const int N, const int C, const int H, const int W, const int kH,
    const int kW, const int autoPad, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd, const int stride, const int dilation,
    const int isDynamic,
    /* conv param out */
    int &NOut, int &COut, int &HOut, int &WOut) {

  if (autoPad != AUTO_PAD_NOTSET) {
    // Make sure all pads are initially zero, only value tolarated.
    assert(pHBegin == 0 && pHEnd == 0 && pWBegin == 0 && pWEnd == 0);
  }

  MLIRContext ctx;
  registerDialects(ctx);

  // We use the Ns for the shape of the input, and the N1s for the construction
  // of the model. That way, when the shape is dynamic, we set the N1s to "-1"
  // (dynamic value) so that the compiler may not infer the size of the model,
  // and instead generate code to figure the sizes at run time.
  int N1 = N;
  int C1 = C;
  int H1 = H;
  int W1 = W;
  if (isDynamic)
    N1 = C1 = H1 = W1 = -1;

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> xShape = {N, C, H, W};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {N1, C1, H1, W1};
  llvm::SmallVector<int64_t, 1> bShape = {C};
  llvm::SmallVector<int64_t, 4> wShape = {C, C, kH, kW};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto xTypeSymbol = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{xTypeSymbol, wType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto xVal = entryBlock->getArgument(0);
  auto wVal = entryBlock->getArgument(1);
  auto bVal = builder.create<ONNXNoneOp>(UnknownLoc::get(&ctx)).getResult();

  auto dilations = builder.getI64ArrayAttr({dilation, dilation});
  auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
  auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
  auto strides = builder.getI64ArrayAttr({stride, stride});

  auto convOp = builder.create<ONNXConvOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType,
      /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
      /*auto_pad=*/builder.getStringAttr(getAutoPadName(autoPad)),
      /*dilations=*/dilations,
      /*group=*/
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 1, /*isSigned=*/true)),
      /*kernel_shape=*/kernel_shape, /*pads=*/pads,
      /*strides=*/strides);

  // Use the convOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in a inconsistent state.
  convOp.X().setType(xType); // Use static dims to infer shape.
  LogicalResult res = convOp.inferShapes([](mlir::Region &) {});
  if (failed(res)) {
    return false;
  }
  auto outputShape = convOp.getResult().getType().cast<ShapedType>().getShape();
  NOut = outputShape[0];
  COut = outputShape[1];
  HOut = outputShape[2];
  WOut = outputShape[3];
  convOp.getResult().setType(yType);
  convOp.X().setType(xTypeSymbol);

  llvm::SmallVector<Value, 1> results = {convOp.getResult()};
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

  // Compile model.
  OwningModuleRef moduleRef(module);
  compileModule(moduleRef, ctx, modelName, onnx_mlir::EmitLib);
  return true;
}
