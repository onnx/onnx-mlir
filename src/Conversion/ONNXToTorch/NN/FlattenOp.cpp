/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- FlattenOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <fstream>
#include <iostream>
#include <set>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"


#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


/**
 * 
 * Attributes 
 * axis    i64-bit signed integer attribute
 * 
 * /scripts/docker/build_with_docker.py --external-build --build-dir build --command "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir --EmitONNXIR --debug --run-torch-pass third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/test_leakyrelu/model.onnx"
 * 
 */


class ONNXFlattenOpToTorchLowering : public ConversionPattern {
public:
  ONNXFlattenOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ::mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXFlattenOp op1 = llvm::dyn_cast<ONNXFlattenOp>(op);
    ONNXFlattenOpAdaptor adaptor(operands);
    
    Value input = adaptor.input();
    auto inputTy = input.getType().cast<MemRefType>();
    auto inputShape = inputTy.getShape();
    int inputRank = inputShape.size();
    auto axisValue = op1.axis();       // ::mlir::IntegerAttr
    if (axisValue < 0)
      axisValue = inputRank + axisValue;
   
    llvm::outs() << "input from Flatten Op:   " << "\n" << input << "\n" << "\n"; 
    llvm::outs() << "axisValue from Flatten Op:   " << "\n" << axisValue << "\n" << "\n";

    TensorType op_tensor_type = op1.getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(),
                                                        op_tensor_type.getElementType());

    int64_t startDim = -1;
    int64_t endDim = -1;
    if (startDim < 0)
      startDim += inputRank;
    if (endDim < 0)
      endDim += inputRank;
    auto ty = IntegerType::get(op1.getContext(), 64);
    auto f0 = IntegerAttr::get(ty, (startDim));
    Value p0v = rewriter.create<ConstantIntOp>(loc, f0);

    auto f1 = IntegerAttr::get(ty, (endDim));
    Value p1v = rewriter.create<ConstantIntOp>(loc, f1);

    auto axisVal = IntegerAttr::get(ty, (axisValue));
    Value p2v = rewriter.create<ConstantIntOp>(loc, axisVal);

    Value atenleakyrelu = rewriter.create<AtenFlattenUsingIntsOp>(loc, resultTy, p2v, p0v, p1v);

    Value result = atenleakyrelu;

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op->getResult(0).getType() , result);
    return success();
#if 0
    TensorType x_tensor_type  = x.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto xTy      = Torch::ValueTensorType::get(context, x_tensor_type.getShape(), 
		    x_tensor_type.getElementType());
    auto xtt      = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, xTy, x); 
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(), 
		    op_tensor_type.getElementType());

    Value atenleakyrelu = rewriter.create<AtenLeakyReluOp>(loc, resultTy, xtt, f3v); 

    llvm::outs() << "ATENRELU CREATED is " << atenleakyrelu << "\n"; 
    Value result = atenleakyrelu; 

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op->getResult(0).getType() , result);
    return success();
#endif
  }
};

void populateLoweringONNXToTorchFlattenOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXFlattenOpToTorchLowering>(typeConverter, ctx);
}
