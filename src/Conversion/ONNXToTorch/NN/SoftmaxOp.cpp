/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SoftmaxOp.cpp - ONNX Op Transform ------------------===//
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
 * ONNX Softmax operation 
 *
 * “Softmax takes input data (Tensor) and an argument alpha, and produces one" 
 * "output data (Tensor) where the function `f(x) = alpha * x for x < 0`," 
 * "`f(x) = x for x >= 0`, is applied to the data tensor elementwise."
 *
 * Operands :
 * X            tensor of 16-bit/32-bit/64-bit float values or memref of any type values
 * Output   : 
 * Y            tensor of 16-bit/32-bit/64-bit float values or memref of any type values 
 *
 * Attributes 
 * alpha    32-bit float attribute
 * 
 * Validation 
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build --command "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir --EmitONNXIR --debug --run-torch-pass third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/test_leakyrelu/model.onnx"
 * 
 */


class ONNXSoftmaxOpToTorchLowering : public ConversionPattern {
public:
  ONNXSoftmaxOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXSoftmaxOp op1 = llvm::dyn_cast<ONNXSoftmaxOp>(op);
    ONNXSoftmaxOpAdaptor adapter(op1);

    auto axis = op1.axisAttr();       // ::mlir::IntegerAttr

    Value input = op1.input();

    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    TensorType data_tensor_type  = input.getType().cast<TensorType>();
    auto dataTy = Torch::ValueTensorType::get(context, data_tensor_type.getShape(),
                                             data_tensor_type.getElementType());

    auto dtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, dataTy, input);

    Value f1v = rewriter.create<ConstantIntOp>(loc,axis);

    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(), 
		    op_tensor_type.getElementType());

    Value f0v = rewriter.create<ConstantBoolOp>(loc, true);
    llvm::outs() << "softmax input is " << input << "\n" << "\n";
    llvm::outs() << "softmax dim is " << f1v << "\n" << "\n";
    llvm::outs() << "softmax dtype is " << dtt << "\n" << "\n";

    Value atensoftmax = rewriter.create<Aten_SoftmaxOp>(loc, resultTy, 
		    dtt, f1v, f0v);

    llvm::outs() << "ATENSOFTMAX CREATED is " << atensoftmax << "\n" << "\n"; 
    Value result = atensoftmax; 

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy, result);
    return success();
  }
};

void populateLoweringONNXToTorchSoftmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXSoftmaxOpToTorchLowering>(typeConverter, ctx);
}
