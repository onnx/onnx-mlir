/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ReluOp.cpp - ONNX Op Transform ------------------===//
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
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 *
 * ONNX Relu operation
 *
 * “Relu takes one input data (Tensor) and produces one output data" "
 * (Tensor) where the rectified linear function, y = max(0, x), is applied 
 * to" "the tensor elementwise."
 *
 * Operands :
 *    X    tensor of 16-bit/32-bit/64-bit float values or memref of any 
 *         type values
 *
 * Results:
 *    Y	   tensor of 16-bit/32-bit/64-bit float values or tensor of 
 *         bfloat16 type values or memref of any type values
 *
 * Validation
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build
 * --command
 * "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
 * --EmitONNXIR --debug --run-torch-pass
 * third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/
 * test_relu/model.onnx"
 *
 */

class ONNXReluOpToTorchLowering : public ConversionPattern {
public:
  ONNXReluOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXReluOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXReluOp op1 = llvm::dyn_cast<ONNXReluOp>(op);
    ONNXReluOpAdaptor adapter(op1);

    Value x = op1.X();

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(
        context, x_tensor_type.getShape(), x_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());

    llvm::outs() << "resultTy is: \n " << resultTy << "\n"
                 << "\n";
    llvm::outs() << "xtt is: \n " << xtt << "\n"
                 << "\n";

    Value atenrelu = rewriter.create<AtenReluOp>(loc, resultTy, xtt);

    llvm::outs() << "ATENRELU CREATED is: \n"
                 << atenrelu << "\n"
                 << "\n";
    Value result = atenrelu;

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op,
		    resultTy, result);
    return success();
  }
};

void populateLoweringONNXToTorchReluOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReluOpToTorchLowering>(typeConverter, ctx);
}
