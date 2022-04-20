/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- GlobalAveragePool.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ========================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------------===//

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

/*
 * ONNX GlobalAveragePool operation
 *
 * “GlobalAveragePool consumes an input tensor X and applies average 
 * pooling across” “ the values in the same channel. 
 * This is equivalent to AveragePool with kernel size” “ equal to the 
 * spatial dimension of input tensor.”
 *
 * Operands:
 *  X	tensor of 16-bit/32-bit/64-bit float values or memref of any 
 *      type values
 * Results:
 *  Y	tensor of 16-bit/32-bit/64-bit float values or memref of any 
 *      type values
 *
 */ 
struct ONNXGlobalAveragePoolOpToTorchLowering : public ConversionPattern {
public:
  ONNXGlobalAveragePoolOpToTorchLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXGlobalAveragePoolOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXGlobalAveragePoolOp op1 = 
	    llvm::dyn_cast<ONNXGlobalAveragePoolOp>(op);
    ONNXGlobalAveragePoolOpAdaptor adapter(op1);
    mlir::MLIRContext *context = op1.getContext();
    Location loc = op1.getLoc();

    Value x = op1.X(); // ONNX operands

    auto ty = IntegerType::get(op1.getContext(), 64);
    auto inputShape = x.getType().cast<ShapedType>().getShape();
    int64_t inputRank = inputShape.size();
    auto f1 = IntegerAttr::get(ty, inputRank);
    Value f1v = rewriter.create<ConstantIntOp>(loc, f1);

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = 
	    op->getResult(0).getType().cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(
        context, x_tensor_type.getShape(), x_tensor_type.getElementType());
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);

    llvm::outs() << "\n resultTy:"
                 << "\n"
                 << resultTy << "\n"
                 << "\n";
    llvm::outs() << "xtt torch tensor from MLIR tensor:"
                 << "\n"
                 << xtt << "\n"
                 << "\n";

    Value atenGlobAvgpool2d =
        rewriter.create<AtenAdaptiveAvgPool2dOp>(loc, resultTy, xtt, f1v);

    llvm::outs() << "AtenAdaptiveAvgPool2dOp operation creation"
                 << "\n"
                 << atenGlobAvgpool2d << "\n"
                 << "\n";

    Value result = atenGlobAvgpool2d;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchGlobalAveragePoolOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGlobalAveragePoolOpToTorchLowering>(typeConverter, ctx);
}
