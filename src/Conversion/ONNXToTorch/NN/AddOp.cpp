/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AddOp.cpp - ONNX Op Transform ------------------===//
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

/**
 *
 * ONNX Add operation
 *
 * “Performs element-wise binary addition (with Numpy-style broadcasting
 * support).” “” “This operator supports multidirectional
 * (i.e. Numpy-style) broadcasting; for more details please check the doc.”
 *
 * Operands :
 *    A	  tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *    B   tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *
 * Results:
 *   C    tensor of 32-bit/64-bit unsigned integer values or
 *        tensor of 32-bit/64-bit signless integer values or
 *        tensor of 16-bit/32-bit/64-bit float values or
 *        tensor of bfloat16 type values or memref of any type values
 *
 */

class ONNXAddOpToTorchLowering : public ConversionPattern {
public:
  ONNXAddOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXAddOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXAddOp op1 = llvm::dyn_cast<ONNXAddOp>(op);
    ONNXAddOpAdaptor adapter(op1);

    auto a = op1.A();
    auto b = op1.B();
    TensorType a_tensor_type = a.getType().cast<TensorType>();
    TensorType b_tensor_type = b.getType().cast<TensorType>();
    TensorType op_tensor_type =
	    op->getResult(0).getType().cast<TensorType>();
    auto ty = IntegerType::get(op1.getContext(), 64);
    auto one = 1;
    auto f1 = IntegerAttr::get(ty, one);
    Value f1v = rewriter.create<ConstantIntOp>(loc, f1);

    auto aTy = Torch::ValueTensorType::get(
        context, a_tensor_type.getShape(), a_tensor_type.getElementType());
    auto att = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        	loc, aTy, a);
    auto bTy = Torch::ValueTensorType::get(context,
		b_tensor_type.getShape(), b_tensor_type.getElementType());
    auto btt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        	loc, bTy, b);
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());
    Value atenaddresult =
        rewriter.create<AtenAddTensorOp>(loc, resultTy, att, btt, f1v);

    llvm::outs() << "ATENADDOP CREATED is " << atenaddresult << "\n";
    Value result = atenaddresult;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);
    return success();
  }
};

void populateLoweringONNXToTorchAddOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXAddOpToTorchLowering>(typeConverter, ctx);
}
