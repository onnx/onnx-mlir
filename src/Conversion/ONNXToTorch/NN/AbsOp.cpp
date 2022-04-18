/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AbsOp.cpp - ONNX Op Transform ------------------===//
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

/* ONNX Abs operation

 * “Absolute takes one input data (Tensor) and produces one output 
 * data" "(Tensor) where the absolute is, y = abs(x), is applied to" "the 
 * tensor elementwise."

 * Operands:
 * Operand Description
   * X	tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned 
   * integer values or tensor of 32-bit unsigned integer values or tensor of 
   * 64-bit unsigned integer values or tensor of 8-bit signless integer values
   * or tensor of 16-bit signless integer values or tensor of 32-bit signless 
   * integer values or tensor of 64-bit signless integer values or tensor of 
   * 16-bit float values or tensor of 32-bit float values or tensor of 64-bit 
   * float values or tensor of bfloat16 type values or memref of any type values.
 * Results:
 * Result Description
   * Y	tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned 
   * integer values or tensor of 32-bit unsigned integer values or tensor of 
   * 64-bit unsigned integer values or tensor of 8-bit signless integer values
   * or tensor of 16-bit signless integer values or tensor of 32-bit signless 
   * integer values or tensor of 64-bit signless integer values or tensor of 
   * 16-bit float values or tensor of 32-bit float values or tensor of 64-bit 
   * float values or tensor of bfloat16 type values or memref of any type values.
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


class ONNXAbsOpToTorchLowering : public ConversionPattern {
public:
  ONNXAbsOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXAbsOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXAbsOp op1 = llvm::dyn_cast<ONNXAbsOp>(op);
    Value x = op1.X();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();


    TensorType x_tensor_type  = x.getType().cast<TensorType>();
    auto xTy      = Torch::ValueTensorType::get(context, x_tensor_type.getShape(),
                    x_tensor_type.getElementType());
    auto xtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, xTy, x);

    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(), 
		    op_tensor_type.getElementType());
    llvm::outs() << "abs input is " << xtt << "\n" << "\n";
    Value atenabs = rewriter.create<AtenAbsOp>(loc, resultTy, xtt); 
    llvm::outs() << "ATENABS CREATED is " << atenabs << "\n" << "\n"; 
    Value result = atenabs; 
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy,
		    result);
    return success();
  }
};

void populateLoweringONNXToTorchAbsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXAbsOpToTorchLowering>(typeConverter, ctx);
}
