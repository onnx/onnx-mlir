/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ConcatOp.cpp - ONNX Op Transform ------------------===//
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


class ONNXConcatOpToTorchLowering : public ConversionPattern {
public:
  ONNXConcatOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ::mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXConcatOp op1 = llvm::dyn_cast<ONNXConcatOp>(op);
    ONNXConcatOpAdaptor adaptor(op1);
    
    ValueRange inputs = op1.inputs();
    auto axisValue = op1.axisAttr();       // ::mlir::IntegerAttr
    Value  axisVal = rewriter.create<ConstantIntOp>(loc,axisValue);
    
    TensorType op_tensor_type = op1.getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(),
                                                      op_tensor_type.getElementType());
    std::vector<Value> inputArrayValues;
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
      TensorType x_tensor_type = inputs[i].getType().cast<TensorType>();
      auto xTy      = Torch::ValueTensorType::get(context, x_tensor_type.getShape(),
                    x_tensor_type.getElementType());
      auto xtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
		      loc, xTy, inputs[i]);
      inputArrayValues.push_back(xtt);
    }
    Value inputShapeList = rewriter.create<PrimListConstructOp>(loc, 
       Torch::ListType::get(inputArrayValues.front().getType()), 
       			ValueRange{inputArrayValues}); 
    llvm::outs() << "inputs Value:   " << "\n" << inputShapeList << "\n" << "\n";
    llvm::outs() << "axisValue Value:   " << "\n" << axisValue << "\n" << "\n";

    Value atenconcat = rewriter.create<AtenCatOp>(loc, resultTy, inputShapeList, axisVal);
    
    Value result = atenconcat;
    llvm::outs() << "AtenConcat Op created" << "\n" << "\n";

    llvm::outs() << "Aten Concat Op:   " << "\n" << atenconcat << "\n" << "\n";
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy,
		    result);
    return success();
  }
};

void populateLoweringONNXToTorchConcatOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXConcatOpToTorchLowering>(typeConverter, ctx);
}
