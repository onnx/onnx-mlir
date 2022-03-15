/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv2D.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Torch dialect.
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

typedef struct dim_pads{
  int dim_start;
  int dim_end;
} dim_pads;

struct ONNXConvOpToTorchLowering : public ConversionPattern {
  ONNXConvOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    
    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp op1 = llvm::dyn_cast<ONNXConvOp>(op);

    mlir::MLIRContext *context =  op1.getContext();

    Value x = op1.X();                           // ONNX operands
    Value w = op1.W();                           // ONNX operands
    Value b = op1.B();                           // ONNX operands
    bool biasIsNone = b.getType().isa<mlir::NoneType>();
    
    auto autopad = op1.auto_padAttr();           // ::mlir::StringAttr
    auto dilations = op1.dilationsAttr();        // ::mlir::ArrayAttr
    auto group = op1.groupAttr();                // ::mlir::IntegerAttr
    auto kernal_shape = op1.kernel_shapeAttr();  // ::mlir::ArrayAttr
    auto pads = op1.padsAttr();                  // ::mlir::ArrayAttr
    auto strides = op1.stridesAttr();            // ::mlir::ArrayAttr

    auto b0 = strides.getValue();
    auto bs = strides.begin();   
    auto es = strides.end();

    auto groupValue = group.getAPSInt();
    auto strides_AR = strides.getValue();
    ::mlir::ArrayAttr stridesArrayAttr = mlir::ArrayAttr::get(context, strides);

    auto one = 1;
    auto three = 3;
    auto zero  = 0;
    
    // new implementation
    dim_pads dimArray[pads.size()];
    int j = 0;
    for (unsigned int i = 0; i < pads.size(); i++) {
      dimArray[j].dim_start = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      i++;
      dimArray[j].dim_end = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      j++;
    }
    int translateArray[pads.size()];
    int k = 0;
    for (unsigned int i = 0; i < pads.size(); i=i+2) {
      llvm::outs() << "DimStartValue   : " << "\n" << dimArray[k].dim_start << "\n" << "\n";
      translateArray[k] = dimArray[k].dim_start;
      k++;
    }

    int l = 0;
    for (unsigned int i = 0; i < pads.size(); i=i+2) {
      llvm::outs() << "dimEndValue   : " << "\n" << dimArray[l].dim_end << "\n" << "\n";
      translateArray[k] = dimArray[l].dim_end;
      k++;
      l++;
    }

    auto f0 = IntegerAttr::get(group.getType(), (pads[0].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
    auto f2 = IntegerAttr::get(group.getType(), (pads[2].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
    auto f1 = IntegerAttr::get(group.getType(), (pads[1].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
    auto f3 = IntegerAttr::get(group.getType(), (pads[3].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
   
      
    Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
    Value p1v = rewriter.create<ConstantIntOp>(loc, f2);
    Value p2v = rewriter.create<ConstantIntOp>(loc, f1);
    Value p3v = rewriter.create<ConstantIntOp>(loc, f3);
    //
    //
    Value f3v = rewriter.create<ConstantIntOp>(loc,f3);
    Value f0v = rewriter.create<ConstantIntOp>(loc,f0);
    Value f1v = rewriter.create<ConstantIntOp>(loc,group);
    Value f2v = rewriter.create<ConstantIntOp>(loc,group);

    Value groupVal = rewriter.create<ConstantIntOp>(loc,group);

    Value stridesList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v}); 

    Value dilationList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v});

    Value padsList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{p0v, p1v, p2v, p3v});

    Value kernalShapeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f3v, f3v});

    TensorType x_tensor_type  = x.getType().cast<TensorType>();
    TensorType w_tensor_type  = w.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto xTy      = Torch::ValueTensorType::get(context, x_tensor_type.getShape(), x_tensor_type.getElementType());
    auto wTy      = Torch::ValueTensorType::get(context, w_tensor_type.getShape(), w_tensor_type.getElementType());
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(), op_tensor_type.getShape(), op_tensor_type.getElementType());

    auto xtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, xTy, x); 
    auto wtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, wTy, w); 
    Value btt;
    if (biasIsNone) {
      btt = rewriter.create<Torch::ConstantNoneOp>( loc);
    }   
    else {
      TensorType b_tensor_type  = b.getType().cast<TensorType>();
      auto bTy = Torch::ValueTensorType::get(op1.getContext(), b_tensor_type.getShape(), b_tensor_type.getElementType());
      btt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, bTy, b); 
    }   
    llvm::outs() << "loc   : " << "\n" << loc << "\n" << "\n";
    llvm::outs() << "ResultType : " << "\n" << resultTy << "\n" << "\n";
    llvm::outs() << "xtt torch tensor from MLIR tensor " << "\n" << xtt << "\n" << "\n";
    llvm::outs() << "wtt torch tensor from MLIR tensor " << "\n" << wtt << "\n" << "\n";
    llvm::outs() << "btt : " << "\n" << btt << "\n" << "\n";
    llvm::outs() << "stridesList:   " << "\n" << stridesList << "\n" << "\n";
    llvm::outs() << "padsList:   " << "\n" << padsList << "\n" << "\n";
    llvm::outs() << "dilationList:   " << "\n" << dilationList << "\n" << "\n";
    llvm::outs() << "f1v:   " << "\n" << f1v << "\n" << "\n";
    Value atenconv2d = rewriter.create<AtenConv2dOp>(loc, resultTy, xtt, wtt, btt, stridesList, padsList, dilationList, f1v);

    llvm::outs() << "AtenConv2d operation creation" << "\n" << atenconv2d << "\n" << "\n";

    Value result = atenconv2d; 

    llvm::outs() << "Before Writer replace Op " << "\n"; 

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(op, op->getResult(0).getType(), result);
 
    llvm::outs() << "After Writer replace Op " << "\n"; 

    llvm::outs() << "matchAndRewrite of conversionPattern is called successfully " << "\n";
    return success();
  }
};

void populateLoweringONNXToTorchConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXConvOpToTorchLowering>(typeConverter, ctx);
}
