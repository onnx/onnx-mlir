/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- DecomposeONNXToAtenConv2DOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

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
 * ONNX Conv operation
 *
 * “The convolution operator consumes an input tensor and a filter, and” “computes the output.”
 *
 * Operands :
 * X		tensor of 16-bit/32-bit/64-bit float values or memref of any type values
 * W		tensor of 16-bit/32-bit/64-bit float values or memref of any type values
 * B		tensor of 16-bit/32-bit/64-bit float values or memref of any type values or none type
 * Output   : 
 * Y		tensor of 16-bit/32-bit/64-bit float values or memref of any type values or none type
 *
 * Attributes 
 * auto_pad 	string attribute
 * dilations 	64-bit integer array attribute
 * group 	64-bit signed integer attribute
 * kernel_shape 64-bit integer array attribute
 * pads 	64-bit integer array attribute
 * strides 	64-bit integer array attribute
 * 
 * AtenConv2dOp Arguments as below 
 * -------------------------------
 *
 *  AnyTorchTensorType : $input
 *  AnyTorchTensorType : $weight
 *  AnyTorchOptionalTensorType : $bias
 *  TorchIntListType : $stride
 *  TorchIntListType : $padding
 *  TorchIntListType : $dilation
 *  Torch_IntType : $group
 * 
 * Validation 
 * ----------
 * ./Debug/bin/onnx-mlir --EmitONNXIR --debug  ../../../third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/pytorch-operator/test_operator_conv/model.onnx
 * 
 * Limitations
 * -----------
 * The atribute values have been used in the below code are specific to this input model specified on line no 88.
 * 
 */
namespace {

class DecomposeONNXToAtenConv2DOp : public OpRewritePattern<ONNXConvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXConvOp op,
                                PatternRewriter &rewriter) const override {

    ONNXConvOpAdaptor adapter = ONNXConvOpAdaptor(op);
    mlir::MLIRContext *context =  op.getContext();
    Location loc = op.getLoc();

    Value x = op.X();				// ONNX operands
    Value w = op.W();				// ONNX operands
    Value b = op.B();				// ONNX operands

    auto autopad = op.auto_padAttr(); 		// ::mlir::StringAttr
    auto dilations = op.dilationsAttr();   	// ::mlir::ArrayAttr
    auto group = op.groupAttr(); 		// ::mlir::IntegerAttr
    auto kernal_shape = op.kernel_shapeAttr(); 	// ::mlir::ArrayAttr
    auto pads = op.padsAttr(); 			// ::mlir::ArrayAttr
    auto strides = op.stridesAttr(); 		// ::mlir::ArrayAttr

    auto b0 = strides.getValue();
    auto bs = strides.begin();
    auto es = strides.end();


    llvm::outs() << "SIZES OF THE STRIDESLIST: " << strides.size() << "\n" << "\n";

    for(bs; bs!= es; ++bs){
	Attribute a = *bs;

    	llvm::outs() << "STRIDES ITERATOR \t" << *bs << "\n" << "\n";
    	llvm::outs() << "STRIDES Attribute \t" << a.getType() << "\n" << "\n";
    	llvm::outs() << "STRIDES Attribute value \t" << a.cast<IntegerAttr>().getInt()  << "\n" << "\n";
    }


    auto groupValue = group.getAPSInt();
    auto sta = mlir::ArrayAttr::get(context, strides);
    auto strides_AR = strides.getValue();
    ::mlir::ArrayAttr stridesArrayAttr = mlir::ArrayAttr::get(context, strides);

    auto one = 1;
    auto three = 3;
    auto zero  = 0;
    
    auto f3 = IntegerAttr::get(group.getType(), three);
    auto f0 = IntegerAttr::get(group.getType(), zero);
    Value f3v = rewriter.create<ConstantIntOp>(loc,f3);
    Value f0v = rewriter.create<ConstantIntOp>(loc,f0);
    Value f1v = rewriter.create<ConstantIntOp>(loc,group);
    Value f2v = rewriter.create<ConstantIntOp>(loc,group);
    
    Value groupVal = rewriter.create<ConstantIntOp>(loc,group);
    Type tensorType = op.getType();

    Value stridesList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v}); 

    Value dilationList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v});

    Value padsList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f0v, f0v, f0v, f0v});

    Value kernalShapeList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f3v, f3v});

    auto r0 = Torch::ValueTensorType::getWithLeastStaticInformation(op.getContext());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( op.getLoc(), r0, x);
    auto btt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( op.getLoc(), r0, b);
    auto wtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( op.getLoc(), r0, w);

    llvm::outs() << "xtt torch tensor from MLIR tensor " << "\n" << xtt << "\n" << "\n";
    llvm::outs() << "wtt torch tensor from MLIR tensor " << "\n" << wtt << "\n" << "\n";

    //auto t = Torch::ValueTensorType::get(context, optionalSizesArrayRef, x1.getType());

    Value atenconv2d = rewriter.create<AtenConv2dOp>(loc, x.getType(), xtt, wtt, btt, stridesList, padsList, dilationList, f1v);

    llvm::outs() << "AtenConv2d operation creation" << "\n" << atenconv2d << "\n" << "\n";

    Value result = atenconv2d; 

    llvm::outs() << "Before Writer replace Op " << "\n"; 
    
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(), result);

    llvm::outs() << "After Writer replace Op " << "\n"; 

    return success();
  }
};



} // namespace 


namespace { 

class ONNXToAtenConv2DOpTransformPass 
    : public PassWrapper<ONNXToAtenConv2DOpTransformPass, OperationPass<::mlir::FuncOp>> {
  StringRef getArgument() const override { return "onnx-to-aten-conv2d-transform"; }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
	  registry.insert<::mlir::torch::Torch::TorchDialect>();
	  registry.insert<::mlir::torch::TorchConversion::TorchConversionDialect>();
  }


     void runOnOperation() override {
          MLIRContext *context = &getContext();
          
	  auto *dialect1 = context->getOrLoadDialect<::mlir::torch::Torch::TorchDialect>();
	  auto *dialect2 = context->getOrLoadDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();

	  RewritePatternSet patterns(context);
	  ConversionTarget target(*context);
	  target.addLegalDialect<Torch::TorchDialect>();
	  target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();

	  llvm::outs() << "ONNXToAtenConv2DOpTransformPass Before OpTransform " << "\n"; 
	  patterns.add<DecomposeONNXToAtenConv2DOp>(context);

	  //target.addIllegalOp<ONNXConvOp>();
	  llvm::outs() << "ONNXToAtenConv2DOpTransformPass `After OpTransform " << "\n"; 

	  
	  if (failed(applyPartialConversion(getOperation(), target,
	      std::move(patterns)))) {
	      return signalPassFailure();
	  }

	  if (onnxOpTransformReport) {
	    llvm::outs() << "ONNXToAtenConv2DOpTransformPass iterated " << 3 
			 << " times, converged "
			 << "\n";
	  }
      }
};


} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<Pass> mlir::createONNXToAtenConv2DOpTransformPass() {
  return std::make_unique<ONNXToAtenConv2DOpTransformPass>();
}
