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

class DecomposeONNXToConstOp : public OpRewritePattern<ONNXConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXConstantOp op,
                                PatternRewriter &rewriter) const override {

    ONNXConstantOpAdaptor adapter = ONNXConstantOpAdaptor(op);
    mlir::MLIRContext *context =  op.getContext();
    Location loc = op.getLoc();

    auto sparese_value_attr = op.sparse_valueAttr(); 		// ::mlir::Attribute
    auto value_attr = op.valueAttr();   			// ::mlir::Attribute
    bool v00 = value_attr.isa<::mlir::FloatAttr>();

    llvm::outs() << "is value_attr of type floatattr :"<<  v00 << "\n" << "\n";
    auto val = adapter.value(); 
    ::mlir::FloatAttr va = adapter.value_floatAttr();


    auto value_flt_attr = op.value_floatAttr(); 			// ::mlir::FloatAttr
    auto value_flts_attr_array = op.value_floatsAttr(); 		// ::mlir::ArrayAttr
    auto value_int_attr = op.value_intAttr(); 				// ::mlir::IntegerAttr
    auto value_ints_attr_array = op.value_intsAttr(); 			// ::mlir::ArrayAttr
    auto value_str_attr = op.value_stringAttr(); 			// ::mlir::StringAttr
    auto value_strs_attr_array = op.value_stringsAttr(); 		// ::mlir::ArrayAttr

        //      Steps
	//	1) Extract float attributes array from ONNX and compare with the Netron file, 
	//	2) Find the shape of this array in step 1,
	//	3) Create the result type, 
	//	4) Create the torch tensor of shape as in 2,
	//	5) Create the torch op and replace it.

    llvm::outs() << "sparese_value_attr:" <<  sparese_value_attr << "\n" << "\n";
    llvm::outs() << "value_attr :" <<  value_attr << "\n" << "\n";
    llvm::outs() << "va  :" <<  va << "\n" << "\n";
    llvm::outs() << "value_flt_attr  :" <<  value_flt_attr << "\n" << "\n";
    llvm::outs() << "value_flts_attr_array  :" <<  value_flts_attr_array << "\n" << "\n";
    llvm::outs() << "value_int_attr :" <<  value_int_attr << "\n" << "\n";
    llvm::outs() << "value_ints_attr_array :" <<  value_ints_attr_array << "\n" << "\n";
    llvm::outs() << "value_str_attr :" <<  value_str_attr << "\n" << "\n";
    llvm::outs() << "value_strs_attr_array :" <<  value_strs_attr_array << "\n" << "\n";


    llvm::outs() << "CONSTFLOATOP operation creation value_attr type: " <<  value_attr.getType() << "\n" << "\n";
    llvm::outs() << "CONSTFLOATOP array tensor type 1: " <<  value_attr << "\n" << "\n";

    TensorType flt_array_tensor_type  = value_attr.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op.getContext(), op_tensor_type.getShape(), op_tensor_type.getElementType());

    llvm::outs() << "CONSTFLOATOP operation creation: result type " << "\n" << resultTy << "\n" << "\n";

    auto one = 1;
    auto three = 3;


    auto ty = IntegerType::get(op.getContext(), 32);
    auto f33 = IntegerAttr::get(ty, three);
    Value f3v = rewriter.create<ConstantIntOp>(loc,f33);
    auto xTy      = Torch::ValueTensorType::get(context, flt_array_tensor_type.getShape(), flt_array_tensor_type.getElementType());

    llvm::outs() << "XTY IS HERE " << "\n" << xTy << "\n"; 

    Value constfloatop = rewriter.create<Torch::ConstantFloatOp>(loc, resultTy, value_attr.cast<FloatAttr>());
    //Value constfloatop = rewriter.create<Torch::ConstantFloatOp>(loc, value_attr.cast<FloatAttr>());

    //llvm::outs() << "constfloatop operation creation" << "\n" << constfloatop << "\n" << "\n";

    Value result = f3v; 

    llvm::outs() << "Before Writer replace Op " << "\n"; 

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(op, op->getResult(0).getType(), result);
 
    llvm::outs() << "After Writer replace Op " << "\n"; 

    return success();
  }
};



} // namespace 


namespace { 

class ONNXToAtenConstantOpTransformPass 
    : public PassWrapper<ONNXToAtenConstantOpTransformPass, OperationPass<::mlir::FuncOp>> {
     StringRef getArgument() const override { return "onnx-to-aten-constop-transform"; }
     void runOnOperation() override {
          MLIRContext *context = &getContext();

	  auto *dialect1 = context->getOrLoadDialect<::mlir::torch::Torch::TorchDialect>();
	  auto *dialect2 = context->getOrLoadDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();

	  RewritePatternSet patterns(context);
	  ConversionTarget target(*context);
	  target.addLegalDialect<Torch::TorchDialect>();
	  target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();
	  target.addLegalDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();
	  
	  llvm::outs() << "ONNXToAtenConstantOpTransformPass Before OpTransform " << "\n"; 
	  patterns.add<DecomposeONNXToConstOp>(context);

	  llvm::outs() << "ONNXToAtenConstantOpTransformPass After OpTransform " << "\n"; 


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
std::unique_ptr<Pass> mlir::createONNXToAtenConstantOpTransformPass() {
  return std::make_unique<ONNXToAtenConstantOpTransformPass>();
}
