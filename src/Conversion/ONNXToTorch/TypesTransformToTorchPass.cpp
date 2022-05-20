//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ADT/StringExtras.h"

#include "src/Pass/Passes.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

static void
setupValueTensorToBuiltinTensorConversion(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToBuiltinTensorOp,
                    TorchConversion::FromBuiltinTensorOp>();
  typeConverter.addConversion(
      [](TensorType type) -> Optional<Type> {
        return Torch::ValueTensorType::get(type.getContext(), type.getShape(), type.getElementType());
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, Torch::ValueTensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<torch::TorchConversion::FromBuiltinTensorOp>(loc, type, inputs[0]);
  });

  auto sourceMaterialization = [](OpBuilder &builder,
                                  TensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, inputs[0]);
  };

  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

/*
static void setupTorchBoolToI1Conversion(ConversionTarget &target,
                                         TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToI1Op, TorchConversion::FromI1Op>();
  typeConverter.addConversion([](Torch::BoolType type) -> Optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    // Other builtin integer types could be handled by other materializers.
    if (!(type.getWidth() == 1 && type.isSignless()))
      return None;
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BoolType>());
    return builder.create<ToI1Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::BoolType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<FromI1Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}
static void setupTorchIntToI64Conversion(ConversionTarget &target,
                                         TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToI64Op, TorchConversion::FromI64Op>();
  typeConverter.addConversion([](Torch::IntType type) -> Optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    // Other builtin integer types could be handled by other materializers.
    if (!(type.getWidth() == 64 && type.isSignless()))
      return None;
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::IntType>());
    return builder.create<ToI64Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::IntType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<FromI64Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}
static void setupTorchFloatToF64Conversion(ConversionTarget &target,
                                           TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToF64Op, TorchConversion::FromF64Op>();
  typeConverter.addConversion([](Torch::FloatType type) -> Optional<Type> {
    return Float64Type::get(type.getContext());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            Float64Type type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::FloatType>());
    return builder.create<ToF64Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::FloatType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Float64Type>());
    return builder.create<FromF64Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
} */

void setupBackendTypeTransforms(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupValueTensorToBuiltinTensorConversion(target, typeConverter);
  //setupTorchBoolToI1Conversion(target, typeConverter);
  //setupTorchIntToI64Conversion(target, typeConverter);
  //setupTorchFloatToF64Conversion(target, typeConverter);
}

//===----------------------------------------------------------------------===//
// ONNXToAtenTypesTransformPass
//===----------------------------------------------------------------------===//
namespace onnx_mlir {
class ONNXToAtenTypesTransformPass 
    : public PassWrapper<ONNXToAtenTypesTransformPass, OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<Torch::TorchDialect>();
    target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();
    target.addLegalDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();

    target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
    
    typeConverter.addConversion([](Type type) { return type; });
    setupBackendTypeTransforms(target, typeConverter);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<Pass> createONNXToAtenTypesTransformPass() {
  return std::make_unique<ONNXToAtenTypesTransformPass>();
}
}

//===----------------------------------------------------------------------===//
// FinalizingBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};
} // namespace


template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(typeConverter,
                                              patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

namespace onnx_mlir {
class ONNXToAtenFinalizeTypesTransformPass
    : public PassWrapper<ONNXToAtenFinalizeTypesTransformPass, OperationPass<func::FuncOp>> {

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    
    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
 
    typeConverter.addConversion([](Type type) { return type; });
    setupBackendTypeTransforms(target, typeConverter);

    target.addLegalDialect<Torch::TorchDialect>();
 
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
    
    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    setupFinalization<ToBuiltinTensorOp, FromBuiltinTensorOp>(target, patterns,
                                                              typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
					 [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

class ONNXToAtenModifyMainFunctionPass
    : public PassWrapper<ONNXToAtenModifyMainFunctionPass, OperationPass<::mlir::ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module  = getOperation();
    auto *context    = &getContext();
    
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    module.walk([&](ONNXEntryPointOp op) {
	auto functionName = op.func().getRootReference().getValue();
	auto mainFuncOp   = module.lookupSymbol<func::FuncOp>(functionName);
	if (mainFuncOp) {
	  StringRef forwardRef = "forward";
	  auto forwardAttr     = StringAttr::get(module.getContext(), forwardRef);	  
	  mainFuncOp->setAttr(llvm::StringRef("sym_name"), forwardAttr);
	}
	op.erase();	
      });
    
    target.addIllegalOp<ONNXEntryPointOp>();
    
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createONNXToAtenModifyMainFunctionPass() {
  return std::make_unique<ONNXToAtenModifyMainFunctionPass>();
}

std::unique_ptr<Pass> createONNXToAtenFinalizeTypesTransformPass() {
  return std::make_unique<ONNXToAtenFinalizeTypesTransformPass>();
}

} // namespace onnx_mlir
