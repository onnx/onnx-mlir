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
#include "src/Dialect/ONNX/ONNXOps.hpp"


using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

//===----------------------------------------------------------------------===//
// ONNXToAtenTypesTransformPass
//===----------------------------------------------------------------------===//
namespace onnx_mlir {
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
} // namespace onnx_mlir
