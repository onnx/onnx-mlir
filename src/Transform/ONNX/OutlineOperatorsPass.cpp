/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OutlineOperatorsPass.cpp - Operator Outlining ---------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass to outline each operator for debugging purposes
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include <iostream>

using BlockListType = llvm::iplist<mlir::Block>;
using namespace mlir;

namespace {



/*!
 *  Pass that puts each operator in a separate function called from the
 *  main graph
 *  
 */
class OutlineOperatorsPass : public mlir::PassWrapper<OutlineOperatorsPass,
                               OperationPass<mlir::ModuleOp>> {
private:

public:
  OutlineOperatorsPass() {}

  std::string getOpName(Operation *op) {
    auto symbolAttr =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    return (op->getName().getStringRef().str());
  }

  static void genOutlinedFunction(PatternRewriter &rewriter,
      MLIRContext *context, std::string funcName,
      Location loc) {
    /*    
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    llvm::SmallVector<Type, 1> outputsType{opaquePtrTy};

    auto funcType = rewriter.getFunctionType(llvm::None, outputsType);
    llvm::SmallVector<NamedAttribute, 1> attrs;
    auto funcOp = rewriter.create<FuncOp>(
        UnknownLoc::get(context), funcName, funcType, attrs);

    auto entryBlock = funcOp.addEntryBlock();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    auto sigAddr = rewriter.create<LLVM::AddressOfOp>(loc, sigvar);
    auto sigVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, opaquePtrTy, sigAddr);
    llvm::SmallVector<Value, 1> results = {sigVoidPtr};
    rewriter.create<ReturnOp>(UnknownLoc::get(context), results);
    */
  }

  void processOp(Operation *op) {
      //auto onnxtype = ONNXOpsDialect.getTypeID();
      auto opName = getOpName(op);
      std::cout << "Operation is " << opName << std::endl;
      if (op->getDialect()->getNamespace() == "onnx") {
          std::cout << "   is an onnx op" << std::endl;
          if (opName == "onnx.Gemm") {
          std::cout << "   --- outline Gemm" << std::endl;
          }
      }
      for (Region &region : op->getRegions())
         processRegion(region);
  }

 void processRegion(Region &region) {
     std::cout << "   -- entering region" << std::endl;
     for (mlir::Block &block : region.getBlocks())
        processBlock(block);
  } 

  void processBlock(mlir::Block &block) {
    std::cout << "   -- entering block" << std::endl;
    for (mlir::Operation &op : block.getOperations())
        processOp(&op);
  }


  void runOnOperation() override { processOp(getOperation()); }

};
} // end anonymous namespace

/*!
 * Create an Outline Operators pass.
 */
std::unique_ptr<mlir::Pass> mlir::createOutlineOperatorsPass() {
  return std::make_unique<OutlineOperatorsPass>();
}
