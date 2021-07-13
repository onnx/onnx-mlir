/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ShapeInferencePass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass performing propagation of array
// shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include <iostream>

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

  std::string getOpName(Operation &op) {
    auto symbolAttr =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    return (op.getName().getStringRef().str());
  }

  // Print all the ops in a module.
  void processModule(ModuleOp module) {
    for (Operation &op : module) {
      // Modules may actually be nested, recurse on nesting.
      if (auto nestedModule = dyn_cast<ModuleOp>(op)) {
        processModule(nestedModule);
        continue;
      }
      auto opName = getOpName(op);
      std::cout << "Operation is " << opName << std::endl;
      for (Region &region : op.getRegions()) {
        for (auto indexed_block : llvm::enumerate(region))
          std::cout << "     block is " << indexed_block.index() << std::endl;
      }
    }
  }

  void runOnOperation() override { processModule(getOperation()); }
  /*
  void runOnOperation() override {
    auto module = getOperation();
    auto result = module.walk([&](Operation &op) -> WalkResult {
      return runOutlineOpOn(op);
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

  static LogicalResult runOutlineOpOnRegion(mlir::Region &r) {
    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    for (Operation &op : r.getOps()) {

    }
    return success();
  }

  static LogicalResult runOutlineOpOn(Operation &op) {
    //auto &funcBody = op.getBody();
    //if (failed(runOutlineOpOnRegion(funcBody)))
    //  return failure();

    return success();
  }
  */

};
} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<mlir::Pass> mlir::createOutlineOperatorsPass() {
  return std::make_unique<OutlineOperatorsPass>();
}
