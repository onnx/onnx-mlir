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

using namespace mlir;

namespace {

static SmallVector<mlir::FuncOp, 4> lookUpFuncsMatching(
    mlir::ModuleOp module, std::regex pattern) {
  SmallVector<mlir::FuncOp, 4> matchedFuncs;
  module.walk([&](FuncOp funcOp) {
    if (std::regex_search(funcOp.getName().str(), pattern))
      matchedFuncs.emplace_back(funcOp);
  });
  return matchedFuncs;
}

/*!
 *  Function pass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 *
 * Shape inference proceeds recursively, starting with the entry point function
 * corresponding to the main computation graph. This is because sometimes an
 * operation is associated with a different (sub) computation graph in the forms
 * of mlir functions, and the operation's output shape and type depends on the
 * shape and type of that (sub) graph outputs. In such scenarios, operations can
 * initiate shape inference on its dependent (sub) graph, and resume infering
 * its output shape only after shape inference completes for the associated
 * (sub) graph.
 *
 * In the absence of a main computation graph, we will treat every mlir
 * function as a main computation graph; this is mostly just for testing
 * purposes.
 */
class ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass,
                               OperationPass<mlir::ModuleOp>> {
private:
  bool analyzeAllFunctions;

public:
  ShapeInferencePass(bool analyzeAllFunctions_)
      : analyzeAllFunctions(analyzeAllFunctions_) {}

  StringRef getArgument() const override { return "shape-inference"; }

  StringRef getDescription() const override {
    return "Shape inference for frontend dialects.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (!analyzeAllFunctions) {
      auto matchedFuncs =
          lookUpFuncsMatching(module, std::regex("[a-zA-Z0-9_]*main_graph"));
      if (!matchedFuncs.empty()) {
        for (auto func : matchedFuncs) {
          if (failed(runShapeInferenceOn(func)))
            signalPassFailure();
        }
        return;
      }
    }
    auto result = module.walk([&](FuncOp funcOp) -> WalkResult {
      return runShapeInferenceOn(funcOp);
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

  static LogicalResult runShapeInferenceOnRegion(mlir::Region &r) {
    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    for (Operation &op : r.getOps()) {
      std::function<void(mlir::Region &)> doShapeInference =
          &ShapeInferencePass::runShapeInferenceOnRegion;
      // The shape of graph output has been imported from onnx protobuf model,
      // so the ops followed by a return op may not have dynamic shape output.
      // However, shape inference is still need on these ops
      // to infer optional attributes.
      if (containSubgraph(&op) || isUsedByReturnOp(&op) ||
          returnsDynamicOrUnknownShape(&op)) {
        if (auto shape_op = llvm::dyn_cast<ShapeInference>(op)) {
          if (failed(shape_op.inferShapes(doShapeInference))) {
            op.emitError("shape inference failed");
            return failure();
          }
        } else if (!llvm::dyn_cast<CallOpInterface>(op)) {
          op.emitError("unable to infer shape of operation without shape "
                       "inference interface");
          return failure();
        }
      }
    }
    return success();
  }

  static LogicalResult runShapeInferenceOn(mlir::FuncOp f) {
    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape or followed by a return op.
    auto &funcBody = f.getBody();
    if (failed(runShapeInferenceOnRegion(funcBody)))
      return failure();

    // Check if a terminator op exists for function.
    if (!funcBody.empty() && !funcBody.back().empty() &&
        funcBody.back().back().hasTrait<OpTrait::IsTerminator>())
      if (auto returnOp = f.getBody().back().getTerminator()) {
        auto results = returnOp->getOperandTypes();
        f.setType(FunctionType::get(f.getContext(), f.getType().getInputs(),
            std::vector<Type>(results.begin(), results.end())));
      }
    return success();
  }

  static bool isUsedByReturnOp(Operation *op) {
    for (auto *user : op->getUsers()) {
      if (dyn_cast<ReturnOp>(user)) {
        return true;
      }
    }
    return false;
  }

  // Op needs shape inference when contains a subgraph
  // Temporary fix: only LoopOp is checked
  static bool containSubgraph(Operation *op) {
    if (dyn_cast<ONNXLoopOp>(*op))
      return true;
    return false;
  }

  /*!
   *  Check if the given operation has a dynamically shaped result.
   */
  static bool returnsDynamicOrUnknownShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type result_type) {
      if (result_type.isa<RankedTensorType>()) {
        return llvm::any_of(result_type.dyn_cast<RankedTensorType>().getShape(),
            [](int64_t dim) { return dim < 0; });
      } else {
        return !result_type.isa<NoneType>();
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<mlir::Pass> mlir::createShapeInferencePass(
    bool analyzeAllFunctions) {
  return std::make_unique<ShapeInferencePass>(analyzeAllFunctions);
}
