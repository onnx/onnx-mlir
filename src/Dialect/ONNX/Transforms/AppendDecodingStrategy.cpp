/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AppendDecodingStrategy.cpp - Decoding Strategy ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that appends a decoding strategy such as greedy
// algorith, top_k, top_p, to the end of the main graph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_APPENDDECODINGSTRATEGYPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

namespace {

class AppendDecodingStrategyPass
    : public onnx_mlir::impl::AppendDecodingStrategyPassBase<
          AppendDecodingStrategyPass> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AppendDecodingStrategyPass)

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find the ONNXEntryPoint operation that knows which is the main function.
    ONNXEntryPointOp entryPointOp;
    moduleOp.walk([&entryPointOp](ONNXEntryPointOp op) -> WalkResult {
      entryPointOp = op;
      return WalkResult::advance();
    });
    if (!entryPointOp)
      return;

    // Get the main function.
    SymbolRefAttr funcRefAttr =
        entryPointOp.getOperation()->getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName());
    StringRef mainFuncName = funcRefAttr.getLeafReference().getValue();
    Operation *mainFuncOp = moduleOp.lookupSymbol(mainFuncName);
    if (!mainFuncOp)
      return;
    func::FuncOp mainFunc = mlir::dyn_cast<func::FuncOp>(mainFuncOp);
    if (!mainFunc)
      return;

    // Find the ONNXReturnOp.
    mlir::Operation *term = mainFunc.getBody().back().getTerminator();
    ONNXReturnOp returnOp = mlir::dyn_cast<ONNXReturnOp>(term);
    if (!returnOp)
      return;
    if (returnOp.getOperands().size() < 1)
      return;

    // Emit computation for the decoding strategy.
    OpBuilder b(returnOp);
    Location loc = returnOp.getLoc();
    Value logits = returnOp.getOperands()[0];
    Operation *decodingOp = emitGreedyAlgorithm(b, loc, logits);
    Value nextTokens = decodingOp->getResults()[0];

    // Replace the first return value by the output of the decoding strategy.
    returnOp.getOperation()->moveAfter(decodingOp);
    returnOp->setOperand(0, nextTokens);

    // Update the function signature to reflect the new return type.
    SmallVector<Type> newResultTypes;
    newResultTypes.push_back(nextTokens.getType());
    for (unsigned i = 1; i < returnOp.getOperands().size(); ++i)
      newResultTypes.push_back(returnOp.getOperands()[i].getType());
    FunctionType newFuncType = FunctionType::get(
        mainFunc.getContext(), mainFunc.getArgumentTypes(), newResultTypes);
    mainFunc.setFunctionType(newFuncType);
  }

private:
  Operation *emitGreedyAlgorithm(OpBuilder b, Location loc, Value logits) {
    UnrankedTensorType resultType = UnrankedTensorType::get(
        cast<TensorType>(logits.getType()).getElementType());
    ONNXArgMaxOp argmaxOp = ONNXArgMaxOp::create(b, loc, resultType, logits,
        /*axis*/ -1, /*keepdims*/ 1, /*select_last_index*/ 0);
    return argmaxOp.getOperation();
  }
};

} // namespace
