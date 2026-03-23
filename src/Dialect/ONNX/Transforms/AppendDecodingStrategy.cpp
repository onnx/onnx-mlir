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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
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

    // Find the entry point function.
    func::FuncOp mainFunc = getMainFunc(moduleOp);
    if (!mainFunc)
      return;

    // Find the ONNXReturnOp.
    Operation *term = mainFunc.getBody().back().getTerminator();
    ONNXReturnOp returnOp = mlir::dyn_cast<ONNXReturnOp>(term);
    if (!returnOp || returnOp.getOperands().size() < 1)
      return;

    // Start rewriting.
    OpBuilder b(term->getContext());
    Location loc = term->getLoc();
    OnnxBuilder createOnnx(b, loc);

    // Insert operations before the terminator.
    OpBuilder::InsertPoint savedIP = b.saveInsertionPoint();
    b.setInsertionPoint(term);
    Value logits = returnOp.getOperands()[0];
    UnrankedTensorType unrankedType = UnrankedTensorType::get(
        cast<TensorType>(logits.getType()).getElementType());
    // Emit code to extract logits for the last token.
    // ```python
    // last_token_logits = logits[batch_size, -1, :]
    // ```
    Value lastTokenLogits = createOnnx.slice(unrankedType, logits,
        /*starts=*/createOnnx.constantInt64({-1}),
        /*ends=*/createOnnx.constantInt64({INT_MAX}),
        /*axes=*/createOnnx.constantInt64({1}),
        /*steps=*/createOnnx.constantInt64({1}));

    // Emit code for the decoding strategy.
    Value nextTokens = emitDecodingStrategies(createOnnx, lastTokenLogits);
    // Squeeze nextTokens from the shape [batch_size, 1, 1] to [batch_size, 1].
    nextTokens = createOnnx.squeeze(
        unrankedType, nextTokens, createOnnx.constantInt64({1}));

    // Restore the original insertion point.
    b.restoreInsertionPoint(savedIP);

    // Replace the first return value by the output of the decoding strategy.
    returnOp->setOperand(0, nextTokens);

    // Update the function signature to reflect the new return type.
    SmallVector<Type> newResultTypes;
    newResultTypes.push_back(nextTokens.getType());
    for (unsigned i = 1; i < returnOp.getOperands().size(); ++i)
      newResultTypes.push_back(returnOp.getOperands()[i].getType());
    FunctionType newFuncType = FunctionType::get(
        mainFunc.getContext(), mainFunc.getArgumentTypes(), newResultTypes);
    mainFunc.setFunctionType(newFuncType);
    // Update onnx.name of the 1st output.
    NamedAttribute namedAttr =
        b.getNamedAttr("onnx.name", b.getStringAttr("generated_ids"));
    mainFunc.setResultAttrs(0, namedAttr);
  }

private:
  func::FuncOp getMainFunc(ModuleOp moduleOp) const {
    // Find the ONNXEntryPoint operation that knows which is the main function.
    ONNXEntryPointOp entryPointOp;
    moduleOp.walk([&entryPointOp](ONNXEntryPointOp op) -> WalkResult {
      entryPointOp = op;
      return WalkResult::advance();
    });
    if (!entryPointOp)
      return nullptr;

    // Get the main function.
    SymbolRefAttr funcRefAttr =
        entryPointOp.getOperation()->getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName());
    StringRef mainFuncName = funcRefAttr.getLeafReference().getValue();
    Operation *mainFuncOp = moduleOp.lookupSymbol(mainFuncName);
    if (!mainFuncOp)
      return nullptr;
    func::FuncOp mainFunc = mlir::dyn_cast<func::FuncOp>(mainFuncOp);
    if (!mainFunc)
      return nullptr;

    return mainFunc;
  }

  Value emitDecodingStrategies(OnnxBuilder &createOnnx, Value logits) const {
    UnrankedTensorType resultType = UnrankedTensorType::get(
        cast<TensorType>(logits.getType()).getElementType());
    // Greed algorithm.
    Value argmax = createOnnx.argMax(resultType, logits,
        /*axis*/ -1, /*keepdims*/ 1, /*select_last_index*/ 0);
    return argmax;
  }
};

} // namespace
