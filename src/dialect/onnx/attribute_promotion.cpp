//===----- shape_inference_pass.cpp - Shape Inference ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass performing propagation of array
// shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "src/dialect/onnx/promotable_const_operand.hpp"
#include "src/dialect/onnx/onnx_ops.hpp"
#include "src/pass/passes.hpp"

using namespace mlir;

#include "src/dialect/onnx/promotable_const_operand.cpp.inc"

namespace {
/*!
 *  FunctionPass that performs shape inference by iterating over a list of
 *  candidate operations and propagating the shape information until the list
 *  of operations is empty [credit MLIR authors].
 */
class AttributePromotionPass
    : public mlir::FunctionPass<AttributePromotionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    llvm::Optional<mlir::Value> none;
    f.walk([&](mlir::Operation *op) {
      if (op->getNumResults() && op->getOpResult(0).getType().isa<NoneType>())
        none = op->getOpResult(0);
    });

    if (!none) {
      OpBuilder builder(f.getContext());
      builder.setInsertionPointToStart(&f.front());
      none =
          builder.create<mlir::ConstantOp>(f.getLoc(), builder.getUnitAttr());
    }

    // Iterate on the operations that need shape inference i.e the operations
    // that return a dynamic shape.
    f.walk([&](mlir::Operation *op) {
      if (PromotableConstOperandsOpInterface opWithConstOperand =
              dyn_cast<PromotableConstOperandsOpInterface>(op)) {
        auto promotableOperands = opWithConstOperand.promotableConstOperands();
        for (const auto& operandNameToIdx : promotableOperands) {
          auto name = operandNameToIdx.first;
          auto idx = operandNameToIdx.second;

          auto operandToPromote = op->getOperand(idx);
          if (auto constantOp =
                  dyn_cast_or_null<ConstantOp>(operandToPromote.getDefiningOp())) {
            op->setAttr(name, constantOp.value());
            op->setOperand(idx, *none);
          }
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create a Shape Inference pass.
 */
std::unique_ptr<mlir::Pass> mlir::createAttributePromotionPass() {
  return std::make_unique<AttributePromotionPass>();
}

static PassRegistration<AttributePromotionPass> pass(
    "attribute-promotion", "Shape inference for frontend dialects.");
