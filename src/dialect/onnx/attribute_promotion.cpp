//===----- attribute_promotion.cpp - Attribute Promotion -------------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a function level pass to move an operand to become
// an attribute if desirable and legal.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "src/interface/promotable_const_operands.hpp"
#include "src/pass/passes.hpp"

using namespace mlir;



namespace {

/*!
 * Helper function to create a NoneTyped constant value if `none` is empty.
 */
static void getOrCreateNoneValue(llvm::Optional<mlir::Value> &none, FuncOp f) {
  if (none.hasValue())
    return;

  OpBuilder builder(f.getContext());
  builder.setInsertionPointToStart(&f.front());
  none = builder.create<mlir::ConstantOp>(f.getLoc(), builder.getUnitAttr());
}

/*!
 *  FunctionPass that performs attribute promotion by iterating over a list of
 *  candidate operations and moves constant operands to attributes whenever
 *  desirable (as instructed by the PromotableConstOperandsOpInterface).
 */
class AttributePromotionPass
    : public mlir::FunctionPass<AttributePromotionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    // A function-scope shared none value used to indicate an missing operand.
    llvm::Optional<mlir::Value> none;

    // Iterate on the operations that may need attribute promotion.
    f.walk([&](mlir::Operation *op) {
      if (PromotableConstOperandsOpInterface opWithConstOperand =
              dyn_cast<PromotableConstOperandsOpInterface>(op)) {
        auto promotableOperands = opWithConstOperand.promotableConstOperands();
        for (const auto &operandNameToIdx : promotableOperands) {
          auto name = operandNameToIdx.first;
          auto i = operandNameToIdx.second;

          // If the i-th operand is defined by an constant operation, then
          // move it to become an attribute.
          auto operandToPromote = op->getOperand(i);
          if (auto constantOp = dyn_cast_or_null<ConstantOp>(
                  operandToPromote.getDefiningOp())) {
            op->setAttr(name, constantOp.value());
            getOrCreateNoneValue(none, f);
            op->setOperand(i, *none);
          }
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create a Attribute Promotion pass.
 */
std::unique_ptr<mlir::Pass> mlir::createAttributePromotionPass() {
  return std::make_unique<AttributePromotionPass>();
}

static PassRegistration<AttributePromotionPass> pass(
    "attribute-promotion", "Pass to promote constant operands to attributes.");
