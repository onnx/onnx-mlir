//===----- attribute_promotion.cpp - Attribute Promotion
//-------------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a function level pass to move an operand to become
// an attribute if desirable and legal.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Interface/PromotableConstOperandsOpInterface.hpp"
#include "src/Pass/Passes.hpp"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

/*!
 * Helper function to create a NoneTyped constant value if `none` is empty.
 */
void getOrCreateNoneValue(llvm::Optional<mlir::Value> &none, FuncOp f) {
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
          // move it to an attribute, and use None to indicate the absence
          // of the original operand value.
          auto operandToPromote = op->getOperand(i);
          if (auto constantOp = dyn_cast_or_null<mlir::ONNXConstantOp>(
                  operandToPromote.getDefiningOp())) {
            if (constantOp.valueAttr() &&
                !constantOp.valueAttr().dyn_cast_or_null<UnitAttr>())
              op->setAttr(name, constantOp.valueAttr());
            if (constantOp.sparse_valueAttr() &&
                !constantOp.sparse_valueAttr().dyn_cast_or_null<UnitAttr>())
              op->setAttr(name, constantOp.sparse_valueAttr());
            getOrCreateNoneValue(none, f);
            op->setOperand(i, *none);
          }
          if (auto constantOp = dyn_cast_or_null<ConstantOp>(
                  operandToPromote.getDefiningOp())) {
            if (!constantOp.valueAttr().dyn_cast_or_null<UnitAttr>()) {
              op->setAttr(name, constantOp.value());
              getOrCreateNoneValue(none, f);
              op->setOperand(i, *none);
            }
          }
        }
      }
    });

    // Dispatch canonicalization pattern rewriters to eliminate redundant
    // constant operaions.
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    ConstantOp::getCanonicalizationPatterns(patterns, context);
    applyPatternsGreedily(f, patterns);
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
    "attribute-promotion", "Promote constant operands to attributes.");
