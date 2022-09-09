/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentONNXSignaturePass.cpp - Instrumentation
//---------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts krnl print statements
// that print the operation name and its input type signature at runtime.
//
//===----------------------------------------------------------------------===//

#include <set>

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 * This pass insert KrnlPrint and KrnlPrintTensor before each ONNX ops to print
 * an operation name and input operand type signatures at runtime.
 */

class InstrumentONNXSignaturePass
    : public mlir::PassWrapper<InstrumentONNXSignaturePass,
          OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentONNXSignaturePass)

  InstrumentONNXSignaturePass() = default;
  InstrumentONNXSignaturePass(const InstrumentONNXSignaturePass &pass)
      : mlir::PassWrapper<InstrumentONNXSignaturePass,
            OperationPass<func::FuncOp>>() {}

private:
public:
  StringRef getArgument() const override {
    return "instrument-onnx-runtime-signature";
  }

  StringRef getDescription() const override {
    return "instrument on onnx ops to print their input operand's type "
           "signature";
  }

  void runOnOperation() override {
    // Iterate on the operations nested in this function.
    getOperation().walk([&](mlir::Operation *op) {
      if (isa<ONNXDialect>(op->getDialect())) {
        if (!isa<ONNXPrintSignatureOp>(op)) {
          Location loc = op->getLoc();
          OpBuilder builder(op);
          ValueRange operands = op->getOperands();
          std::string opName(
              op->getName().getStringRef().data() + 5); // Skip "onnx.".
          StringAttr opNameAttr = builder.getStringAttr(opName);
          if (isa<ONNXConstantOp>(op)) {
            // Constant has the type in the output, so use it.
            operands = op->getResults();
            // Since we use the result of the constant operation, we must insert
            // the print operation after the constant operation.
            builder.setInsertionPointAfter(op);
            builder.create<ONNXPrintSignatureOp>(loc, opNameAttr, operands);
          } else if (operands.size() > 0) {
            builder.create<ONNXPrintSignatureOp>(loc, opNameAttr, operands);
          }
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentONNXSignaturePass() {
  return std::make_unique<InstrumentONNXSignaturePass>();
}
