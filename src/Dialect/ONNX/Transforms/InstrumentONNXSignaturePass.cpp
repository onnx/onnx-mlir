/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentONNXSignaturePass.cpp - Instrumentation ------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts statements that print
// the operation name and its input type signature at runtime.
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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 * This pass insert ONNXPrintSignatureOp before each ONNX ops to print
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
            OperationPass<func::FuncOp>>() {
    signaturePattern = pass.signaturePattern;
    nodeNamePattern = pass.nodeNamePattern;
  }
  InstrumentONNXSignaturePass(
      const std::string opPattern, const std::string nodePattern)
      : signaturePattern(opPattern), nodeNamePattern(nodePattern) {}

private:
  std::string signaturePattern;
  std::string nodeNamePattern;

public:
  StringRef getArgument() const override {
    return "instrument-onnx-runtime-signature";
  }

  StringRef getDescription() const override {
    return "instrument on onnx ops to print their input operand's type "
           "signature";
  }

  void runOnOperation() override {
    onnx_mlir::EnableByRegexOption traceSpecificOpPattern(

        /*emptyIsNone*/ false);
    onnx_mlir::EnableByRegexOption traceSpecificNodePattern(
        /*emptyIsNone*/ false);

    traceSpecificOpPattern.setRegexString(signaturePattern);
    traceSpecificNodePattern.setRegexString(nodeNamePattern);
    // Iterate on the operations nested in this function.
    getOperation().walk([&](mlir::Operation *op) {
      auto dialect = op->getDialect();
      Location loc = op->getLoc();
      // Define a lambda function to check whether the node is selected by
      // its op name or node name, and if yes, insert ONNXSignatureOp
      auto checkAndInsert = [&](onnx_mlir::EnableByRegexOption &pattern,
                                std::string matchString, int detail) {
        if (pattern.isEnabled(matchString)) {
          // Add signature printing op.
          OpBuilder builder(op);
          std::string opName = op->getName().getStringRef().str();
          std::string nodeName = onnx_mlir::getNodeNameInPresenceOfOpt(op);
          std::string fullName = opName + ", " + nodeName;
          StringAttr fullNameAttr = builder.getStringAttr(fullName);
          // Enqueue all input operands, and then the results.
          llvm::SmallVector<Value, 6> operAndRes(op->getOperands());
          for (Value res : op->getResults())
            operAndRes.emplace_back(res);
          // Since we may use the result of an operation, we must insert the
          // print operation after the operation.
          builder.setInsertionPointAfter(op);
          // When one node is selected, print the details of the tensor.
          builder.create<ONNXPrintSignatureOp>(
              loc, fullNameAttr, detail, operAndRes);
        }
      };

      if (isa<func::FuncDialect>(dialect) || isa<ONNXPrintSignatureOp>(op)) {
        // Always skip function dialects (such as function call/return), as well
        // as ONNX print signature ops.
      } else if (signaturePattern != "NONE") {
        std::string opName = op->getName().getStringRef().str();
        checkAndInsert(traceSpecificOpPattern, opName, 0);
      } else if (nodeNamePattern != "NONE") {
        StringAttr onnxNodeName =
            op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
        if (onnxNodeName && !onnxNodeName.getValue().empty()) {
          std::string nodeNameString = onnxNodeName.getValue().str();
          checkAndInsert(traceSpecificNodePattern, nodeNameString, 1);
        }
      }
    });
  }
};
} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createInstrumentONNXSignaturePass(
    const std::string pattern, const std::string nodePattern) {
  return std::make_unique<InstrumentONNXSignaturePass>(pattern, nodePattern);
}
