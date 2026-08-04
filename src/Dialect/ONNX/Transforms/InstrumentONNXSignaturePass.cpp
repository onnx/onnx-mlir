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

#include <algorithm>
#include <cctype>
#include <regex>
#include <set>
#include <sstream>

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

// Parsed representation of one comma-separated entry of the
// --instrument-onnx-node option: a node-name pattern plus an optional
// operand/result index filter. Syntax: "pattern[:selector(+selector)*]",
// where each selector is "inN" or "outN" (0-based). When no ":" suffix is
// given, hasIOFilter is false and every operand/result of a matched node is
// printed, preserving the flag's original (pre-filter) behavior.
struct NodeIOEntry {
  std::regex nameRegex;
  bool hasIOFilter = false;
  std::set<int64_t> inputIdx;
  std::set<int64_t> outputIdx;
};

static bool isAllDigits(const std::string &s) {
  return !s.empty() &&
         std::all_of(s.begin(), s.end(), [](unsigned char c) {
           return std::isdigit(c);
         });
}

// Parse the --instrument-onnx-node option string into a list of NodeIOEntry.
// The '.'/'*' literal-vs-regex convention matches the rest of onnx-mlir's
// instrument options (EnableByRegexOption), applied per entry here rather
// than to the whole option string, so an io-filter suffix on one entry
// cannot suppress '*' expansion on another.
static std::vector<NodeIOEntry> parseNodeNamePattern(const std::string &opt) {
  std::vector<NodeIOEntry> entries;
  if (opt.empty() || opt == "NONE")
    return entries;
  std::stringstream ss(opt);
  std::string token;
  while (std::getline(ss, token, ',')) {
    size_t b = token.find_first_not_of(" \t");
    size_t e = token.find_last_not_of(" \t");
    if (b == std::string::npos)
      continue;
    token = token.substr(b, e - b + 1);

    std::string namePart = token;
    std::string ioPart;
    size_t colon = token.find(':');
    if (colon != std::string::npos) {
      namePart = token.substr(0, colon);
      ioPart = token.substr(colon + 1);
    }

    bool hasRegexPattern = namePart.find(".*") != std::string::npos ||
                           namePart.find("\\.") != std::string::npos ||
                           namePart.find("^") != std::string::npos ||
                           namePart.find("$") != std::string::npos ||
                           namePart.find("[") != std::string::npos ||
                           namePart.find("+") != std::string::npos ||
                           namePart.find("?") != std::string::npos;
    if (!hasRegexPattern) {
      namePart = std::regex_replace(namePart, std::regex("\\."), "\\.");
      namePart = std::regex_replace(namePart, std::regex("\\*"), ".*");
    }

    NodeIOEntry entry;
    entry.nameRegex = std::regex(namePart);
    if (!ioPart.empty()) {
      entry.hasIOFilter = true;
      std::stringstream ioss(ioPart);
      std::string sel;
      while (std::getline(ioss, sel, '+')) {
        if (sel.compare(0, 2, "in") == 0 && isAllDigits(sel.substr(2))) {
          entry.inputIdx.insert(std::stoll(sel.substr(2)));
        } else if (sel.compare(0, 3, "out") == 0 &&
                   isAllDigits(sel.substr(3))) {
          entry.outputIdx.insert(std::stoll(sel.substr(3)));
        } else {
          llvm::errs() << "Warning: ignoring malformed --instrument-onnx-node"
                        << " selector \"" << sel
                        << "\" (expected inN or outN)\n";
        }
      }
    }
    entries.emplace_back(std::move(entry));
  }
  return entries;
}

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
    traceSpecificOpPattern.setRegexString(signaturePattern);
    std::vector<NodeIOEntry> nodeNameEntries =
        parseNodeNamePattern(nodeNamePattern);

    // Insert an ONNXPrintSignatureOp for a node matched by name, printing
    // only the operand/result indices selected by entry's io filter (or all
    // of them, if the entry has no filter).
    auto insertNodeSignature = [&](mlir::Operation *op,
                                   const NodeIOEntry &entry,
                                   const std::string &opName) {
      OpBuilder builder(op);
      std::string nodeName = onnx_mlir::getNodeNameInPresenceOfOpt(op);
      std::string fullName = opName + ", " + nodeName;
      StringAttr fullNameAttr = builder.getStringAttr(fullName);
      llvm::SmallVector<Value, 6> operAndRes;
      // Per-value label (e.g. "in0", "out1"), kept in lockstep with
      // operAndRes so a caller-selected subset can still be told apart.
      llvm::SmallVector<Attribute, 6> ioLabels;
      if (!entry.hasIOFilter) {
        for (auto it : llvm::enumerate(op->getOperands())) {
          operAndRes.emplace_back(it.value());
          ioLabels.emplace_back(
              builder.getStringAttr("in" + std::to_string(it.index())));
        }
        for (auto it : llvm::enumerate(op->getResults())) {
          operAndRes.emplace_back(it.value());
          ioLabels.emplace_back(
              builder.getStringAttr("out" + std::to_string(it.index())));
        }
      } else {
        OperandRange operands = op->getOperands();
        for (int64_t idx : entry.inputIdx) {
          if (idx >= 0 && (size_t)idx < operands.size()) {
            operAndRes.emplace_back(operands[idx]);
            ioLabels.emplace_back(
                builder.getStringAttr("in" + std::to_string(idx)));
          } else
            llvm::errs() << "Warning: --instrument-onnx-node selector in"
                          << idx << " out of range for node \"" << nodeName
                          << "\" (" << operands.size()
                          << " operand(s)); ignoring.\n";
        }
        ResultRange results = op->getResults();
        for (int64_t idx : entry.outputIdx) {
          if (idx >= 0 && (size_t)idx < results.size()) {
            operAndRes.emplace_back(results[idx]);
            ioLabels.emplace_back(
                builder.getStringAttr("out" + std::to_string(idx)));
          } else
            llvm::errs() << "Warning: --instrument-onnx-node selector out"
                          << idx << " out of range for node \"" << nodeName
                          << "\" (" << results.size()
                          << " result(s)); ignoring.\n";
        }
      }
      builder.setInsertionPointAfter(op);
      ONNXPrintSignatureOp::create(builder, op->getLoc(), fullNameAttr,
          /*detail=*/1, builder.getArrayAttr(ioLabels), operAndRes);
    };

    // Pre-order walk so we can skip ONNXFusedOp bodies with WalkResult::skip().
    getOperation().walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation *op) -> WalkResult {
          auto dialect = op->getDialect();
          Location loc = op->getLoc();
          // Define a lambda function to check whether the node is selected by
          // its op name or node name, and if yes, insert ONNXSignatureOp.
          // displayName overrides the op-name component in the printed header.
          auto checkAndInsert = [&](onnx_mlir::EnableByRegexOption &pattern,
                                    std::string matchString, int detail,
                                    std::string displayName = "") -> bool {
            if (pattern.isEnabled(matchString)) {
              // Add signature printing op.
              OpBuilder builder(op);
              if (displayName.empty())
                displayName = op->getName().getStringRef().str();
              std::string nodeName = onnx_mlir::getNodeNameInPresenceOfOpt(op);
              std::string fullName = displayName + ", " + nodeName;
              StringAttr fullNameAttr = builder.getStringAttr(fullName);
              // Enqueue all input operands, and then the results.
              llvm::SmallVector<Value, 6> operAndRes(op->getOperands());
              for (Value res : op->getResults())
                operAndRes.emplace_back(res);
              // Since we may use the result of an operation, we must insert the
              // print operation after the operation.
              builder.setInsertionPointAfter(op);
              // When one node is selected, print the details of the tensor.
              // No io_labels here: this path (op-type pattern match) keeps
              // its original, unlabeled output format.
              ONNXPrintSignatureOp::create(builder, loc, fullNameAttr, detail,
                  builder.getStrArrayAttr({}), operAndRes);
              return true;
            }
            return false;
          };
          if (isa<func::FuncDialect>(dialect) ||
              isa<ONNXPrintSignatureOp, KrnlInstrumentOp>(op)) {
            // Always skip function dialects (such as function call/return), as
            // well as ONNX instrument operations.
            return WalkResult::advance();
          }

          // getProfilingName returns "onnx.fused.<kind>" for ONNXFusedOp and
          // the dialect op-name for every other op, so it drives both the
          // match string and the display name uniformly.
          std::string opName = onnx_mlir::getProfilingName(op);
          bool gotOne = false;
          if (!nodeNameEntries.empty()) {
            StringAttr onnxNodeName =
                op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
            if (onnxNodeName && !onnxNodeName.getValue().empty()) {
              std::string name = onnxNodeName.getValue().str();
              for (const NodeIOEntry &entry : nodeNameEntries) {
                if (std::regex_match(name, entry.nameRegex)) {
                  insertNodeSignature(op, entry, opName);
                  gotOne = true;
                  break;
                }
              }
            }
          }
          if (!gotOne && signaturePattern != "NONE" && signaturePattern != "") {
            checkAndInsert(traceSpecificOpPattern, opName, 0, opName);
          }
          // Skip the body of ONNXFusedOp — its inner ops are not individually
          // profiled; the fused op itself was already reported above.
          return isa<ONNXFusedOp>(op) ? WalkResult::skip()
                                      : WalkResult::advance();
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
