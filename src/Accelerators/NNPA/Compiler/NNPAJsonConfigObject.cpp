/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- NNPAJsonConfigObject.cpp - NNPA JSON Config ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements NNPA-specific JSON configuration functionality.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <regex>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Accelerators/NNPA/Compiler/NNPAJsonConfigObject.hpp"
#include "src/Compiler/JsonConfigObject.hpp"

using namespace llvm;
using namespace mlir;

namespace onnx_mlir {

// Accessor function to get the global config object with thread-safe
// initialization.
NNPAJsonConfigObject &getGlobalNNPAConfig() {
  static NNPAJsonConfigObject globalNNPAConfig;
  static std::once_flag initFlag;

  std::call_once(initFlag, []() {
    if (!globalNNPAConfig.isLoaded()) {
      JsonConfigObject *globalConfig = &getGlobalOMConfig();
      if (globalConfig->isLoaded())
        globalNNPAConfig.loadFromFile(getGlobalOMConfig().getFilePath());
    }
  });

  return globalNNPAConfig;
}

void NNPAJsonConfigObject::constructTensorInfo(
    Value v, json::Object &tensorInfoObj) {
  // Read info from value and put them into a JSON object as follows:
  // -1 is used for a dynamic dimension.
  // {
  //   "rank": 4,
  //   "type":  "f32"
  //   "dims": {
  //     0: "-1",
  //     1: "3",
  //     2: "5",
  //   },
  // },
  ShapedType tensorType = mlir::dyn_cast<ShapedType>(v.getType());
  if (!tensorType)
    return;

  // Tensor information
  ArrayRef<int64_t> dims = tensorType.getShape();
  // Rank
  tensorInfoObj["rank"] = std::to_string(dims.size());
  // Element type
  Type elemTy = tensorType.getElementType();
  std::string typeStr;
  llvm::raw_string_ostream(typeStr) << elemTy;
  tensorInfoObj["type"] = typeStr;
  // Dimension size
  json::Object dimObj;
  for (uint64_t i = 0; i < dims.size(); ++i) {
    int64_t d = dims[i];
    if (ShapedType::isDynamic(d))
      dimObj[std::to_string(i)] = "-1";
    else
      dimObj[std::to_string(i)] = std::to_string(d);
  }
  tensorInfoObj["dims"] = std::move(dimObj);
}

bool NNPAJsonConfigObject::matchNodeType(mlir::Operation *op, std::regex re) {
  std::string opName = op->getName().getStringRef().str();
  return std::regex_match(opName, re);
}

bool NNPAJsonConfigObject::matchNodeName(mlir::Operation *op, std::regex re) {
  if (auto nameAttr =
          op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
    std::string name = nameAttr.getValue().str();
    return std::regex_match(name, re);
  }
  return false;
}

// Helper function to check if an integer string value satisfies a constraint
// pattern.
//
// Pattern Syntax:
//   Comparison Operators:
//     "3"      - Exact match (implicit equality): value must equal 3
//     ">3"     - Greater than: value must be > 3
//     ">=3"    - Greater than or equal: value must be >= 3
//     "<3"     - Less than: value must be < 3
//     "<=3"    - Less than or equal: value must be <= 3
//     "==3"    - Explicit equality: value must equal 3
//     "!=3"    - Not equal: value must not equal 3
//
//   Modulo Operations (for divisibility/alignment checks):
//     "%32==0" - Modulo constraint: (value % 32) must equal 0
//     "%64==0" - Divisibility by 64: (value % 64) must equal 0
//     "%N==R"  - General form: (value % N) must equal R
//
// Parameters:
//   s   - String representation of integer value to test (e.g., "128", "-1")
//   reg - Pattern/constraint string (e.g., ">5", ">=10", "%32==0")
//
// Returns:
//   true  - The integer value in 's' satisfies the constraint in 'reg'
//   false - Constraint not satisfied, or error (empty strings, invalid
//   integers)
//
// Examples:
//   satisfiesIntegerConstraint("10", ">5")     -> true  (10 > 5)
//   satisfiesIntegerConstraint("64", "%32==0") -> true  (64 % 32 == 0)
//   satisfiesIntegerConstraint("5", ">=5")     -> true  (5 >= 5)
//   satisfiesIntegerConstraint("7", "7")       -> true  (7 == 7)
//   satisfiesIntegerConstraint("5", "!=3")     -> true  (5 != 3)
//   satisfiesIntegerConstraint("-1", "-1")     -> true  (-1 == -1, for dynamic
//   dims)
static bool satisfiesIntegerConstraint(
    const std::string &s, const std::string &reg) {
  // Handle empty strings - return false for invalid input.
  if (s.empty() || reg.empty())
    return false;

  try {
    // Convert string 's' to integer value for comparison.
    int64_t sValue = std::stoi(s);

    // Check for modulo operation first (pattern: "%N==R").
    // Example: "%32==0" checks if value is divisible by 32.
    if (reg[0] == '%') {
      size_t eqPos = reg.find("==");
      if (eqPos != std::string::npos) {
        // Extract divisor N from "%N==R" (between '%' and "==").
        int64_t modValue = std::stoi(reg.substr(1, eqPos - 1));
        // Extract expected remainder R from "%N==R" (after "==").
        int64_t expectedRemainder = std::stoi(reg.substr(eqPos + 2));
        // Check if (sValue % modValue) equals expectedRemainder.
        return (sValue % modValue) == expectedRemainder;
      }
      return false; // Malformed modulo pattern.
    }

    // Parse comparison operator and extract numeric value.
    std::string op;
    int64_t regValue;

    // Check two-character operators FIRST (to avoid misinterpreting ">=" as
    // ">").
    if (reg.size() >= 2 && reg.substr(0, 2) == ">=") {
      op = ">=";
      regValue = std::stoi(reg.substr(2)); // Extract number after ">=".
    } else if (reg.size() >= 2 && reg.substr(0, 2) == "<=") {
      op = "<=";
      regValue = std::stoi(reg.substr(2)); // Extract number after "<=".
    } else if (reg.size() >= 2 && reg.substr(0, 2) == "==") {
      op = "==";
      regValue = std::stoi(reg.substr(2)); // Extract number after "==".
    } else if (reg.size() >= 2 && reg.substr(0, 2) == "!=") {
      op = "!=";
      regValue = std::stoi(reg.substr(2)); // Extract number after "!=".
    }
    // Check single-character operators.
    else if (reg[0] == '>') {
      op = ">";
      regValue = std::stoi(reg.substr(1)); // Extract number after ">".
    } else if (reg[0] == '<') {
      op = "<";
      regValue = std::stoi(reg.substr(1)); // Extract number after "<".
    }
    // No operator detected - treat as implicit equality.
    else {
      op = "==";
      regValue = std::stoi(reg); // Entire string is the number.
    }

    // Perform the appropriate comparison based on the operator.
    if (op == ">")
      return sValue > regValue;
    else if (op == ">=")
      return sValue >= regValue;
    else if (op == "<")
      return sValue < regValue;
    else if (op == "<=")
      return sValue <= regValue;
    else if (op == "!=")
      return sValue != regValue;
    else // op == "=="
      return sValue == regValue;

  } catch (const std::exception &e) {
    // Handle any conversion errors (invalid integer format) - return false.
    return false;
  }
}

bool NNPAJsonConfigObject::matchTensorInfo(
    Value tensor, json::Object *patternObj) {
  // clang-format off
  // patternObj format
  //   {
  //     "rank": "4", "type":  "f32", "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"}
  //   },
  // clang-format on

  // Construct a target json object from the tensor.
  json::Object targetObj;
  constructTensorInfo(tensor, targetObj);

  // Match the target object against the pattern object.
  bool matched = true;
  for (const auto &kv : *patternObj) {
    StringRef k = kv.first;
    if (matched && k.equals_insensitive("rank")) {
      matched = satisfiesIntegerConstraint(
          targetObj.getString(k)->str(), patternObj->getString(k)->str());
    }
    if (matched && k.equals_insensitive("type")) {
      matched = (targetObj.getString(k) == patternObj->getString(k));
    }
    if (matched && k.equals_insensitive("dims")) {
      // Match dimension constraints.
      json::Object *targetDims = targetObj.getObject(k);
      json::Object *regDims = patternObj->getObject(k);
      int64_t rank = std::stoi(targetObj.getString("rank")->str());
      if (targetDims && regDims) {
        for (const auto &dimKv : *regDims) {
          StringRef dimStrRef = dimKv.first;
          int64_t dimIdx = std::stoi(dimStrRef.str());
          std::optional<StringRef> regDimVal = regDims->getString(dimStrRef);
          // dimIdx can be < 0.
          if (dimIdx < 0)
            dimIdx += rank;
          std::optional<StringRef> targetDimVal =
              targetDims->getString(std::to_string(dimIdx));
          if (targetDimVal && regDimVal) {
            matched = satisfiesIntegerConstraint(
                targetDimVal->str(), regDimVal->str());
          } else {
            matched = false;
          }
          if (!matched)
            break;
        }
      }
    }
    if (!matched)
      break;
  }

  return matched;
}

bool NNPAJsonConfigObject::matchTensorInfo(
    ValueRange tensors, json::Object *patternObj) {
  // clang-format off
  // patternObj format
  // {
  //   "0": { "rank": "4", "type":  "f32" "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"} },
  //   "1": { "rank": "4", "type":  "f32" "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"} },
  // }
  // clang-format on

  int64_t numValues = tensors.size();

  bool matched = true;
  for (const auto &kv : *patternObj) {
    StringRef k = kv.first;
    json::Object *v = patternObj->getObject(k);
    int id = std::stoi(k.str());
    if (id < 0)
      id += numValues;
    if (id < 0 || id >= numValues) {
      matched = false;
      break;
    }
    if (!matchTensorInfo(tensors[id], v)) {
      matched = false;
      break;
    }
  }

  return matched;
}

void NNPAJsonConfigObject::applyConfigToOps(
    llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<void(json::Object *, mlir::Operation *)> updateAttrFn) {
  if (empty())
    return;

  // Get the nnpa_ops_config array.
  json::Array *opConfigsArr = getArray(OPS_CONFIG_KEY);
  if (!opConfigsArr || opConfigsArr->empty())
    return;

  // Collect operations to work on.
  llvm::DenseSet<mlir::Operation *> workingOps(ops.begin(), ops.end());

  // Process each configuration rule in the nnpa_ops_config array.
  for (json::Value &v : *opConfigsArr) {
    json::Object *configObj = v.getAsObject();
    if (!configObj)
      continue;

    // Get the pattern object.
    json::Object *patternObj = configObj->getObject(PATTERN_KEY);
    if (!patternObj)
      continue;

    // Get the match and rewrite objects.
    json::Object *matchObj = patternObj->getObject(MATCH_KEY);
    json::Object *rewriteObj = patternObj->getObject(REWRITE_KEY);
    if (!matchObj || !rewriteObj)
      continue;

    // Extract matching criteria.
    std::optional<StringRef> nodeTypeStr = matchObj->getString(NODE_TYPE_KEY);
    std::optional<StringRef> onnxNodeNameStr =
        matchObj->getString(ONNX_NODE_NAME_KEY);
    json::Object *inputPatternObj = matchObj->getObject(INPUTS_KEY);
    json::Object *outputPatternObj = matchObj->getObject(OUTPUTS_KEY);

    if (!nodeTypeStr) {
      llvm::errs()
          << "Warning: Config entry missing required 'node_type' field\n";
      continue;
    }

    // Create regex patterns for matching with exception handling.
    std::regex nodeTypeRegex;
    std::regex onnxNodeNameRegex;
    bool hasNodeNamePattern = false;
    try {
      nodeTypeRegex = std::regex(nodeTypeStr->str());
      if (onnxNodeNameStr) {
        onnxNodeNameRegex = std::regex(onnxNodeNameStr->str());
        hasNodeNamePattern = true;
      }
    } catch (const std::regex_error &e) {
      llvm::errs() << "Error: Invalid regex pattern in config - " << e.what()
                   << "\n";
      if (onnxNodeNameStr) {
        llvm::errs() << "  node_type: " << nodeTypeStr->str()
                     << ", onnx_node_name: " << onnxNodeNameStr->str() << "\n";
      } else {
        llvm::errs() << "  node_type: " << nodeTypeStr->str() << "\n";
      }
      continue;
    }

    // Find matching operations and apply rewrite.
    llvm::SmallVector<mlir::Operation *> matchedOps;
    for (mlir::Operation *op : workingOps) {
      // Check node type.
      if (!matchNodeType(op, nodeTypeRegex))
        continue;

      // Check onnx_node_name if specified.
      if (hasNodeNamePattern && !matchNodeName(op, onnxNodeNameRegex))
        continue;

      // Check the tensor information.
      ValueRange inputTensors = ValueRange(op->getOperands());
      if (inputPatternObj && !matchTensorInfo(inputTensors, inputPatternObj))
        continue;
      ValueRange outputTensors = ValueRange(op->getResults());
      if (outputPatternObj && !matchTensorInfo(outputTensors, outputPatternObj))
        continue;

      // Operation matches - apply rewrite.
      updateAttrFn(rewriteObj, op);
      matchedOps.push_back(op);
    }

    // Remove matched operations from working set (first match wins).
    for (mlir::Operation *op : matchedOps) {
      workingOps.erase(op);
    }

    // Stop if no more operations to process.
    if (workingOps.empty())
      break;
  }
}

void NNPAJsonConfigObject::writeOpsConfig(llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<bool(mlir::Operation *, json::Object &rewrite)>
        buildConfigFn) {
  json::Array opConfigsArray;

  for (mlir::Operation *op : ops) {
    json::Object match;
    json::Object rewrite;

    // Get the operation type.
    std::string nodeType = op->getName().getStringRef().str();
    match[NODE_TYPE_KEY] = nodeType;

    // Get the onnx_node_name if present.
    if (auto nameAttr =
            op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
      match[ONNX_NODE_NAME_KEY] = nameAttr.getValue().str();
    }

    // Get the tensor info from inputs and outputs.
    if (op->getOperands().size() > 0) {
      json::Object inputs;
      for (uint64_t i = 0; i < op->getOperands().size(); ++i) {
        json::Object tensorInfo;
        constructTensorInfo(op->getOperands()[i], tensorInfo);
        inputs[std::to_string(i)] = std::move(tensorInfo);
      }
      match[INPUTS_KEY] = std::move(inputs);
    }
    if (op->getResults().size() > 0) {
      json::Object outputs;
      for (uint64_t i = 0; i < op->getResults().size(); ++i) {
        json::Object tensorInfo;
        constructTensorInfo(op->getResults()[i], tensorInfo);
        outputs[std::to_string(i)] = std::move(tensorInfo);
      }
      match[OUTPUTS_KEY] = std::move(outputs);
    }

    // Let the callback build the rewrite object.
    if (!buildConfigFn(op, rewrite))
      continue;

    // Build the pattern object.
    json::Object pattern;
    pattern[MATCH_KEY] = std::move(match);
    pattern[REWRITE_KEY] = std::move(rewrite);

    // Build the config object.
    json::Object config;
    config[PATTERN_KEY] = std::move(pattern);

    opConfigsArray.push_back(std::move(config));
  }

  // Store the nnpa_ops_config array in the JSON object.
  json::Object *jsonObj = getJsonObject();
  if (jsonObj) {
    (*jsonObj)[OPS_CONFIG_KEY] = std::move(opConfigsArray);
  }
}

} // namespace onnx_mlir
