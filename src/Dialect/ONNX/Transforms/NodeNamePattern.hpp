/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- NodeNamePattern.hpp - Node name/IO selector parsing ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Shared parser for the "REGEX[:inN+outN]" node-name pattern syntax used by
// both --instrument-onnx-node (InstrumentONNXSignaturePass, prints matched
// tensors) and --instrument-onnx-node-return (AppendInstrumentedOutputsPass,
// returns matched tensors as extra model outputs). Kept dependency-free of
// either pass so the two flags share one implementation of the syntax.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cctype>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

namespace onnx_mlir {

// Parsed representation of one comma-separated entry of the
// --instrument-onnx-node/--instrument-onnx-node-return option: a node-name
// pattern plus an optional operand/result index filter. Syntax:
// "pattern[:selector(+selector)*]", where each selector is "inN" or "outN"
// (0-based). When no ":" suffix is given, hasIOFilter is false and every
// operand/result of a matched node is selected, preserving the flags'
// original (pre-filter) behavior.
struct NodeIOEntry {
  std::regex nameRegex;
  bool hasIOFilter = false;
  std::set<int64_t> inputIdx;
  std::set<int64_t> outputIdx;
};

inline bool isAllDigits(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(),
                           [](unsigned char c) { return std::isdigit(c); });
}

// Parse a --instrument-onnx-node/--instrument-onnx-node-return option string
// into a list of NodeIOEntry. The '.'/'*' literal-vs-regex convention
// matches the rest of onnx-mlir's instrument options (EnableByRegexOption),
// applied per entry here rather than to the whole option string, so an
// io-filter suffix on one entry cannot suppress '*' expansion on another.
inline std::vector<NodeIOEntry> parseNodeNamePattern(const std::string &opt) {
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

    // Node names often contain literal dots (e.g. "layer.0"), so treat the
    // pattern as a glob-style literal name by default: escape '.' and
    // expand '*' to ".*". If it already looks like deliberate regex (has
    // ".*", "\.", "^", "$", "[", "+", or "?"), use it verbatim instead.
    // e.g. "onnx.Add_0" -> "onnx\.Add_0" (matches only that exact name);
    //      "onnx.Add_*" -> "onnx\.Add_.*" (matches any Add_<suffix>);
    //      "/layer\.0/MatMul" is left as-is, since it already has "\.".
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
          llvm::errs() << "Warning: ignoring malformed node-name-pattern"
                       << " selector \"" << sel
                       << "\" (expected inN or outN)\n";
        }
      }
    }
    entries.emplace_back(std::move(entry));
  }
  return entries;
}

} // namespace onnx_mlir
