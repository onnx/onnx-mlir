/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OptionUtils.hpp -------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing options.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_OPTION_UTILS_H
#define ONNX_MLIR_OPTION_UTILS_H

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "llvm/ADT/StringRef.h"

#include <map>
#include <set>
#include <string>

namespace onnx_mlir {

// Class that takes a list of comma separated regex expression to define a set
// of names that are "enabled" or not by a given compiler option. The
// "isEnabled(name)" function let a user determine if that name satisfies any of
// the regex or not. Class uses caching to reduce overheads.
//
// List of regex have the following properties.
// The presence of presence of "NONE" signifies that all names are disabled. The
// presence of "ALL" signifies that all names are enabled. A '.' char is treated
// as a regular char (aka "\."); and a '*' char is treated as a any string
// sequence (aka ".*").
class EnableByRegexOption {
public:
  EnableByRegexOption() = delete;
  // Constructor provides a string that is a comma separated list of regex.
  // These regex will determine which names are enabled or disabled by the
  // option. The emptyIsNone defines what to do when the provided string is
  // empty. If true, empty string means NONE; if false, empty string means ALL.
  EnableByRegexOption(
      bool emptyIsNone, std::string regexString = std::string());
  // Delayed initialization of the list of regex, permissible prior to a first
  // "isEnabled" query.
  void setRegexString(std::string regexString);

  // Returns true/false depending on wether that name matches any of the
  // regex.
  bool isEnabled(const std::string &name);
  bool isEnabled(const llvm::StringRef &name) { return isEnabled(name.str()); }

private:
  bool emptyIsNone; // If true, empty string is NONE; otherwise empty is ALL.
  bool allEnabled;  // Short-circuit test when all names are enabled.
  bool allDisabled; // Short-circuit test when all names are disabled.
  std::set<std::string> regexOfAllowedNames; // List of regex.
  std::map<std::string, bool> nameCache;     // Map of name -> enabled/disabled.
};

} // namespace onnx_mlir
#endif
