/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OptionUtils.cpp -------------------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing options.
//
//===----------------------------------------------------------------------===//

#include "OptionUtils.hpp"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using namespace onnx_mlir;

namespace onnx_mlir {

// =============================================================================
// Support for enabled ops by regex option
// =============================================================================

EnableByRegexOption::EnableByRegexOption(
    bool emptyIsNone, std::string regexString)
    : emptyIsNone(emptyIsNone) {
  setRegexString(regexString);
}

void EnableByRegexOption::setRegexString(std::string regexString) {
  assert(nameCache.empty() && "can set regex string only before any queries");
  allEnabled = allDisabled = false;
  if (regexString.empty()) {
    if (emptyIsNone)
      allDisabled = true;
    else
      allEnabled = true;
  } else {
    if (regexString.find("NONE") != std::string::npos)
      allDisabled = true;
    if (regexString.find("ALL") != std::string::npos)
      allEnabled = true;
  }
  assert(!(allDisabled && allEnabled) && "cannot have both ALL and NONE");
  if (allDisabled || allEnabled)
    // No need to scan regexs.
    return;

  // We have a finite list of regex, preprocess them now. Lifted from the
  // InstrumentPass.cpp original implementation.
  // Separate multiple expressions with space.
  regexString = std::regex_replace(regexString, std::regex(","), " ");
  // The '.' character in regex string is recognized as normal character, not
  // regular expression.
  regexString = std::regex_replace(regexString, std::regex("\\."), "\\.");
  // The '*' character in regex string is recognized as '.*' pattern.
  regexString = std::regex_replace(regexString, std::regex("\\*"), ".*");
  std::stringstream ss(regexString);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  // Create the set of regex defined by the original regexString.
  regexOfAllowedNames = std::set<std::string>(begin, end);
}

bool EnableByRegexOption::isEnabled(const std::string &name) {
  if (allEnabled)
    return true;
  if (allDisabled)
    return false;
  // Now check if we have already seen this op.
  std::map<std::string, bool>::iterator it = nameCache.find(name);
  if (it != nameCache.end()) {
    return it->second;
  }

  // We have not seen this op, then test using the regex and cache answer.
  for (auto itr = regexOfAllowedNames.begin(); itr != regexOfAllowedNames.end();
       ++itr) {
    std::regex re(*itr);
    if (std::regex_match(name, re)) {
      // We have a match, cache and return true.
      nameCache[name] = true;
      return true;
    }
  }
  // We did not find a match; cache and return false.
  nameCache[name] = false;
  return false;
}

} // namespace onnx_mlir
