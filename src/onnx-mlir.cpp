/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ onnx-mlir.cpp - Compiler Driver  ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
// Main function for onnx-mlir.
// Implements main for onnx-mlir driver.
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Version/Version.hpp"
#include <OnnxMlirCompiler.h>
#include <errno.h>
#include <iostream>

using namespace std;
using namespace onnx_mlir;

int main(int argc, char *argv[]) {

  const char *errorMessage = NULL;
  std::string commandLineStr = "";
  for (int i = 1; i < argc; i++) {
    commandLineStr.append(std::string(argv[i]).append(" "));
  }
  const char *flags = commandLineStr.c_str();
  int rc = onnx_mlir::omCompileFromFileViaCommand(flags, &errorMessage);
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to compile options with error code " << rc << "."
              << std::endl;
    if (errorMessage) {
      std::cerr << errorMessage << std::endl;
    }
    return rc;
  }
  return 0;
}