/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompilerDriver.cpp - compiler driver  -----------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx files using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#include "src/Compiler/OMCompilerDriver.hpp"

#include <cstdlib>
#include <filesystem>

#include "src/Compiler/Command.hpp"
#include "src/Compiler/DriverUtils.hpp"

using onnx_mlir_compiler_driver;
namespace fs = std::filesystem;

extern int64_t omCompile(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename,
    std::string &errorMessage) {

  std::string onnxMlirPath = "onnx-mlir";
  onnx_mlir::Command compile(onnxMlirPath, /*verbose*/ true);
  std::vector<std::string> flagVect = onnx_mlir::parseFlags(flags);
  compile.appendList(flagVect);
  compile.appendStr(inputFilename);
  compile.print();
  int rc = compile.exec();
  if (rc != 0) {
    errorMessage = "Compiler failed with error code " + std::to_string(rc);
    outputFilename = "";
    return rc;
  }
  errorMessage = "";
  outputFilename = onnx_mlir::getOutputFilename(inputFilename, flagVect);
  return 0;
}

#if 0
// clang++ ../src/Compiler/OMCompilerDriver.cpp ../src/Compiler/Command.cpp -I
// /Users/alexe/OM/onnx-mlir  -o alextest
int main() {
  std::string outputFilename, errorMessage;
  std::string model = "add1.mlir";
  int64_t results = omCompile(
      model, "-O3 -march=arm64  -o=bibi", outputFilename, errorMessage);
  fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);
  results = omCompile(
      model, "-O3 -march=arm64  -o bobo/bibi", outputFilename, errorMessage);
        fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);
  results = omCompile(
      model, "-O3 -march=arm64  ", outputFilename, errorMessage);
        fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);

  return results;
}
#endif
