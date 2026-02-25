/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Compiler/OMCompilerSession.hpp"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace onnx_mlir;

std::string flags;

// Read the arguments from command line and save them as a std::string which may
// be processed by the ONNX-MLIR compiler.
void readArgsFromCommandLine(int argc, char *argv[]) {
  for (int i = 1; i < argc; i++) {
    flags.append(std::string(argv[i]) + " ");
  }
}

int main(int argc, char *argv[]) {

  int retVal;
  readArgsFromCommandLine(argc, argv);

  // Compile.
  onnx_mlir::CompilerSession compilerSession;
  try {
    compilerSession.compile("" /*input model name included in the flags*/, flags);
    retVal = 0;
  } catch (const onnx_mlir::CompilerSessionException &error) {
    std::cerr << "error during compiler session: " << error.what() << std::endl;
    retVal = 1;
  }

  return retVal;
}
