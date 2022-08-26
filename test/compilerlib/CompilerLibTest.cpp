/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace onnx_mlir;

std::string testFileName;
std::string outputBaseName;
bool compileFromFile = false;

#define IGNORE_ARG(FLAG)                                                       \
  if (arg.find(FLAG) == 0) {                                                   \
    return false;                                                              \
  }
#define PARSE_ARG(NAME, FLAG)                                                  \
  if (arg.find(FLAG) == 0) {                                                   \
    NAME = arg.substr(sizeof(FLAG));                                           \
    return true;                                                               \
  }
#define PARSE_FLAG(NAME, FLAG)                                                 \
  if (arg.find(FLAG) == 0) {                                                   \
    NAME = true;                                                               \
    return true;                                                               \
  }
#define PARSE_UNSUPPORTED_FLAG(FLAG)                                           \
  if (arg.find(FLAG) == 0) {                                                   \
    return true;                                                               \
  }

// Return 1 if arg used, 0 if unused.
bool readArg(const std::string &arg) {
  PARSE_ARG(outputBaseName, "-o");
  PARSE_FLAG(compileFromFile, "--fromfile");
  PARSE_UNSUPPORTED_FLAG("--EmitLib");
  IGNORE_ARG("-"); // Ignore all other options.
  testFileName = arg;
  return true;
}

// Read the arguments used by this program, and leave in argc/argv
// the unused arguments, which may then be processed by the
// ONNX-MLIR compiler.
void readCommandLineAndKeepUnused(int &argc, char *argv[]) {
  int num = argc;
  argc = 0;
  argv[argc++] = argv[0]; // Keep program name.
  for (int i = 1; i < num; ++i) {
    if (!readArg(std::string(argv[i]))) {
      argv[argc++] = argv[i];
    }
  }
}

int main(int argc, char *argv[]) {
  // Read the compiler options from env and args.
  readCommandLineAndKeepUnused(argc, argv);
  omSetCompilerOptionsFromArgsAndEnv(argc, argv, nullptr);

  if (outputBaseName == "") {
    outputBaseName = testFileName.substr(0, testFileName.find_last_of("."));
  }

  int retVal = 0;
  const char *errorMessage = NULL;
  const char *compiledFilename;
  if (compileFromFile) {
    retVal = omCompileFromFile(testFileName.c_str(), outputBaseName.c_str(),
        onnx_mlir::EmitLib, &compiledFilename, &errorMessage);
    if (retVal != CompilerSuccess && errorMessage != NULL)
      std::cerr << errorMessage;
  } else {
    std::ifstream inFile(
        testFileName, std::ios_base::in | std::ios_base::binary);
    std::string test((std::istreambuf_iterator<char>(inFile)),
        std::istreambuf_iterator<char>());
    retVal =
        omCompileFromArray(test.data(), test.size(), outputBaseName.c_str(),
            onnx_mlir::EmitLib, &compiledFilename, &errorMessage);
    if (retVal != CompilerSuccess && errorMessage != NULL) {
      std::cerr << errorMessage;
    }
  }
  if (retVal != 0) {
    std::cerr << "Compiling " << testFileName << "failed with code" << retVal;
  }
  return retVal;
}
