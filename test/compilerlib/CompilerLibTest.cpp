/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace onnx_mlir;

std::string testFileName;
std::string outputBaseName;
std::string mcpu;
std::string mtriple;
bool compileFromFile = false;

#define PARSE_ARG(NAME, FLAG)                                                  \
  if (arg.find(FLAG) == 0) {                                                   \
    NAME = arg.substr(sizeof(FLAG));                                           \
    return;                                                                    \
  }

#define PARSE_FLAG(NAME, FLAG)                                                 \
  if (arg.find(FLAG) == 0) {                                                   \
    NAME = true;                                                               \
    return;                                                                    \
  }

void readArg(const std::string &arg) {
  PARSE_ARG(mcpu, "--mcpu=");
  PARSE_ARG(mtriple, "--mtriple=");
  PARSE_ARG(outputBaseName, "-o");
  PARSE_FLAG(compileFromFile, "--fromfile");
  testFileName = arg;
}

void readCommandLine(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    readArg(std::string(argv[i]));
  }
}

int main(int argc, char *argv[]) {
  readCommandLine(argc, argv);

  if (outputBaseName == "") {
    outputBaseName = testFileName.substr(0, testFileName.find_last_of("."));
  }
  int retVal = 0;
  if (compileFromFile) {
    const char *errorMessage = NULL;
    retVal = omCompileFromFile(testFileName.c_str(), outputBaseName.c_str(),
        onnx_mlir::EmitLib, mcpu.empty() ? nullptr : mcpu.c_str(),
        mtriple.empty() ? nullptr : mtriple.c_str(), &errorMessage);
    if (errorMessage != NULL) {
      std::cerr << errorMessage;
      retVal = 0xf;
    }
  } else {
    std::ifstream inFile(
        testFileName, std::ios_base::in | std::ios_base::binary);
    std::string test((std::istreambuf_iterator<char>(inFile)),
        std::istreambuf_iterator<char>());
    retVal =
        omCompileFromArray(test.data(), test.size(), outputBaseName.c_str(),
            onnx_mlir::EmitLib, mcpu.empty() ? nullptr : mcpu.c_str(),
            mtriple.empty() ? nullptr : mtriple.c_str());
  }
  if (retVal != 0) {
    std::cerr << "Compiling " << testFileName << "failed with code" << retVal;
  }
  return retVal;
}
