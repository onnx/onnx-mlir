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
std::string march;
std::string mtriple;
bool compileFromFile = false;
bool optO0 = false;
bool optO1 = false;
bool optO2 = false;
bool optO3 = false;

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
  PARSE_ARG(march, "--march=");
  PARSE_ARG(mtriple, "--mtriple=");
  PARSE_ARG(outputBaseName, "-o");
  PARSE_FLAG(optO0, "-O0");
  PARSE_FLAG(optO1, "-O1");
  PARSE_FLAG(optO2, "-O2");
  PARSE_FLAG(optO3, "-O3");
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

  // Build list of compiler option.
  OptionKind optionKey[4];
  const char *optionVal[4];
  int optionNum = 0;
  if (!mtriple.empty()) {
    optionKey[optionNum] = OptionKind::TargetTriple;
    optionVal[optionNum] = mtriple.c_str();
    optionNum++;
  }
  if (!march.empty()) {
    optionKey[optionNum] = OptionKind::TargetArch;
    optionVal[optionNum] = march.c_str();
    optionNum++;
  }
  if (!mcpu.empty()) {
    optionKey[optionNum] = OptionKind::TargetCPU;
    optionVal[optionNum] = mcpu.c_str();
    optionNum++;
  }
  if (optO0) {
    optionKey[optionNum] = OptionKind::CompilerOptLevel;
    optionVal[optionNum] = "0";
    optionNum++;
  } else if (optO1) {
    optionKey[optionNum] = OptionKind::CompilerOptLevel;
    optionVal[optionNum] = "1";
    optionNum++;
  } else if (optO2) {
    optionKey[optionNum] = OptionKind::CompilerOptLevel;
    optionVal[optionNum] = "2";
    optionNum++;
  } else if (optO3) {
    optionKey[optionNum] = OptionKind::CompilerOptLevel;
    optionVal[optionNum] = "3";
    optionNum++;
  }

  int retVal = 0;
  if (compileFromFile) {
    const char *errorMessage = NULL;
    retVal = omCompileFromFile(testFileName.c_str(), outputBaseName.c_str(),
        onnx_mlir::EmitLib, optionKey, optionVal, optionNum, &errorMessage);
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
            onnx_mlir::EmitLib, optionKey, optionVal, optionNum);
  }
  if (retVal != 0) {
    std::cerr << "Compiling " << testFileName << "failed with code" << retVal;
  }
  return retVal;
}
