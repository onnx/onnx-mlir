/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DriverUtils.cpp - Utils for compiler driver  -------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#include "src/Compiler/DriverUtils.hpp"

#include <sstream>

using namespace onnx_mlir;

namespace onnx_mlir {

std::vector<std::string> parseFlags(const std::string &flags) {
  std::vector<std::string> flagVect;
  std::istringstream iss(flags);
  std::string arg;
  while (iss >> arg) {
    flagVect.push_back(arg);
  }
  return flagVect;
}

static std::string getOutputBasenameFromFlags(const std::string &inputFilename,
    const std::vector<std::string> &flagVect) {
  // Get output file name from the flags.
  std::string outputBasename;
  int num = flagVect.size();
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-o=", 0, 3) == 0) {
      if (flagVect[i].length() > 3) {
        outputBasename = flagVect[i].substr(3);
        break;
      } else
        fprintf(
            stderr, "Parsing `-o=` option, expected a name. Use default.\n");
    } else if (flagVect[i].find("-o") == 0) {
      if (i < num - 1) {
        outputBasename = flagVect[i + 1];
        break;
      } else {
        fprintf(stderr, "Parsing `-o` option, expected a name. Use default.\n");
      }
    }
  }
  // If no output file name, derive it from input file basename
  if (outputBasename.empty())
    outputBasename = inputFilename.substr(0, inputFilename.find_last_of("."));
  return outputBasename;
}

static EmissionTargetType getEmissionTargetFromFlags(
    const std::string &inputFilename,
    const std::vector<std::string> &flagVect) {
  // Get Emit target (approximate, enough to get output name). There are many
  // more Emit target than in the base definition because Accelerators may
  // have their own. That is why the Emit target has to be part of the flags
  // and cannot be a direct enum, as there is none that encompass all the
  // possible options.
  int num = flagVect.size();
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-Emit") == 0 || flagVect[i].find("--Emit") == 0) {
      if (flagVect[i].find("Lib") <= 6)
        return EmissionTargetType::EmitLib;
      else if (flagVect[i].find("JNI") <= 6)
        return EmissionTargetType::EmitJNI;
      else if (flagVect[i].find("Obj") <= 6)
        return EmissionTargetType::EmitObj;
      else // There are many other targets, all of the MLIR type.
        return EmissionTargetType::EmitMLIR;
    }
  }
  // Default is EmitLib.
  return EmissionTargetType::EmitLib;
}

std::string getTargetFilename(
    const std::string &outputBasename, EmissionTargetType target) {
  switch (target) {

#ifdef _WIN32
  case EmitObj:
    return outputBasename + ".obj";
  case EmitLib:
    return outputBasename + ".dll";
#else
  case EmitObj:
    return outputBasename + ".o";
  case EmitLib:
    return outputBasename + ".so";
#endif

  case EmitJNI:
    return outputBasename + ".jar";
  default:
    return outputBasename + ".onnx.mlir";
  }
}

std::string getOutputFilename(const std::string &inputFilename,
    const std::vector<std::string> &flagVect) {
  std::string outputBasename =
      getOutputBasenameFromFlags(inputFilename, flagVect);
  EmissionTargetType targetType =
      getEmissionTargetFromFlags(inputFilename, flagVect);
  return getTargetFilename(outputBasename, targetType);
}

} // namespace onnx_mlir
