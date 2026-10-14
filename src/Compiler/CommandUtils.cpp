/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- CommandUtils.cpp - Utils for compiler driver -------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#include "src/Compiler/CommandUtils.hpp"

#include <filesystem>
#include <sstream>
namespace fs = std::filesystem;

using namespace onnx_mlir;

namespace onnx_mlir {

std::vector<std::string> parseFlags(const std::string &flags) {
  std::vector<std::string> flagVect;
  std::string current;
  bool inDoubleQuote = false;
  bool inSingleQuote = false;
  bool escapeNext = false;

  for (size_t i = 0; i < flags.length(); ++i) {
    char c = flags[i];

    if (escapeNext) {
      // Add the escaped character literally
      current += c;
      escapeNext = false;
      continue;
    }

    if (c == '\\') {
      // Escape the next character
      escapeNext = true;
      continue;
    }

    if (c == '"' && !inSingleQuote) {
      // Toggle double quote mode
      inDoubleQuote = !inDoubleQuote;
      continue;
    }

    if (c == '\'' && !inDoubleQuote) {
      // Toggle single quote mode
      inSingleQuote = !inSingleQuote;
      continue;
    }

    if (std::isspace(c) && !inDoubleQuote && !inSingleQuote) {
      // Space outside quotes - end current token
      if (!current.empty()) {
        flagVect.push_back(current);
        current.clear();
      }
      continue;
    }

    // Regular character - add to current token
    current += c;
  }

  // Add the last token if any
  if (!current.empty()) {
    flagVect.push_back(current);
  }

  return flagVect;
}

static std::string getOutputBasenameFromFlags(
    const std::vector<std::string> &flagVect) {
  // Get output file name from the flags.
  int num = flagVect.size();
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-o=", 0, 3) == 0) {
      if (flagVect[i].length() > 3) {
        return flagVect[i].substr(3);
      } else
        fprintf(
            stderr, "Parsing `-o=` option, expected a name. Use default.\n");
    } else if (flagVect[i] == "-o") {
      if (i < num - 1) {
        return flagVect[i + 1];
        break;
      } else {
        fprintf(stderr, "Parsing `-o` option, expected a name. Use default.\n");
      }
    }
  }
  return "";
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

std::string getAbsolutePathUsingCurrentDir(const std::string &filename) {
  if (filename.empty())
    return filename;
  fs::path currFilename(filename);
  if (!currFilename.is_relative())
    return filename;
  // Has relative path, add current working dir.
  fs::path curWdir = fs::current_path();
  fs::path newFilename = curWdir / currFilename;
  return newFilename.string();
}

std::string getInputFilename(
    const std::string &inputFilename, const std::vector<std::string> &flags) {
  if (!inputFilename.empty())
    return inputFilename;
  for (size_t i = 0; i < flags.size(); ++i) {
    const std::string &arg = flags[i];
    if (!arg.empty() && arg[0] == '-')
      continue;
    // Check if it ends with ".onnx, mlir, or onnxtext".
    if ((arg.length() >= 5 && arg.substr(arg.length() - 5) == ".mlir") ||
        (arg.length() >= 5 && arg.substr(arg.length() - 5) == ".onnx") ||
        (arg.length() >= 9 && arg.substr(arg.length() - 9) == ".onnxtext"))
      return arg;
  }
  // Not found, return empty.
  return "";
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
  std::string outputBasename = getOutputBasenameFromFlags(flagVect);
  // If no output file name, derive it from input file basename
  if (outputBasename.empty()) {
    std::string filename = getInputFilename(inputFilename, flagVect);
    // Base name, strip
    outputBasename = filename.substr(0, filename.find_last_of("."));
  }
  EmissionTargetType targetType =
      getEmissionTargetFromFlags(inputFilename, flagVect);
  return getTargetFilename(outputBasename, targetType);
}

std::string getModelTag(const std::vector<std::string> &flagVect) {
  std::string modelTag = "";
  for (int i = 0; i < (int)flagVect.size(); ++i) {
    if (flagVect[i].find("--tag=") == 0) {
      modelTag = flagVect[i].substr(6);
      break;
    }
    if (flagVect[i].find("-tag=") == 0) {
      modelTag = flagVect[i].substr(5);
      break;
    }
  }
  return modelTag;
}

void applyOutputPath(std::vector<std::string> &flagVect,
    const std::string &outputPath, const std::string &inputFilename) {
  // If outputPath is empty, do nothing.
  if (outputPath.empty())
    return;

  // Loop through flags to find -o option.
  int num = flagVect.size();
  for (int i = 0; i < num; ++i) {
    if (flagVect[i].find("-o=", 0, 3) == 0 && flagVect[i].length() > 3) {
      // Found -o=value format.
      std::string outputBasename = flagVect[i].substr(3);
      fs::path p(outputBasename);
      fs::path parentPath = p.parent_path();

      if (parentPath.empty()) {
        // No path component, prepend outputPath.
        fs::path newPath = fs::path(outputPath) / p.filename();
        flagVect[i] = "-o=" + newPath.string();
      }
      // Otherwise path exists, leave unchanged.
      return;
    } else if (flagVect[i] == "-o" && i < num - 1) {
      // Found -o value format.
      std::string outputBasename = flagVect[i + 1];
      fs::path p(outputBasename);
      fs::path parentPath = p.parent_path();

      if (parentPath.empty()) {
        // No path component, prepend outputPath.
        fs::path newPath = fs::path(outputPath) / p.filename();
        flagVect[i + 1] = newPath.string();
      }
      // Otherwise path exists, leave unchanged.
      return;
    }
  }

  // Mo -o option found, add one based on outputPath and input basename.
  fs::path inputPath(inputFilename);
  std::string inputBasename = inputPath.stem().string();
  fs::path outputFilePath = fs::path(outputPath) / inputBasename;

  flagVect.push_back("-o");
  flagVect.push_back(outputFilePath.string());
}

} // namespace onnx_mlir
