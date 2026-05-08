/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OMUnifiedCompile.cpp - Unified compiler driver ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile ONNX model files using onnx-mlir
// either locally or inside a Docker/Podman container.
//
//===----------------------------------------------------------------------===//

#include "OMUnifiedCompile.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

// Include onnx-mlir infrastructure
#include "Command.hpp"
#include "CommandUtils.hpp"
#include <onnx-mlir/Compiler/OMCompilerTypes.h>

using namespace onnx_mlir;
namespace fs = std::filesystem;

namespace onnx_mlir {

// Known container image configurations
const std::map<std::string, std::string> OMUnifiedCompile::knownImageConfigs = {
    {"ghcr.io/onnxmlir/onnx-mlir", "/usr/local/bin/bin/onnx-mlir"},
    {"ghcr.io/onnxmlir/onnx-mlir-dev",
        "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"},
    {"onnxmlir/onnx-mlir-dev", "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"},
    {"icr.io/ibmz/zdlc:5.0.0", ""}, // Entry point is the compiler
};

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

// Constructor for local compilation
OMUnifiedCompile::OMUnifiedCompile(
    const std::string &compilerPath, bool verbose)
    : mode(CompilationMode::Local), verbose(verbose),
      localCompilerPath(compilerPath.empty()
#ifdef _WIN32
                            ? "onnx-mlir.exe"
#else
                            ? "onnx-mlir"
#endif
                            : compilerPath),
      containerImage(), compilerPathInContainer(),
      containerEngine(ContainerEngine::Auto), autoPullImage(false),
      successfullyInitialized(false), successfullyCompiled(false) {

  // Verify compiler is available (only if verbose)
  if (this->verbose) {
    try {
      Command verifyCmd(this->localCompilerPath, false);
      verifyCmd.appendStr("--version");
      int status = verifyCmd.exec();
      if (status != 0) {
        std::cerr << "Warning: Local compiler verification failed for: "
                  << this->localCompilerPath << std::endl;
      } else {
        std::cout << "Local compiler verified: " << this->localCompilerPath
                  << std::endl;
      }
    } catch (...) {
      std::cerr << "Warning: Could not verify local compiler: "
                << this->localCompilerPath << std::endl;
    }
  }

  // Mark as successfully initialized
  successfullyInitialized = true;
}

// Constructor for container compilation
OMUnifiedCompile::OMUnifiedCompile(ContainerEngine engine,
    const std::string &containerImage, const std::string &compilerPath,
    bool verbose, bool autoPull)
    : mode(CompilationMode::Container), verbose(verbose), localCompilerPath(),
      containerEngine(engine), autoPullImage(autoPull),
      successfullyInitialized(false), successfullyCompiled(false) {

  // Set container image - use first known image if not provided
  if (containerImage.empty()) {
    if (!knownImageConfigs.empty()) {
      this->containerImage = knownImageConfigs.begin()->first;
    } else {
      throw OMCompileException(
          "No container image provided and no known images configured");
    }
  } else {
    this->containerImage = containerImage;
  }

  // Set compiler path - auto-detect for known images if not provided
  if (compilerPath.empty()) {
    // Try to find compiler path for the selected image
    auto it = knownImageConfigs.find(this->containerImage);
    if (it != knownImageConfigs.end()) {
      compilerPathInContainer = it->second;
    } else {
      compilerPathInContainer = "";
    }
  } else {
    compilerPathInContainer = compilerPath;
  }

  // Perform one-time container setup
  verifyContainerSetup();

  // Mark as successfully initialized (only if verifyContainerSetup didn't
  // throw)
  successfullyInitialized = true;
}

//===----------------------------------------------------------------------===//
// Container Helper Methods
//===----------------------------------------------------------------------===//

void OMUnifiedCompile::detectContainerEngine() {
  if (containerEngine != ContainerEngine::Auto) {
    detectedEngineName =
        (containerEngine == ContainerEngine::Docker) ? "docker" : "podman";
    return;
  }

  // Try docker first
  try {
    Command dockerCheck("docker", false);
    dockerCheck.appendStr("--version");
    int status = dockerCheck.exec();
    if (status == 0) {
      detectedEngineName = "docker";
      if (verbose) {
        std::cout << "Detected container engine: docker" << std::endl;
      }
      return;
    }
  } catch (...) {
    // Docker not available
  }

  // Try podman
  try {
    Command podmanCheck("podman", false);
    podmanCheck.appendStr("--version");
    int status = podmanCheck.exec();
    if (status == 0) {
      detectedEngineName = "podman";
      if (verbose) {
        std::cout << "Detected container engine: podman" << std::endl;
      }
      return;
    }
  } catch (...) {
    // Podman not available
  }

  throw OMCompileException(
      "No container engine found. Please install Docker or Podman.");
}

bool OMUnifiedCompile::isImageAvailable(const std::string &imageName) {
  try {
    Command imageCheck(detectedEngineName, false);
    imageCheck.appendStr("images");
    imageCheck.appendStr("-q");
    imageCheck.appendStr(imageName);
    int status = imageCheck.exec();
    return status == 0;
  } catch (...) {
    return false;
  }
}

void OMUnifiedCompile::pullImage(const std::string &imageName) {
  if (verbose) {
    std::cout << "Pulling container image: " << imageName << std::endl;
  }

  try {
    Command pullCmd(detectedEngineName, verbose);
    pullCmd.appendStr("pull");
    pullCmd.appendStr(imageName);
    int status = pullCmd.exec();
    if (status != 0) {
      throw OMCompileException("Failed to pull container image: " + imageName);
    }
  } catch (const CommandException &e) {
    throw OMCompileException(
        "Failed to pull container image: " + std::string(e.what()));
  }
}

void OMUnifiedCompile::verifyContainerSetup() {
  // Detect container engine
  detectContainerEngine();

  // Verify image is available
  if (!isImageAvailable(containerImage)) {
    if (autoPullImage) {
      pullImage(containerImage);
    } else {
      throw OMCompileException("Container image not found: " + containerImage +
                               ". Enable auto-pull or pull manually.");
    }
  }

  // Verify compiler path is set
  if (compilerPathInContainer.empty()) {
    throw OMCompileException(
        "Compiler path in container not set. Use a known image or specify "
        "compiler path.");
  }

  // Optionally verify compiler exists in container
  if (verbose) {
    try {
      Command verifyCmd(detectedEngineName, false);
      verifyCmd.appendStr("run");
      verifyCmd.appendStr("--rm");
      verifyCmd.appendStr(containerImage);
      verifyCmd.appendStr(compilerPathInContainer);
      verifyCmd.appendStr("--version");
      int status = verifyCmd.exec();
      if (status != 0) {
        std::cerr << "Warning: Compiler verification failed in container"
                  << std::endl;
      }
    } catch (...) {
      std::cerr << "Warning: Could not verify compiler in container"
                << std::endl;
    }
  }
}

//===----------------------------------------------------------------------===//
// Core Compilation Logic - Separate Methods for Local and Container
//===----------------------------------------------------------------------===//

std::unique_ptr<Command> OMUnifiedCompile::createLocalCompileCommand(
    const std::string &modelPath, const std::vector<std::string> &flagVect,
    const std::string &inputFilename) {

  // Local Mode: Direct compiler execution
  auto cmd = std::make_unique<Command>(localCompilerPath, verbose);
  cmd->appendList(flagVect);
  if (!modelPath.empty()) {
    cmd->appendStr(inputFilename);
  }
  return cmd;
}

std::unique_ptr<Command> OMUnifiedCompile::createContainerCompileCommand(
    const std::vector<std::string> &flagVect, const std::string &inputFilename,
    const std::string &modelDir, const std::string &outputDir) {

  // Container Mode: Docker/Podman run with volume mounts

  // Build the compiler command string that will run inside the container
  std::ostringstream cmdStream;
  cmdStream << compilerPathInContainer;

  // Add flags
  for (const auto &flag : flagVect) {
    // Escape any quotes in the flag
    std::string escapedFlag = flag;
    size_t pos = 0;
    while ((pos = escapedFlag.find('"', pos)) != std::string::npos) {
      escapedFlag.insert(pos, "\\");
      pos += 2;
    }
    cmdStream << " " << escapedFlag;
  }

  // Add input file (using container path)
  std::string containerInputPath =
      (fs::path(modelDir) / fs::path(inputFilename).filename()).string();
  cmdStream << " " << containerInputPath;

  std::string containerCmd = cmdStream.str();

  if (verbose) {
    std::cout << "Container compilation with image: " << containerImage
              << std::endl;
    std::cout << "Container command: " << containerCmd << std::endl;
  }

  // Build the docker/podman run command
  auto cmd = std::make_unique<Command>(detectedEngineName, verbose);
  cmd->appendStr("run");
  cmd->appendStr("--rm"); // Remove container after execution

  // Mount model directory
  cmd->appendStr("-v");
  cmd->appendStr(modelDir + ":" + modelDir + ":rw");

  // Mount output directory if different
  if (outputDir != modelDir) {
    cmd->appendStr("-v");
    cmd->appendStr(outputDir + ":" + outputDir + ":rw");
  }

  // Specify the image
  cmd->appendStr(containerImage);

  // Use sh -c to execute the command string inside the container
  // This is necessary because we're passing a command string with arguments
  cmd->appendStr("sh");
  cmd->appendStr("-c");
  cmd->appendStr(containerCmd);

  return cmd;
}

//===----------------------------------------------------------------------===//
// Public Compilation Method - Shared Logic for Both Modes
//===----------------------------------------------------------------------===//

void OMUnifiedCompile::compile(const std::string &modelPath,
    const std::string &flags, const std::string &logFilename) {

  // Verify object was successfully initialized
  if (!successfullyInitialized) {
    throw OMCompileException(
        "Compiler not properly initialized. Constructor may have failed.");
  }

  // Initialize state
  successfullyCompiled = false;
  outputFilename = {};
  outputConstantFilename = {};

  // Parse flags and determine input/output files (shared logic)
  std::vector<std::string> flagVect = parseFlags(flags);
  std::string inputFilename = onnx_mlir::getInputFilename(modelPath, flagVect);

  if (inputFilename.empty()) {
    throw OMCompileException("Compilation failed: missing input model file");
  }

  if (!fs::exists(inputFilename)) {
    throw OMCompileException(
        "Compilation failed: could not locate input model file \"" +
        inputFilename + "\"");
  }

  // Get absolute paths for directories (needed for container mode)
  fs::path inputPath = fs::absolute(inputFilename);
  std::string modelDir = inputPath.parent_path().string();

  // Determine output directory
  std::string outputDir;
  std::string predictedOutput =
      onnx_mlir::getOutputFilename(inputFilename, flagVect);
  if (!predictedOutput.empty()) {
    fs::path outputPath = fs::absolute(predictedOutput);
    outputDir = outputPath.parent_path().string();
  } else {
    outputDir = modelDir;
  }

  // Create the appropriate Command based on mode
  std::unique_ptr<Command> cmd;
  if (mode == CompilationMode::Local) {
    cmd = createLocalCompileCommand(modelPath, flagVect, inputFilename);
  } else {
    cmd = createContainerCompileCommand(
        flagVect, inputFilename, modelDir, outputDir);
  }

  // Redirect logs if specified (shared logic)
  if (!logFilename.empty()) {
    cmd->redirectExecStreams(logFilename);
  }

  // Execute the command (shared logic)
  int status;
  try {
    status = cmd->exec();
  } catch (const CommandException &error) {
    std::string errorMessage;
    if (verbose) {
      errorMessage = error.what();
      std::cerr << "Command exception: " << errorMessage << std::endl;
    }
    throw OMCompileException(
        "Compilation failed: " + std::string(error.what()));
  }

  // Check compilation status (shared logic)
  if (status != OnnxMlirCompilerErrorCodes::CompilerSuccess) {
    std::string errorMessage(
        onnx_mlir::getOnnxMlirCompilerErrorDescription(status));
    throw OMCompileException(
        "Compilation failed with error code: " + errorMessage);
  }

  // Success - determine output filename (shared logic)
  std::string name = onnx_mlir::getOutputFilename(inputFilename, flagVect);
  outputFilename = getAbsolutePathUsingCurrentDir(name);

  // Check if output file was actually created
  if (!fs::exists(outputFilename)) {
    throw OMCompileException(
        "Compilation appeared to succeed but output file not found: " +
        outputFilename);
  }

  successfullyCompiled = true;

  // Check for constant file (shared logic)
  std::string constFilename = fs::path(outputFilename).stem().string();
  constFilename += ".constants.bin";
  if (fs::exists(constFilename)) {
    outputConstantFilename = constFilename;
  }

  if (verbose) {
    std::cout << "Compilation successful. Output: " << outputFilename
              << std::endl;
  }
}

//===----------------------------------------------------------------------===//
// Accessor Methods
//===----------------------------------------------------------------------===//

std::string OMUnifiedCompile::getOutputFilename() const {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  return outputFilename;
}

std::string OMUnifiedCompile::getOutputConstantFilename() const {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  return outputConstantFilename;
}

std::string OMUnifiedCompile::getModelTag() const {
  if (!successfullyCompiled) {
    throw OMCompileException(
        "Compiler session: has no successfully compiled model");
  }
  // Would need to store original flags to extract tag
  return "";
}

std::string OMUnifiedCompile::getContainerEngineName() const {
  if (mode != CompilationMode::Container) {
    return "";
  }
  return detectedEngineName;
}

} // namespace onnx_mlir
