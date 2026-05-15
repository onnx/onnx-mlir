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
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

// Include onnx-mlir infrastructure
#include "Command.hpp"
#include "CommandUtils.hpp"
#include <onnx-mlir/Compiler/OMCompilerTypes.h>

using namespace onnx_mlir;
namespace fs = std::filesystem;
//===----------------------------------------------------------------------===//
// Container Support - Internal Helper Class
//===----------------------------------------------------------------------===//

namespace {

/**
 * @class ContainerSupport
 * @brief Internal helper class for all container-related operations.
 *
 * This class encapsulates container engine detection, image management,
 * Docker-in-Docker support, and path resolution. It maintains verbose state
 * to avoid passing it to every method call.
 */
class ContainerSupport {
public:
  explicit ContainerSupport(bool verbose) : verbose(verbose) {}

  // Container Engine Detection
  std::string detectEngine(OMUnifiedCompile::ContainerEngine preferredEngine);

  // Image Management
  bool isImageAvailable(const std::string &engineName, const std::string &imageName);
  void pullImage(const std::string &engineName, const std::string &imageName);
  void verifyCompilerInContainer(const std::string &engineName,
      const std::string &imageName, const std::string &compilerPath);

  // Docker-in-Docker (DinD) Support
  bool detectDinDEnvironment();
  bool isDinDDisabled();
  void verifyDockerSocket(const std::string &engineName);
  std::string resolveHostPath(const std::string &containerPath);

private:
  const bool verbose;
};

//===----------------------------------------------------------------------===//
// ContainerSupport Implementation
//===----------------------------------------------------------------------===//

std::string ContainerSupport::detectEngine(
    OMUnifiedCompile::ContainerEngine preferredEngine) {
  // If user specified an engine, use it
  if (preferredEngine != OMUnifiedCompile::ContainerEngine::Auto) {
    return (preferredEngine == OMUnifiedCompile::ContainerEngine::Docker)
               ? "docker"
               : "podman";
  }

  // Try docker first
  try {
    Command dockerCheck("docker", verbose);
    dockerCheck.appendStr("--version");
    if (dockerCheck.exec() == 0) {
      if (verbose) {
        std::cout << "Detected container engine: docker" << std::endl;
      }
      return "docker";
    }
  } catch (...) {
    // Docker not available
  }

  // Try podman
  try {
    Command podmanCheck("podman", verbose);
    podmanCheck.appendStr("--version");
    if (podmanCheck.exec() == 0) {
      if (verbose) {
        std::cout << "Detected container engine: podman" << std::endl;
      }
      return "podman";
    }
  } catch (...) {
    // Podman not available
  }

  throw OMCompileException(
      "No container engine found. Please install Docker or Podman.");
}

bool ContainerSupport::isImageAvailable(
    const std::string &engineName, const std::string &imageName) {
  try {
    Command imageCheck(engineName, verbose);
    imageCheck.appendStr("images");
    imageCheck.appendStr("-q");
    imageCheck.appendStr(imageName);
    return imageCheck.exec() == 0;
  } catch (...) {
    return false;
  }
}

void ContainerSupport::pullImage(
    const std::string &engineName, const std::string &imageName) {
  if (verbose) {
    std::cout << "Pulling container image: " << imageName << std::endl;
  }

  try {
    Command pullCmd(engineName, verbose);
    pullCmd.appendStr("pull");
    pullCmd.appendStr(imageName);
    if (pullCmd.exec() != 0) {
      throw OMCompileException("Failed to pull container image: " + imageName);
    }
  } catch (const CommandException &e) {
    throw OMCompileException(
        "Failed to pull container image: " + std::string(e.what()));
  }
}

void ContainerSupport::verifyCompilerInContainer(const std::string &engineName,
    const std::string &imageName, const std::string &compilerPath) {
  if (!verbose)
    return;

  try {
    Command verifyCmd(engineName, verbose);
    verifyCmd.appendStr("run");
    verifyCmd.appendStr("--rm");
    verifyCmd.appendStr(imageName);
    verifyCmd.appendStr(compilerPath);
    verifyCmd.appendStr("--version");
    if (verifyCmd.exec() != 0) {
      std::cerr << "Warning: Compiler verification failed in container"
                << std::endl;
    }
  } catch (...) {
    std::cerr << "Warning: Could not verify compiler in container" << std::endl;
  }
}

bool ContainerSupport::detectDinDEnvironment() {
  // Method 1: Check for /.dockerenv file (Docker-specific marker)
  if (fs::exists("/.dockerenv")) {
    return true;
  }

  // Method 2: Check /proc/1/cgroup for container indicators
  std::ifstream cgroup("/proc/1/cgroup");
  if (cgroup.is_open()) {
    std::string line;
    while (std::getline(cgroup, line)) {
      if (line.find("docker") != std::string::npos ||
          line.find("containerd") != std::string::npos ||
          line.find("podman") != std::string::npos) {
        return true;
      }
    }
  }

  // Method 3: Check for container-specific environment variables
  // These are commonly set by container runtimes
  const char* containerEnv = std::getenv("container");
  if (containerEnv) {
    return true;
  }

  // Method 4: Check /run/.containerenv (Podman-specific marker)
  if (fs::exists("/run/.containerenv")) {
    return true;
  }

  // Method 5: Check if running as PID 1 with limited /proc/1/cgroup
  // In containers, especially with cgroups v2, /proc/1/cgroup might just show "0::/"
  // Combined with other indicators, this suggests containerization
  std::ifstream cgroupCheck("/proc/1/cgroup");
  if (cgroupCheck.is_open()) {
    std::string firstLine;
    std::getline(cgroupCheck, firstLine);
    // If cgroup is just "0::/" and we have container-like hostname, likely in container
    if (firstLine == "0::/") {
      // Check if hostname looks like a container ID (12+ hex chars)
      char hostname[256];
      if (gethostname(hostname, sizeof(hostname)) == 0) {
        std::string hostnameStr(hostname);
        // Container hostnames are typically 12 hex characters
        if (hostnameStr.length() == 12) {
          bool allHex = true;
          for (char c : hostnameStr) {
            if (!std::isxdigit(c)) {
              allHex = false;
              break;
            }
          }
          if (allHex) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool ContainerSupport::isDinDDisabled() {
  const char *dindDisable = std::getenv("DIND_DISABLE");
  return dindDisable && std::string(dindDisable) == "1";
}

void ContainerSupport::verifyDockerSocket(const std::string &engineName) {
  std::string socketPath = "/var/run/docker.sock";
  if (!fs::exists(socketPath)) {
    throw OMCompileException(
        "Docker-in-Docker detected but Docker socket not found at " +
        socketPath +
        ". Mount it with: -v /var/run/docker.sock:/var/run/docker.sock");
  }

  try {
    Command testCmd(engineName, false);
    testCmd.appendStr("info");
    if (testCmd.exec() != 0) {
      throw OMCompileException(
          "Docker socket exists but not accessible. Check permissions or "
          "ensure Docker daemon is running.");
    }

    if (verbose) {
      std::cout << "Docker socket verified: " << socketPath << std::endl;
    }
  } catch (const CommandException &e) {
    throw OMCompileException(
        "Failed to verify Docker socket access: " + std::string(e.what()));
  }
}

std::string ContainerSupport::resolveHostPath(const std::string &containerPath) {
  fs::path absPath = fs::absolute(containerPath);
  std::string pathStr = absPath.string();

  // Check for DOCKER_HOST_PATH_PREFIX environment variable
  const char *hostPrefix = std::getenv("DOCKER_HOST_PATH_PREFIX");
  if (hostPrefix && std::strlen(hostPrefix) > 0) {
    std::string prefix(hostPrefix);

    if (pathStr.find(prefix) == 0) {
      if (verbose) {
        std::cout << "Path already has host prefix: " << pathStr << std::endl;
      }
      return pathStr;
    }

    std::string resolvedPath = prefix + pathStr;
    if (verbose) {
      std::cout << "Resolved DinD path: " << containerPath << " -> "
                << resolvedPath << std::endl;
    }
    return resolvedPath;
  }

  // Default: assume paths are already host-relative
  if (verbose) {
    std::cout << "Using same path for DinD (no prefix): " << pathStr
              << std::endl;
  }

  return pathStr;
}

} // anonymous namespace


namespace onnx_mlir {

// Known container image configurations
// Note: Empty string means the container's entrypoint is the compiler itself
const std::map<std::string, std::string> OMUnifiedCompile::knownImageConfigs = {
    {"ghcr.io/onnxmlir/onnx-mlir", "onnx-mlir"}, // Entrypoint is onnx-mlir
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
      dindDetected(false), dindDetectionDone(false),
      successfullyInitialized(false), successfullyCompiled(false) {

  // Verify compiler is available (only if verbose)
  if (this->verbose) {
    try {
      Command verifyCmd(this->localCompilerPath, verbose);
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
OMUnifiedCompile::OMUnifiedCompile(const std::string &containerImage,
    const std::string &compilerPath, ContainerEngine engine, bool autoPull,
    bool verbose)
    : mode(CompilationMode::Container), verbose(verbose), localCompilerPath(),
      containerEngine(engine), autoPullImage(autoPull),
      dindDetected(false), dindDetectionDone(false),
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
  ContainerSupport support(verbose);
  detectedEngineName = support.detectEngine(containerEngine);
}

bool OMUnifiedCompile::isImageAvailable(const std::string &imageName) {
  ContainerSupport support(verbose);
  return support.isImageAvailable(detectedEngineName, imageName);
}

void OMUnifiedCompile::pullImage(const std::string &imageName) {
  ContainerSupport support(verbose);
  support.pullImage(detectedEngineName, imageName);
}

void OMUnifiedCompile::verifyContainerSetup() {
  ContainerSupport support(verbose);

  // Detect container engine
  detectContainerEngine();

  // Check if running in Docker-in-Docker and verify socket if so
  if (!support.isDinDDisabled() && support.detectDinDEnvironment()) {
    dindDetected = true;
    dindDetectionDone = true;

    if (verbose) {
      std::cout << "Docker-in-Docker (DinD) environment detected" << std::endl;
    }

    // Verify Docker socket is accessible
    support.verifyDockerSocket(detectedEngineName);
  }

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
  support.verifyCompilerInContainer(
      detectedEngineName, containerImage, compilerPathInContainer);
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
  // Note: When running in Docker-in-Docker (DinD), paths are automatically
  // resolved to host paths to ensure the inner container can access them.

  ContainerSupport support(verbose);

  // Resolve paths for Docker-in-Docker scenarios
  // In DinD, volume mounts must use paths from the HOST, not the outer container
  std::string hostModelDir = modelDir;
  std::string hostOutputDir = outputDir;

  if (isRunningInContainer()) {
    hostModelDir = support.resolveHostPath(modelDir);
    hostOutputDir = support.resolveHostPath(outputDir);
  }

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

  // Add input file (using container path - same as modelDir)
  std::string containerInputPath =
      (fs::path(modelDir) / fs::path(inputFilename).filename()).string();
  cmdStream << " " << containerInputPath;

  std::string containerCmd = cmdStream.str();

  if (verbose) {
    std::cout << "Container compilation with image: " << containerImage
              << std::endl;
    std::cout << "Container command: " << containerCmd << std::endl;
    if (isRunningInContainer()) {
      std::cout << "DinD mode: Host model dir: " << hostModelDir << std::endl;
      std::cout << "DinD mode: Host output dir: " << hostOutputDir << std::endl;
    }
  }

  // Build the docker/podman run command
  auto cmd = std::make_unique<Command>(detectedEngineName, verbose);
  cmd->appendStr("run");
  cmd->appendStr("--rm"); // Remove container after execution

  // Override the container's entrypoint to use sh
  // This is necessary because some images have onnx-mlir as the entrypoint
  cmd->appendStr("--entrypoint");
  cmd->appendStr("sh");

  // Mount model directory (using host path for DinD)
  cmd->appendStr("-v");
  cmd->appendStr(hostModelDir + ":" + modelDir + ":rw");

  // Mount output directory if different (using host path for DinD)
  if (outputDir != modelDir) {
    cmd->appendStr("-v");
    cmd->appendStr(hostOutputDir + ":" + outputDir + ":rw");
  }

  // Specify the image
  cmd->appendStr(containerImage);

  // Use -c to execute the command string
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
// Docker-in-Docker (DinD) Public Method
//===----------------------------------------------------------------------===//

bool OMUnifiedCompile::isRunningInContainer() const {
  ContainerSupport support(verbose);

  // Check if DinD detection is disabled
  if (support.isDinDDisabled()) {
    return false;
  }

  // Use cached result if available
  if (dindDetectionDone) {
    return dindDetected;
  }

  // Perform detection
  dindDetected = support.detectDinDEnvironment();
  dindDetectionDone = true;

  if (verbose && dindDetected) {
    std::cout << "Docker-in-Docker (DinD) environment detected" << std::endl;
  }

  return dindDetected;
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
