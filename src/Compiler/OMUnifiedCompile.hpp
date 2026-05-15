/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OMUnifiedCompile.hpp - Unified compiler driver ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile ONNX model files using onnx-mlir
// either locally or inside a Docker/Podman container, with maximum code reuse.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_UNIFIED_COMPILE_HPP
#define ONNX_MLIR_UNIFIED_COMPILE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

// Include OMCompile.hpp to get OMCompileException definition
#include "OMCompile.hpp"

namespace onnx_mlir {

// Forward declarations
class Command;

/**
 * @class OMUnifiedCompile
 * @brief Unified C++ interface for compiling ONNX models locally or in
 * containers.
 *
 * This class provides a single interface that supports both:
 * 1. Local compilation (direct execution of onnx-mlir binary)
 * 2. Container compilation (onnx-mlir running in Docker/Podman)
 * 3. Docker-in-Docker (DinD) compilation with automatic detection
 *
 * ## Design Philosophy
 * - Single class with two constructors (local vs container)
 * - Maximum code reuse - only Command construction differs
 * - Leverages existing OMCompile infrastructure
 * - Immutable configuration after construction
 * - Automatic Docker-in-Docker detection and handling
 *
 * ## Usage Examples
 *
 * ### Local Compilation
 * @code
 *   // Use local onnx-mlir binary
 *   OMUnifiedCompile compiler;  // Uses default binary in PATH
 *   // or
 *   OMUnifiedCompile compiler("/path/to/onnx-mlir");
 *
 *   compiler.compile("model.onnx", "-O3");
 * @endcode
 *
 * ### Container Compilation
 * @code
 *   // Use containerized onnx-mlir
 *   OMUnifiedCompile compiler(
 *       "ghcr.io/onnxmlir/onnx-mlir-dev",  // image
 *       "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"  // compiler in
 * container
 *   );
 *
 *   compiler.compile("model.onnx", "-O3");
 * @endcode
 *
 * ### Docker-in-Docker (DinD) Compilation
 * @code
 *   // DinD is automatically detected and handled - no special code needed!
 *   // Just ensure Docker socket is mounted when running outer container:
 *   // docker run -v /var/run/docker.sock:/var/run/docker.sock ...
 *
 *   OMUnifiedCompile compiler("image", "compiler");
 *   compiler.compile("model.onnx", "-O3");
 *   // Paths are automatically resolved for host access
 * @endcode
 */
class OMUnifiedCompile {
public:
  /// Container engine type
  enum class ContainerEngine { Docker, Podman, Auto };

  /// Compilation mode
  enum class CompilationMode { Local, Container };

  /**
   * @brief Constructor for local compilation.
   *
   * @param compilerPath Path to local onnx-mlir binary (required, but {} uses
   * PATH default)
   * @param verbose Enable verbose output (default: false)
   *
   * @code
   *   // Use onnx-mlir from PATH
   *   OMUnifiedCompile compiler({});
   *
   *   // Use specific compiler
   *   OMUnifiedCompile compiler("/path/to/onnx-mlir");
   *
   *   // With verbose
   *   OMUnifiedCompile compiler({}, true);
   * @endcode
   */
  explicit OMUnifiedCompile(
      const std::string &compilerPath, bool verbose = false);

  /**
   * @brief Constructor for container-based compilation.
   *
   * Performs one-time setup:
   * - Detects container engine (docker/podman) if Auto
   * - Verifies/pulls container image
   * - Auto-detects compiler path for known images
   * - Automatically detects and handles Docker-in-Docker (DinD) scenarios
   *
   * @param containerImage Container image name (required, but {} uses first
   * known image)
   * @param compilerPathInContainer Path to compiler in container (required, but
   * {} auto-detects)
   * @param engine Container engine to use (default: Auto - auto-detect)
   * @param autoPull Automatically pull missing images (default: true)
   * @param verbose Enable verbose output (default: false)
   *
   * @note Docker-in-Docker Support:
   * When running inside a container, this class automatically detects the DinD
   * scenario and adjusts volume mount paths accordingly. For DinD to work:
   * - Mount Docker socket: -v /var/run/docker.sock:/var/run/docker.sock
   * - Use absolute paths for model files
   * - Ensure paths are accessible from the host (not just outer container)
   *
   * Environment variables for advanced DinD configuration:
   * - DOCKER_HOST_PATH_PREFIX: Prefix to add to paths (e.g., "/host")
   * - DIND_DISABLE: Set to "1" to disable DinD detection
   *
   * @code
   *   // Use defaults (first known image, auto-detect compiler, auto-detect
   * engine) OMUnifiedCompile compiler({}, {});
   *
   *   // Specific image with auto-detected compiler path
   *   OMUnifiedCompile compiler("ghcr.io/onnxmlir/onnx-mlir", {});
   *
   *   // With verbose mode (shows DinD detection info)
   *   OMUnifiedCompile compiler({}, {}, ContainerEngine::Auto, true, true);
   *
   *   // Docker-in-Docker works automatically - no special configuration needed
   * @endcode
   */
  OMUnifiedCompile(const std::string &containerImage,
      const std::string &compilerPathInContainer,
      ContainerEngine engine = ContainerEngine::Auto, bool autoPull = true,
      bool verbose = false);

  /**
   * @brief Destructor.
   */
  ~OMUnifiedCompile() = default;

  /**
   * @brief Compile an ONNX model.
   *
   * Works identically for both local and container modes.
   *
   * @param modelPath Path to the input model file
   * @param flags Compilation flags as a single string
   * @param logFilename Optional path to log file
   *
   * @throws OMCompileException if compilation fails
   */
  void compile(const std::string &modelPath, const std::string &flags,
      const std::string &logFilename = {});

  /**
   * @brief Get the output filename of the compiled model.
   *
   * @return Absolute path to the compiled output file
   * @throws OMCompileException if called before successful compilation
   */
  std::string getOutputFilename() const;

  /**
   * @brief Get the output constant filename of the compiled model.
   *
   * @return Absolute path to constant file, or empty if none
   * @throws OMCompileException if called before successful compilation
   */
  std::string getOutputConstantFilename() const;

  /**
   * @brief Get the model tag of the compiled model.
   *
   * @return Model tag string, or empty if no tag was set
   * @throws OMCompileException if called before successful compilation
   */
  std::string getModelTag() const;

  /**
   * @brief Check if the last compilation was successful.
   */
  bool isSuccessfullyCompiled() const { return successfullyCompiled; }

  /**
   * @brief Check if the compiled model has an associated constant file.
   */
  bool hasOutputConstantFilename() const {
    return !outputConstantFilename.empty();
  }

  /**
   * @brief Get the compilation mode.
   */
  CompilationMode getMode() const { return mode; }

  /**
   * @brief Get the container engine name (only for container mode).
   */
  std::string getContainerEngineName() const;

  /**
   * @brief Check if currently running inside a container (Docker-in-Docker).
   *
   * Detects Docker-in-Docker scenarios by checking:
   * - Presence of /.dockerenv file (Docker-specific marker)
   * - Container indicators in /proc/1/cgroup (works for Docker and Podman)
   *
   * This detection is cached after first call for performance.
   * Can be disabled by setting DIND_DISABLE=1 environment variable.
   *
   * @return true if running in a container, false otherwise
   */
  bool isRunningInContainer() const;

private:
  // Compilation mode
  const CompilationMode mode;

  // Common configuration
  const bool verbose;

  // Local mode configuration
  const std::string localCompilerPath;

  // Container mode configuration
  std::string containerImage;          // Not const - set in constructor body
  std::string compilerPathInContainer; // Not const - set in constructor body
  const ContainerEngine containerEngine;
  const bool autoPullImage;
  std::string detectedEngineName; // "docker" or "podman"

  // Docker-in-Docker state (cached detection)
  mutable bool dindDetected;      // Cached DinD detection result
  mutable bool dindDetectionDone; // Whether detection has been performed

  // Compilation state
  std::string outputFilename;
  std::string outputConstantFilename;
  bool successfullyInitialized; // Set to true when constructor completes
  bool successfullyCompiled;

  // Known container image configurations
  static const std::map<std::string, std::string> knownImageConfigs;

  // Helper methods for container mode
  void detectContainerEngine();
  bool isImageAvailable(const std::string &imageName);
  void pullImage(const std::string &imageName);
  void verifyContainerSetup();

  // Docker-in-Docker helper methods
  bool detectDinDEnvironment() const;
  void verifyDockerSocketAccess() const;
  std::string resolveHostPath(const std::string &containerPath) const;

  // Core compilation logic - separate methods for clarity
  std::unique_ptr<Command> createLocalCompileCommand(
      const std::string &modelPath, const std::vector<std::string> &flagVect,
      const std::string &inputFilename);

  std::unique_ptr<Command> createContainerCompileCommand(
      const std::vector<std::string> &flagVect,
      const std::string &inputFilename, const std::string &modelDir,
      const std::string &outputDir);
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_UNIFIED_COMPILE_HPP
