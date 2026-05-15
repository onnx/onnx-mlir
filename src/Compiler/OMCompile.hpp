/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMCompiler.hpp - compiler driver  ------------------------===//
//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to compile onnx model files in .onnx, .mlir, or
// .onnxtext using onnx-mlir.
//
// This file should not include any ONNX-MLIR / MLIR / LLVM dependences except
// for onnx-mlir/include.
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_SESSION
#define ONNX_MLIR_COMPILER_SESSION

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// TODO: should ExecutionSession and OMCompile be in the onnx_mlir
// namespace? They should not depend at all on the onnx-mlir compiler files
// (except implicitly).
namespace onnx_mlir {

// Forward declarations.
class Command;

// Exception class.
class OMCompileException : public std::runtime_error {
public:
  explicit OMCompileException(const std::string &msg)
      : std::runtime_error(msg) {}
};

/**
 * @class OMCompile
 * @brief Unified C++ interface for compiling ONNX models locally or in
 * containers.
 *
 * This class provides a thread-safe interface to compile ONNX models from files
 * (.onnx, .mlir, or .onnxtext formats) into various output formats such as
 * shared libraries (.so/.dll), object files (.o/.obj), or JAR files.
 *
 * ## Compilation Modes
 * The class supports three compilation modes:
 * 1. **Local compilation** - Direct execution of onnx-mlir binary
 * 2. **Container compilation** - onnx-mlir running in Docker/Podman
 * 3. **Docker-in-Docker (DinD)** - Automatic detection and handling
 *
 * ## Thread Safety
 * This interface is thread-safe and does not read any flags from environment
 * variables (except for DinD configuration). All compilation options must be
 * explicitly passed via the flags parameter.
 *
 * ## Compilation Process
 * The class invokes the onnx-mlir compiler executable (locally or in container)
 * to perform the actual compilation. When generating libraries or JAR files,
 * the compiler automatically links in the required lightweight runtime
 * libraries.
 *
 * ## Runtime Library Location
 * By default, runtime libraries are expected in system-wide directories
 * (typically /usr/local/lib). To use a custom location, set the
 * ONNX_MLIR_LIBRARY_PATH environment variable before compilation.
 *
 * ## Usage Examples
 *
 * ### Local Compilation
 * @code
 *   OMCompile session;  // Default constructor for local mode
 *   try {
 *     session.compile("model.onnx", "-O3 -o output");
 *     std::string outputFile = session.getOutputFilename();
 *     std::cout << "Compiled to: " << outputFile << std::endl;
 *   } catch (const OMCompileException& e) {
 *     std::cerr << "Compilation failed: " << e.what() << std::endl;
 *   }
 * @endcode
 *
 * ### Container Compilation
 * @code
 *   // Use containerized onnx-mlir
 *   OMCompile session(
 *       "ghcr.io/onnxmlir/onnx-mlir-dev",  // image
 *       "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"  // compiler in
 * container
 *   );
 *   session.compile("model.onnx", "-O3");
 * @endcode
 *
 * ### Docker-in-Docker (DinD)
 * @code
 *   // DinD is automatically detected - no special code needed!
 *   // Just ensure Docker socket is mounted when running outer container:
 *   // docker run -v /var/run/docker.sock:/var/run/docker.sock ...
 *
 *   OMCompile session("image", "compiler");
 *   session.compile("model.onnx", "-O3");
 *   // Paths are automatically resolved for host access
 * @endcode
 *
 * ## Supported Compiler Flags
 * All flags available to the onnx-mlir command-line tool are supported:
 * - `-o <filename>`: Specify output file name
 * - `-EmitLib`: Generate shared library (default)
 * - `-EmitObj`: Generate object file
 * - `-EmitJNI`: Generate JAR file with JNI bindings
 * - `-O0/-O1/-O2/-O3`: Optimization levels
 * - `--tag=<name>`: Add a tag to the compiled model
 * - And many more (see onnx-mlir documentation)
 *
 * @note Flags can contain quoted strings (e.g., `-o "path with
 * spaces/model.so"`) which will be properly parsed.
 */
class OMCompile {
public:
  /// Container engine type.
  enum class ContainerEngine { Docker, Podman, Auto };

  /// Compilation mode.
  enum class CompilationMode { Local, Container };

  /**
   * @brief Default constructor for local compilation.
   *
   * Creates an OMCompile object for local compilation using the default
   * onnx-mlir binary from PATH. Actual compilation is deferred until
   * the compile() method is called.
   */
  OMCompile();

  /**
   * @brief Constructor for local compilation with custom compiler path.
   *
   * @param compilerPath Path to local onnx-mlir binary (empty uses PATH
   * default)
   * @param verbose Enable verbose output (default: false)
   *
   * @code
   *   // Use onnx-mlir from PATH
   *   OMCompile compiler({});
   *
   *   // Use specific compiler
   *   OMCompile compiler("/path/to/onnx-mlir");
   *
   *   // With verbose
   *   OMCompile compiler({}, true);
   * @endcode
   */
  explicit OMCompile(const std::string &compilerPath, bool verbose = false);

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
   * - OM_DOCKER_HOST_PATH_PREFIX: Prefix to add to paths (e.g., "/host")
   * - OM_DIND_DISABLE: Set to "1" to disable DinD detection
   *
   * @code
   *   // Use defaults (first known image, auto-detect compiler, auto-detect
   * engine) OMCompile compiler({}, {});
   *
   *   // Specific image with auto-detected compiler path
   *   OMCompile compiler("ghcr.io/onnxmlir/onnx-mlir", {});
   *
   *   // With verbose mode (shows DinD detection info)
   *   OMCompile compiler({}, {}, OMCompile::ContainerEngine::Auto, true, true);
   *
   *   // Docker-in-Docker works automatically - no special configuration needed
   * @endcode
   */
  OMCompile(const std::string &containerImage,
      const std::string &compilerPathInContainer,
      ContainerEngine engine = ContainerEngine::Auto, bool autoPull = true,
      bool verbose = false);

  /**
   * @brief Destructor.
   */
  ~OMCompile() = default;

  /**
   * @brief Compile an ONNX model with specified flags.
   *
   * Invokes the onnx-mlir compiler (locally or in container) to compile the
   * input model. The method blocks until compilation completes or fails.
   * Works identically for both local and container modes.
   *
   * @param modelPath Path to the input model file (.onnx, .mlir, or .onnxtext).
   * Can include a directory path. If empty, the flags parameter must contain
   * the input filename.
   * @param flags Compilation flags as a single string (e.g., "-O3 -o output").
   *              Supports quoted strings for paths with spaces.
   * @param compilerPath Optional path to the compiler binary, including the
   * binary name. If empty (default) standard onnx-mlir binary will be used at
   * standard location. Only used in local mode.
   * @param logFilename Optional path to a file where compilation logs will be
   *                    written. If empty, logs go to stdout/stderr.
   *
   * @throws OMCompileException if compilation fails for any reason
   *         (invalid input, compiler errors, missing dependencies, etc.)
   */
  void compile(const std::string &modelPath, const std::string &flags,
      const std::string &compilerPath = {},
      const std::string &logFilename = {});

  /**
   * @brief Get the output filename of the compiled model.
   *
   * Returns the absolute path to the file generated by the most recent
   * successful compilation.
   *
   * @return Absolute path to the compiled output file
   * @throws std::runtime_error if called before a successful compilation
   */
  std::string getOutputFilename();

  /**
   * @brief Get the output constant filename of the compiled model.
   *
   * When compiling large models, the compiler may generate a separate output
   * file that contains some of the larger constants. Note that the compiled
   * model and compiled constant file must share the same filename except for
   * their respective extensions.
   *
   * This call returns the absolute path to the constant file generated by the
   * most recent successful compilation.
   *
   * @return Absolute path to the compiled constant output file. Empty if there
   * are no constant file.
   * @throws std::runtime_error if called before a successful compilation
   */
  std::string getOutputConstantFilename();

  /**
   * @brief Get the model tag of the compiled model.
   *
   * Returns the tag specified via the --tag flag during compilation, or an
   * empty string if no tag was specified.
   *
   * @return Model tag string, or empty if no tag was set
   * @throws std::runtime_error if called before a successful compilation
   */
  std::string getModelTag();

  /**
   * @brief Check if the last compilation was successful.
   *
   * @return true if compile() completed successfully, false otherwise
   */
  bool isSuccessfullyCompiled() { return successfullyCompiled; }

  /**
   * @brief Check if the compiled model has an associated constant file.
   * @return true the compiled model compile() completed successfully and the
   * output generated by the compiler includes a constant file, false otherwise
   */
  bool hasOutputConstantFilename() { return !outputConstantFilename.empty(); }

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
   * - /run/.containerenv file (Podman-specific marker)
   * - Hostname pattern + cgroups v2 (modern Podman)
   *
   * This detection is cached after first call for performance.
   * Can be disabled by setting OM_DIND_DISABLE=1 environment variable.
   *
   * @return true if running in a container, false otherwise
   */
  bool isRunningInContainer() const;

  /**
   * @brief Static helper to extract input filename from model path and flags.
   *
   * Useful for determining the input file before compilation, especially when
   * implementing caching mechanisms.
   *
   * @param modelPath Model path parameter (may be empty)
   * @param flags Compilation flags string
   * @return The input filename that would be used for compilation
   */
  static std::string getInputFilename(
      const std::string &modelPath, const std::string &flags);

  /**
   * @brief Static helper to predict output filename from model path and flags.
   *
   * Determines what the output filename would be based on the input and flags,
   * without actually performing compilation. Useful for caching and
   * pre-checking if output already exists.
   *
   * @param modelPath Model path parameter (may be empty)
   * @param flags Compilation flags string
   * @return The output filename that would be generated
   */
  static std::string getOutputFilename(
      const std::string &modelPath, const std::string &flags);

  /**
   * @brief Static helper to extract model tag from compilation flags.
   *
   * Parses the flags string to find the --tag or -tag option value.
   *
   * @param flags Compilation flags string
   * @return The model tag if specified in flags, empty string otherwise
   */
  static std::string getModelTag(const std::string &flags);

private:
  // Compilation mode.
  const CompilationMode mode;

  // Common configuration.
  const bool verbose;

  // Local mode configuration.
  const std::string localCompilerPath;

  // Container mode configuration.
  std::string containerImage;          // Not const - set in constructor body.
  std::string compilerPathInContainer; // Not const - set in constructor body.
  const ContainerEngine containerEngine;
  const bool autoPullImage;
  std::string detectedEngineName; // "docker" or "podman".

  // Docker-in-Docker state (cached detection).
  mutable bool dindDetected = false; // Cached DinD detection result.
  mutable bool dindDetectionDone =
      false; // Whether detection has been performed.

  // Compilation state.
  /// Parsed compilation flags as a vector of individual arguments.
  std::vector<std::string> flagVect;

  /// Absolute path to the output file from the last successful compilation.
  std::string outputFilename = {};
  std::string outputConstantFilename = {};

  /// Flag indicating whether the last compilation completed successfully.
  bool successfullyInitialized =
      false; // Set to true when constructor completes.
  bool successfullyCompiled = false;

  // Known container image configurations.
  static const std::map<std::string, std::string> knownImageConfigs;

  // Helper methods for container mode.
  void verifyContainerSetup();

  // Core compilation logic - separate methods for clarity.
  std::unique_ptr<Command> createLocalCompileCommand(
      const std::string &modelPath, const std::vector<std::string> &flagVect,
      const std::string &inputFilename);

  std::unique_ptr<Command> createContainerCompileCommand(
      const std::vector<std::string> &flagVect,
      const std::string &inputFilename, const std::string &modelDir,
      const std::string &outputDir);
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_COMPILER_SESSION
