/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ExecutionSession.hpp - ExecutionSession Declaration --------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_EXECUTION_SESSION_H
#define ONNX_MLIR_EXECUTION_SESSION_H

#include <cassert>
#include <memory>
#include <string>

#include "OnnxMlirRuntime.h"

// LLVM provides the wrapper class, llvm::sys::DynamicLibrary, for dynamic
// library. When PYRUNTIME_LIGHT is built without the LLVM, the handle type for
// dynamic library in Linux is used. DynamicLibraryHandleType is defined for
// the two cases.
#ifndef ENABLE_PYRUNTIME_LIGHT
#include "llvm/Support/DynamicLibrary.h"
typedef llvm::sys::DynamicLibrary DynamicLibraryHandleType;
#else
typedef void *DynamicLibraryHandleType;
#endif

namespace onnx_mlir {

using entryPointFuncType = OMTensorList *(*)(OMTensorList *);
using queryEntryPointsFuncType = const char **(*)(int64_t *);
using signatureFuncType = const char *(*)(const char *);
using OMTensorUniquePtr = std::unique_ptr<OMTensor, decltype(&omTensorDestroy)>;

/* ExecutionSession
 * Class that supports executing compiled models.
 *
 * When the execution session does not work for known reasons, this class will
 * throw std::runtime_error errors. Errno info will provide further info about
 * the specific error that was raised.
 *
 * EFAULT when it could not load the library or a needed symbol was not found.
 * EINVAL when it expected an entry point prior to executing a specific
 * function.
 * EPERM when the model executed on a machine without a compatible
 * hardware/specialized accelerator.
 */
class ExecutionSession {
public:
  // Create an execution session using the model given in sharedLibPath.
  // This path must point to the actual file, local directory is not searched.
  ExecutionSession(std::string sharedLibPath, std::string tag = "",
      bool defaultEntryPoint = true);
  ~ExecutionSession();

  // Get a NULL-terminated array of entry point names.
  // For example {"run_addition, "run_subtraction", NULL}
  // In order to get the number of entry points, pass an integer pointer to the
  // function.
  const std::string *queryEntryPoints(int64_t *numOfEntryPoints) const;

  // Set entry point for this session.
  // Call this before running the session or querying signatures if
  // defaultEntryPoint is false or there are multiple entry points in the model.
  void setEntryPoint(const std::string &entryPointName);

  DynamicLibraryHandleType &getSharedLibraryHandle() {
    return _sharedLibraryHandle;
  };

  // Use custom deleter since forward declared OMTensor hides destructor
  std::vector<OMTensorUniquePtr> run(std::vector<OMTensorUniquePtr>);

  // Run using public interface. Explicit calls are needed to free tensor &
  // tensor lists.
  OMTensorList *run(OMTensorList *input);

  // Get input and output signature as a Json string. For example for nminst:
  // `[ { "type" : "f32" , "dims" : [1 , 1 , 28 , 28] , "name" : "image" } ]`
  const std::string inputSignature() const;
  const std::string outputSignature() const;

protected:
  // Constructor that build the object without initialization (for use by
  // subclass only).
  ExecutionSession() = default;

  // Initialization of library. Called by public constructor, or by subclasses.
  void Init(std::string sharedLibPath, std::string tag, bool defaultEntryPoint);

  // Error reporting processing when throwing runtime errors. Set errno as
  // appropriate.
  std::string reportInitError() const;
  std::string reportLibraryOpeningError(const std::string &libraryName) const;
  std::string reportSymbolLoadingError(const std::string &symbolName) const;
  std::string reportUndefinedEntryPointIn(
      const std::string &functionName) const;
  std::string reportErrnoError() const;
  std::string reportCompilerError(const std::string &errorMessage) const;

  // Track if Init was called or not.
  bool isInitialized = false;

  // Handler to the shared library file being loaded.
  DynamicLibraryHandleType _sharedLibraryHandle;

  // Tag used to compile the model. By default, it is the model filename without
  // extension.
  std::string tag;

  // Entry point function.
  std::string _entryPointName;
  entryPointFuncType _entryPointFunc = nullptr;

  // Query entry point function.
  const std::string _queryEntryPointsName = "omQueryEntryPoints";
  queryEntryPointsFuncType _queryEntryPointsFunc = nullptr;

  // Entry point for input/output signatures
  const std::string _inputSignatureName = "omInputSignature";
  const std::string _outputSignatureName = "omOutputSignature";
  signatureFuncType _inputSignatureFunc = nullptr;
  signatureFuncType _outputSignatureFunc = nullptr;
};
} // namespace onnx_mlir
#endif
