//===--------- ExecutionSession.hpp - ExecutionSession Declaration --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of ExecutionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <dlfcn.h>
#include <string>

#include "OnnxMlirInternal.h"

namespace onnx_mlir {

typedef RtMemRefList *(*entryPointFuncType)(RtMemRefList *);

class ExecutionSession {
public:
  ExecutionSession(std::string sharedLibPath, std::string entryPointName);

  // Use custom deleter since forward declared RtMemRef hides destructor
  std::vector<std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>> run(
      std::vector<std::unique_ptr<RtMemRef, decltype(&rmrDestroy)>>);

  ~ExecutionSession();

protected:
  // Handler to the shared library file being loaded.
  void *_sharedLibraryHandle = nullptr;

  // Entry point function.
  entryPointFuncType _entryPointFunc = nullptr;
};
} // namespace onnx_mlir
