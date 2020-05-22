//===--------- ExecusionSession.hpp - ExecutionSession Declaration --------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of ExecusionSession class, which helps C++
// programs interact with compiled binary model libraries.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <dlfcn.h>
#include <string>

#include "src/Runtime/DynMemRef.h"

namespace onnx_mlir {

typedef OrderedDynMemRefDict *(*entryPointFuncType)(OrderedDynMemRefDict *);

class ExecutionSession {
public:
  ExecutionSession(std::string sharedLibPath, std::string entryPointName);

  std::vector<std::unique_ptr<DynMemRef>> run(
      std::vector<std::unique_ptr<DynMemRef>>);

  ~ExecutionSession();

protected:
  // Handler to the shared library file being loaded.
  void *_sharedLibraryHandle = nullptr;

  // Entry point function.
  entryPointFuncType _entryPointFunc = nullptr;
};
} // namespace onnx_mlir