/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- InspectSo.cpp - Inspect compiled .so model files ----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a utility that loads a compiled .so model file produced
// by onnx-mlir and prints its metadata: entry points, input/output signatures,
// and compilation information.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>

using queryEntryPointsFuncType = const char *const *(*)(int64_t *);
using signatureFuncType = const char *(*)(const char *);
using compilationInfoFuncType = const char *(*)();

int main(int argc, char **argv) {
  if (argc != 2 || std::strcmp(argv[1], "--help") == 0 ||
      std::strcmp(argv[1], "-h") == 0) {
    std::cerr << "Usage: " << argv[0] << " <input .so file>\n";
    return (argc != 2) ? 1 : 0;
  }

  std::string filename = argv[1];

  void *handle = dlopen(filename.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    std::cerr << "Error: cannot open '" << filename << "': " << dlerror()
              << "\n";
    return 1;
  }

  auto queryEntryPointsFunc = reinterpret_cast<queryEntryPointsFuncType>(
      dlsym(handle, "omQueryEntryPoints"));
  if (!queryEntryPointsFunc) {
    std::cerr << "Error: cannot find omQueryEntryPoints symbol.\n";
    dlclose(handle);
    return 1;
  }

  auto inputSignatureFunc =
      reinterpret_cast<signatureFuncType>(dlsym(handle, "omInputSignature"));
  auto outputSignatureFunc =
      reinterpret_cast<signatureFuncType>(dlsym(handle, "omOutputSignature"));
  auto compilationInfoFunc = reinterpret_cast<compilationInfoFuncType>(
      dlsym(handle, "omCompilationInfo"));

  // Query and print entry points.
  int64_t numEntryPoints = 0;
  const char *const *entryPoints = queryEntryPointsFunc(&numEntryPoints);

  std::cout << "Entry points (" << numEntryPoints << "):\n";
  for (int64_t i = 0; i < numEntryPoints; ++i) {
    std::cout << "  " << (i + 1) << ". " << entryPoints[i] << "\n";

    if (inputSignatureFunc) {
      const char *sig = inputSignatureFunc(entryPoints[i]);
      std::cout << "\n     Input signature:\n";
      std::cout << "       " << (sig ? sig : "(null)") << "\n";
    }

    if (outputSignatureFunc) {
      const char *sig = outputSignatureFunc(entryPoints[i]);
      std::cout << "\n     Output signature:\n";
      std::cout << "       " << (sig ? sig : "(null)") << "\n";
    }

    std::cout << "\n";
  }

  // Print compilation info.
  if (compilationInfoFunc) {
    const char *info = compilationInfoFunc();
    std::cout << "Compilation info:\n";
    std::cout << "  " << (info ? info : "(null)") << "\n";
  }

  dlclose(handle);
  return 0;
}
