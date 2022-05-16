/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class.
//
// To enable a new accelerator, add the header include, an extern of the
// subclass and pushback that subclass variable onto acceleratorTargets.
//===----------------------------------------------------------------------===//

#include <map>

#include "src/Accelerators/Accelerator.hpp"

namespace onnx_mlir {
namespace accel {

llvm::SmallVector<Accelerator *, 4> Accelerator::acceleratorTargets;

const llvm::SmallVectorImpl<Accelerator *> &Accelerator::getAccelerators() {
  assert(acceleratorTargets.size() <= 1 &&
         "Only support at most one accelerator at this moment");
  return acceleratorTargets;
}

// Help to print accelerator kinds.
static std::map<Accelerator::Kind, std::string> mapKind2Strings;

std::ostream &operator<<(std::ostream &out, const Accelerator::Kind kind) {
  if (mapKind2Strings.empty()) {
    APPLY_TO_ACCELERATORS(ACCEL_CL_ENUM_TO_STRING, mapKind2Strings);
    mapKind2Strings[Accelerator::Kind::NONE] = "NONE";
  }
  return out << mapKind2Strings[kind];
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &out, const Accelerator::Kind kind) {
  if (mapKind2Strings.empty()) {
    APPLY_TO_ACCELERATORS(ACCEL_CL_ENUM_TO_STRING, mapKind2Strings);
    mapKind2Strings[Accelerator::Kind::NONE] = "NONE";
  }
  return out << mapKind2Strings[kind];
}

} // namespace accel
} // namespace onnx_mlir
