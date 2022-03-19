/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.cpp ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class.
//
// To add support for a new accelerator include its header file and provide a
// weak definition for the getInstance() member function.
//===----------------------------------------------------------------------===//

#include "src/Accelerators/Accelerator.hpp"
#include "src/Accelerators/NNPA/NNPAAccelerator.hpp"
#include "src/Support/Common.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nnpa"

namespace onnx_mlir {
namespace accel {

llvm::SmallPtrSet<Accelerator *, 2> Accelerator::accelerators;

// Provide a weak definition for the getInstance() member function. A strong
// definition (to override this one) must be provided by each concrete
// accelerator.
#define CREATE_WEAK_DEF(AcceleratorType)                                       \
  ATTRIBUTE(weak) Accelerator *AcceleratorType::getInstance() {                \
    LLVM_DEBUG(llvm::dbgs() << "Using weak definition for " #AcceleratorType   \
                               "::getInstance()\n";);                          \
    return nullptr;                                                            \
  }
CREATE_WEAK_DEF(nnpa::NNPAAccelerator)
#undef CREATE_WEAK_DEF

void Accelerator::create(Accelerator::Kind kind,
    mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &context,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType emissionTarget) {
  Accelerator *accel = nullptr;
  switch (kind) {
#define CREATE_ACCEL(ID, AcceleratorType)                                      \
  case Kind::ID:                                                               \
    accel = AcceleratorType::getInstance();                                    \
    if (!accel)                                                                \
      llvm::errs() << #ID " accelerator not supported\n";                      \
    break;
    CREATE_ACCEL(NNPA, nnpa::NNPAAccelerator)
#undef CREATE_ACCEL
  }
  assert(accel && "Accelerator not initialized correctly");

  // Initialize the new accelerator and add it to the list of available ones.
  accel->prepare(module, context, pm, emissionTarget);
  accelerators.insert(accel);
}

const llvm::SmallPtrSetImpl<Accelerator *> &Accelerator::getAccelerators() {
  return accelerators;
}

} // namespace accel
} // namespace onnx_mlir
