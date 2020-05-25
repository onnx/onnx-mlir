//===- ElideKrnlGlobalConstants.cpp - Krnl Constant lobal Value Elision ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// In practice, the constant values of Global Krnl operations may be large
// enough to hinder the readability of the MLIR intermediate representation.
//
// This file creates a pass which elides the explicit values of constant
// global operations. This pass has purely cosmetic purposes and should only be
// run to obtain a compact representation of the program when emitting Krnl
// dialect code. This pass should never be invoked on code meant to be run.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

/*!
 *  Function pass that performs constant value elision of Krnl globals.
 */
class PackKrnlGlobalConstantsPass
    : public PassWrapper<PackKrnlGlobalConstantsPass, OperationPass<ModuleOp>> {
public:
  /// Make sure that we have a valid default constructor and copy constructor to
  /// make sure that the options are initialized properly.
  PackKrnlGlobalConstantsPass() = default;
  PackKrnlGlobalConstantsPass(const PackKrnlGlobalConstantsPass &pass) {}

  void runOnOperation() override {
    auto module = getOperation();
    module.walk([](KrnlGlobalOp op) { op.value().reset(); });
    if (move_to_file)
      printf("move to file!");
    else
      printf("not move to file!");
  }

  Option<bool> move_to_file{*this, "move-to-file",
      llvm::cl::desc("Whether to move the packed constant to a file."),
      llvm::cl::init(true)};
};
} // namespace

std::unique_ptr<Pass> mlir::createPackKrnlGlobalConstantsPass() {
  return std::make_unique<PackKrnlGlobalConstantsPass>();
}

static PassRegistration<PackKrnlGlobalConstantsPass> pass("pack-krnl-constants",
    "Elide the constant values of the Global Krnl operations.");