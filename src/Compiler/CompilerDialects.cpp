/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerDialects.cpp ------------------------===//

#include "CompilerDialects.hpp"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

using namespace mlir;

namespace onnx_mlir {

DialectRegistry registerDialects(ArrayRef<accel::Accelerator::Kind> accels) {
  DialectRegistry registry;

  // Note that we cannot consult command line options because they have not yet
  // been parsed when registerDialects() is called.

  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<shape::ShapeDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<ONNXDialect>();
  registry.insert<KrnlDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registerOpenMPDialectTranslation(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);

  // Initialize accelerator(s) if required.
  accel::initAccelerators(accels);

  // Register dialects for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerDialects(registry);

  // Register interface needed by both old and new buffer deallocation pass.
  memref::registerAllocationOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);

  return registry;
}

} // namespace onnx_mlir
