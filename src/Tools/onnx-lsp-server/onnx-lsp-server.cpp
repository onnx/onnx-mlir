#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
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
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();

  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::KrnlDialect>();

  return failed(MlirLspServerMain(argc, argv, registry));
}