#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerDialects.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace onnx_mlir;

std::vector<accel::Accelerator::Kind> maccel;
llvm::cl::OptionCategory OnnxMlirLspOptions("Accelerator Options");

static llvm::cl::list<accel::Accelerator::Kind,
    std::vector<accel::Accelerator::Kind>>
    maccelOpt("maccel",
        llvm::cl::desc("Specify an accelerator for onnx-lsp-server"),
        llvm::cl::location(maccel),
        // clang-format off
        llvm::cl::values(
          APPLY_TO_ACCELERATORS(CREATE_ACCEL_CL_ENUM)
          clEnumValN(accel::Accelerator::Kind::NONE, "NONE", "No accelerator")
        ),
        // clang-format on
        llvm::cl::cat(OnnxMlirLspOptions), llvm::cl::ValueRequired);

int main(int argc, char **argv) {

  auto registry = onnx_mlir::registerDialects(maccel);

  return failed(MlirLspServerMain(argc, argv, registry));
}