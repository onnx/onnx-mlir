#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

void initMLIRPasses() {
  mlir::registerPass("convert-vector-to-scf",
      "Convert vector to SCF.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertVectorToSCFPass();
      });
  mlir::registerPass("lower-affine",
      "Lower Affine Dialect to Standard Dialect.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createLowerAffinePass();
      });
  mlir::registerPass("convert-scf-to-std",
      "Lower SCF to Standard Dialect.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createLowerToCFGPass();
      });
  mlir::registerPass("convert-vector-to-llvm",
      "Lower Vector Dialect to LLVM IR Dialect.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertVectorToLLVMPass();
      });
}
} // namespace onnx_mlir