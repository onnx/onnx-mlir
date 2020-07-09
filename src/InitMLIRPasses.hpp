#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

void initMLIRPasses() {
  mlir::registerPass("convert-vector-to-scf",
      "Decompose ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertVectorToSCFPass();
      });
  mlir::registerPass("lower-affine",
      "Decompose ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createLowerAffinePass();
      });
  mlir::registerPass("convert-scf-to-std",
      "Decompose ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createLowerToCFGPass();
      });
  mlir::registerPass("convert-vector-to-llvm",
      "Decompose ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertVectorToLLVMPass();
      });
}
} // namespace onnx_mlir