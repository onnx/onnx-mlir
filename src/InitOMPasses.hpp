#include "mlir/Pass/Pass.h"

#include "src/Pass/Passes.hpp"

namespace onnx_mlir {

void initOMPasses() {
  // All passes implemented within onnx-mlir should register within this
  // function to make themselves available as a command-line option.
  mlir::registerPass("decompose-onnx",
      "Decompose ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createDecomposeONNXToONNXPass();
      });

  mlir::registerPass("shape-inference",
      "Shape inference for frontend dialects.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createShapeInferencePass();
      });

  mlir::registerPass("constprop-onnx",
      "ConstProp ONNX operations into composition of other ONNX operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConstPropONNXToONNXPass();
      });

  mlir::registerPass("attribute-promotion",
      "Promote constant operands to attributes.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createAttributePromotionPass();
      });

  mlir::registerPass("elide-constants", "Elide values of constant operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createElideConstantValuePass();
      });

  mlir::registerPass("enable-memory-pool",
      "Enable a memory pool for allocating internal MemRefs.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createKrnlEnableMemoryPoolPass();
      });

  mlir::registerPass("bundle-memory-pools",
      "Bundle memory pools of internal MemRefs into a single memory pool.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createKrnlBundleMemoryPoolsPass();
      });

  mlir::registerPass("convert-krnl-to-affine", "Lower Krnl dialect.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertKrnlToAffinePass();
      });

  mlir::registerPass("convert-onnx-to-krnl",
      "Lower frontend ops to Krnl dialect.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createLowerToKrnlPass();
      });

  mlir::registerPass("elide-krnl-constants",
      "Elide the constant values of the Global Krnl operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createElideConstGlobalValuePass();
      });

  mlir::registerPass("convert-krnl-to-llvm",
      "Lower the Krnl Affine and Std dialects to LLVM.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createConvertKrnlToLLVMPass();
      });

  mlir::registerPass("pack-krnl-constants",
      "Elide the constant values of the Global Krnl operations.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createPackKrnlGlobalConstantsPass();
      });

  mlir::registerPass("disconnect-dims", "Disconnect dims from allocs.",
      []() -> std::unique_ptr<mlir::Pass> {
        return mlir::createDisconnectKrnlDimFromAllocPass();
      });
}
} // namespace onnx_mlir