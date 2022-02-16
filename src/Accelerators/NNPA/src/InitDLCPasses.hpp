#include "mlir/Pass/Pass.h"

#include "src/Pass/DLCPasses.hpp"

namespace dlc {

void initDLCPasses(int optLevel) {
  // All passes implemented within DLC should register within this
  // function to make themselves available as a command-line option.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createONNXToZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createRewriteONNXForZHighPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZHighConstPropagationPass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZHighToZLowPass(optLevel);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZLowToLLVMPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFoldStdAllocPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createZHighLayoutPropagationPass();
  });
}
} // namespace dlc
