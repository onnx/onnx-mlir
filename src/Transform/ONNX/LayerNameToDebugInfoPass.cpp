#include "src/Pass/Passes.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include <memory>

using namespace mlir;
using namespace llvm;

namespace onnx_mlir {
namespace {
struct LayerNameToDebugInfoPass
    : public PassWrapper<LayerNameToDebugInfoPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayerNameToDebugInfoPass)

  LayerNameToDebugInfoPass() = default;
  LayerNameToDebugInfoPass(const LayerNameToDebugInfoPass &pass) = default;

  [[nodiscard]] StringRef getArgument() const override {
    return "onnx-layer-name-debug-info";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Inserts the ONNX layer name of each operation into the debug info "
           "associated with it.";
  }

  void runOnOperation() final;
};

void LayerNameToDebugInfoPass::runOnOperation() {
  // Check whether op has a onnx_node_name attribute and put that on the
  // existing debug info.
  Operation *op = getOperation();

  op->walk([](Operation *nestedOp) {
    if (auto layerName =
            nestedOp->getAttrOfType<StringAttr>("onnx_node_name")) {
      Location loc = nestedOp->getLoc();
      auto nameLoc = NameLoc::get(layerName, loc);
      nestedOp->setLoc(nameLoc);
    }
  });
}

} // namespace

std::unique_ptr<Pass> createLayerNameToDebugInfoPass() {
  return std::make_unique<LayerNameToDebugInfoPass>();
}
} // namespace onnx_mlir
