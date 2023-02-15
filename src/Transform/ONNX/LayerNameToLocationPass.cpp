#include "src/Dialect/ONNX/ONNXOps.hpp"
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
struct LayerNameToLocationPass
    : public PassWrapper<LayerNameToLocationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayerNameToLocationPass)

  LayerNameToLocationPass() = default;
  LayerNameToLocationPass(const LayerNameToLocationPass &pass) = default;

  [[nodiscard]] StringRef getArgument() const override {
    return "onnx-layer-name-location";
  }

  [[nodiscard]] StringRef getDescription() const override {
    return "Inserts the ONNX layer name of each operation into the location "
           "info associated with it.";
  }

  void runOnOperation() final;
};

void LayerNameToLocationPass::runOnOperation() {
  // Check whether op has a onnx_node_name attribute and put that on the
  // existing location info.
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();

  // Counter and names to make invalid locations unique
  unsigned invLocSeq = 0;
  StringRef invLocName = "INVALID:FXML-1477:";

  op->walk([&](Operation *nestedOp) {
    Location loc = nestedOp->getLoc();
    NameLoc nameLoc;

    // Extend the existing location with the layer name, if available
    if (auto layerName =
            nestedOp->getAttrOfType<StringAttr>("onnx_node_name")) {
      nameLoc = NameLoc::get(layerName, loc);
    }

    // All onnx ops (except constants) are expected to have an onnx_node_name.
    // If onnx_node_name is not available and this is an onnx operation, extend
    // the location with a name to indicate a missing location.
    else if (isa_and_present<ONNXDialect>(nestedOp->getDialect()) &&
             !isa<ONNXConstantOp>(nestedOp) &&
             !isa<ONNXEntryPointOp>(nestedOp)) {
      auto invalidName =
          StringAttr::get(context, invLocName + std::to_string(invLocSeq++));
      nameLoc = NameLoc::get(invalidName, loc);
    }

    // Do not modify location if we ignored the op
    if (nameLoc)
      nestedOp->setLoc(nameLoc);
  });
}

} // namespace

std::unique_ptr<Pass> createLayerNameToLocationPass() {
  return std::make_unique<LayerNameToLocationPass>();
}
} // namespace onnx_mlir
