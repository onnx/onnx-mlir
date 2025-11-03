#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_BUFFEROMPLOOPHOISTING
#include "src/Transform/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct BufferOMPLoopHoistingPass
    : public impl::BufferOMPLoopHoistingBase<BufferOMPLoopHoistingPass> {
  void runOnOperation() override;
};
} // namespace

void BufferOMPLoopHoistingPass::runOnOperation() {}

namespace onnx_mlir {
std::unique_ptr<Pass> createBufferOMPLoopHoisting() {
  return std::make_unique<BufferOMPLoopHoistingPass>();
};
} // namespace onnx_mlir
