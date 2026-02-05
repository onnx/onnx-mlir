#ifndef ONNX_MLIR_TRANSFORMS_KRNL_PASSES
#define ONNX_MLIR_TRANSFORMS_KRNL_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerKrnlRegionPass : Pass<"lower-krnl-region", "func::FuncOp"> {
  let summary = "Move ops in krnl.region operation out and erase this op.";
  let description = [{
    Move ops in krnl.region operation out and erase this op.
  }];
}

#endif
