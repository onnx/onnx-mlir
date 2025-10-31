#ifndef ONNX_MLIR_TRANSFORMS_PASSES
#define ONNX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def BufferOMPLoopHoisting : Pass<"buffer-omploop-hoisting", "func::FuncOp"> {
  let summary = "Hoist buffer(alloc and dealloc) for omp loops";
  let description = [{
    This pass hoists buffer alloc/dealloc from omp worksharing loop
    to the omp.parallel, so that the buffer is created for the 
    parallel struct only once.
    This function is similar to the BufferLoopHoistingPass, but for 
    omp loops.
    It is assumed that bufferization::buildBufferDeallocationPipeline
    and createConvertBufferizationToMemRefPass() have been called.
  }];
}

#endif
