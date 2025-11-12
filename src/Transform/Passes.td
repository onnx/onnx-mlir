#ifndef ONNX_MLIR_TRANSFORMS_PASSES
#define ONNX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def BufferOMPLoopHoisting : Pass<"buffer-omploop-hoisting", "func::FuncOp"> {
  let summary = "Hoist buffer(alloc and dealloc) for omp loops";
  let description = [{
    This pass hoists buffer alloc/dealloc from omp worksharing loop
    to the omp.parallel, so that the buffer is created for each thread in 
    the parallel struct only once.
    Example from:

    omp.parallel {
      omp.wsloop {
        omp.loop_nest {
          %t = memref.alloc();
          ...
          memref.dealloc(%t);
        }
      }
    }

    to:

    omp.parallel {
      %t = memref.alloc();
      omp.wsloop {
        omp.loop_nest {
          ...
        }
      }
      memref.dealloc(%t);
    }

    This function is similar to the BufferLoopHoistingPass, but for 
    omp loops with alloc and dealloc. The wsloop is the destination place.
    It is assumed that bufferization::buildBufferDeallocationPipeline
    and createConvertBufferizationToMemRefPass() have been called.
  }];
}

#endif
