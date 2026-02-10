#ifndef ONNX_MLIR_TRANSFORMS_PASSES
#define ONNX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ProcessKrnlParallelClausePass : Pass<"process-krnl-parallel-clause", "func::FuncOp"> {
  let summary = "Migrate info from Krnl Parallel Clause into OpenMP Parallel operation.";
  let description = [{
    Migrate info from Krnl Parallel Clause into OpenMP Parallel operation.
  }];
}

def ProcessScfParallelPrivatePass : Pass<"scf-parallel-private", "func::FuncOp"> {
  let summary = "Process scf parallel for op to support private variables.";
  let description = [{
    Process scf parallel for op to support private variables.
  }];
}

def InstrumentPass : Pass<"instrument", "func::FuncOp"> {
  let summary = "instrument on ops.";
  let description = [{
    instrument on ops.
  }];
}

def InstrumentCleanupPass : Pass<"instrument-cleanup", "func::FuncOp"> {
  let summary = "instrument cleanup on ops.";
  let description = [{
    instrument cleanup on ops.
  }];
}

def BufferOMPLoopHoistingPass : Pass<"buffer-omploop-hoisting", "func::FuncOp"> {
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

def ConvertAttrToDiscardable : Pass<"convert-attr-to-discardable"> {
  let summary = "Convert attributes to discardable form";
  let description = [{
    This pass converts specified attributes to discardable form by
    prefixing them with '_.', based on the provided attribute names.
  }];

  let options = [
    ListOption<"attrNames", "attr-names", "std::string",
               "List of attribute names to mark as discardable">
  ];
}

#endif
