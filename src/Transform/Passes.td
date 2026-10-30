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

def OMPWsloopNowaitPass : Pass<"omp-wsloop-nowait", "func::FuncOp"> {
  let summary = "Remove the redundant barrier of the last omp.wsloop of a parallel region";
  let description = [{
    This pass sets the nowait clause on an omp.wsloop that is directly in an
    omp.parallel region when only per-thread code separates the wsloop from
    the end of the region. The implicit barrier at the end of such a loop is
    redundant with the join at the end of the parallel region.
    Example:

    omp.parallel {
      %t = memref.alloc();
      omp.wsloop {
        omp.loop_nest { ... }
      }
      memref.dealloc(%t);
      omp.terminator
    }

    gets "omp.wsloop nowait". Per-thread code is an op with no memory
    effects, or a memref.dealloc of a buffer allocated in the same parallel
    region. Any other op after the wsloop, or a wsloop with reduction,
    private, or linear variables, keeps the barrier.
    It is assumed that this pass runs after buffer-omploop-hoisting, and that
    no later pass adds code between the wsloop and the end of the region.
  }];
}

def WriteOpStatsToModuleAttributePass : Pass<"write-op-stats-to-module-attribute"> {
  let summary = "Write statistics of operations to an attribute in ModuleOp";
}

def ONNXCSEWithNodeNamePass : Pass<"onnx-cse-with-node-name", "ModuleOp"> {
  let summary = "CSE for ONNX operations preserving node names";
  let description = [{
    Performs Common Subexpression Elimination on ONNX operations while
    preserving onnx_node_name attributes. The pass:
    1. Saves onnx_node_name attributes
    2. Removes them temporarily to enable CSE
    3. Applies MLIR CSE pass
    4. Restores node names (keeping first occurrence's name for merged ops)
    
    This pass only operates on ONNX dialect operations.
  }];
}

#endif
