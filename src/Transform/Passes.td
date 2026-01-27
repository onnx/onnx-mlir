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
  let options = [
    Option<"instrumentOps", "instrument-ops", "::std::string",
            /*default=*/"\"\"", [{Specify regex for ops to be instrumented:\n"
                                "\"NONE\" or \"\" for no instrument,\n"
                                "\"regex1,regex2, ...\" for the specified ops.\n"
                                "e.g. \"onnx.,zhigh.\" for onnx and zhigh ops.\n"
                                "e.g. \"onnx.Conv\" for onnx Conv ops.\n}]>,

    Option<"instrumentBefore", "instrument-before", "bool",
            /*default=*/"false", "insert instrument before op">,

    Option<"instrumentAfter", "instrument-after", "bool",
            /*default=*/"false", "insert instrument after op">,

    Option<"reportTime", "report-time", "bool",
            /*default=*/"false", "instrument runtime reports time usage">,

    Option<"reportMemory", "report-memory", "bool",
            /*default=*/"false", "instrument runtime reports memory usage">

  ];
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


#endif
