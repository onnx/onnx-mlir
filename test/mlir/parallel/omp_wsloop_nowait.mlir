// RUN: onnx-mlir-opt --omp-wsloop-nowait %s -split-input-file | FileCheck %s

// Wsloop directly followed by the terminator: the barrier is redundant.
func.func @nowait_terminator(%arg0 : memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @nowait_terminator
// CHECK:           omp.parallel {
// CHECK-NEXT:        omp.wsloop nowait {
// CHECK:             omp.terminator

// -----

// Wsloop followed by deallocs of per-thread buffers, as produced by the
// buffer-omploop-hoisting pass.
func.func @nowait_hoisted_dealloc(%arg0 : memref<1x?x768xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<1x?x768xf32>
  %c4 = arith.constant 4 : index
  omp.parallel {
    %alloc = memref.alloc() alignment = 16 : memref<4x16xf32>
    %alloc_0 = memref.alloc() alignment = 16 : memref<16xf32>
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%dim) step (%c4) {
        omp.yield
      }
    }
    memref.dealloc %alloc_0 : memref<16xf32>
    memref.dealloc %alloc : memref<4x16xf32>
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @nowait_hoisted_dealloc
// CHECK:           omp.parallel {
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:             omp.wsloop nowait {
// CHECK:             memref.dealloc [[RES_1_]] : memref<16xf32>
// CHECK:             memref.dealloc [[RES_]] : memref<4x16xf32>
// CHECK:             omp.terminator

// -----

// Wsloop followed by an effect-free op.
func.func @nowait_pure_op(%arg0 : memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    %c2 = arith.constant 2 : index
    %0 = arith.addi %c2, %c512 : index
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @nowait_pure_op
// CHECK:           omp.parallel {
// CHECK-NEXT:        omp.wsloop nowait {
// CHECK:             arith.addi
// CHECK:             omp.terminator

// -----

// Nested parallel regions: both the inner and the outer wsloop.
func.func @nowait_nested(%arg0 : memref<16x512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c16) step (%c1) {
        omp.parallel {
          omp.wsloop {
            omp.loop_nest (%j) : index = (%c0) to (%c512) step (%c1) {
              memref.store %cst, %arg0[%i, %j] : memref<16x512xf32>
              omp.yield
            }
          }
          omp.terminator
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @nowait_nested
// CHECK:           omp.parallel {
// CHECK-NEXT:        omp.wsloop nowait {
// CHECK:               omp.parallel {
// CHECK-NEXT:            omp.wsloop nowait {

// -----

// Wsloop followed by a load and a store of a shared buffer: keep the barrier.
func.func @barrier_shared_access(%arg0 : memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    %0 = memref.load %arg0[%c0] : memref<512xf32>
    memref.store %0, %arg0[%c1] : memref<512xf32>
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @barrier_shared_access
// CHECK-NOT:       nowait
// CHECK:           omp.wsloop {
// CHECK-NOT:       nowait
// CHECK:           memref.load
// CHECK-NOT:       nowait
// CHECK:           return

// -----

// Wsloop followed by the dealloc of a buffer from outside the region: keep
// the barrier.
func.func @barrier_shared_dealloc() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  %alloc = memref.alloc() : memref<512xf32>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %alloc[%i] : memref<512xf32>
        omp.yield
      }
    }
    memref.dealloc %alloc : memref<512xf32>
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @barrier_shared_dealloc
// CHECK-NOT:       nowait
// CHECK:           omp.wsloop {
// CHECK-NOT:       nowait
// CHECK:           memref.dealloc
// CHECK-NOT:       nowait
// CHECK:           return

// -----

// Wsloop with a reduction: keep the barrier.
omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

func.func @barrier_reduction(%arg0 : memref<512xf32>, %arg1 : memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %acc = llvm.mlir.constant(1 : i64) : i64
  %sum = llvm.alloca %acc x f32 : (i64) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %sum -> %prv : !llvm.ptr) {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        %0 = memref.load %arg0[%i] : memref<512xf32>
        %1 = llvm.load %prv : !llvm.ptr -> f32
        %2 = arith.addf %0, %1 : f32
        llvm.store %2, %prv : f32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @barrier_reduction
// CHECK-NOT:       nowait
// CHECK:           omp.wsloop reduction(
// CHECK-NOT:       nowait
// CHECK:           return

// -----

// Two wsloops in one region: the first keeps its barrier, as the second may
// read what the first wrote; the second gets nowait.
func.func @two_wsloops(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c511 = arith.constant 511 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        %j = arith.subi %c511, %i : index
        %0 = memref.load %arg0[%j] : memref<512xf32>
        memref.store %0, %arg1[%i] : memref<512xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @two_wsloops
// CHECK:           omp.parallel {
// CHECK-NEXT:        omp.wsloop {
// CHECK:             omp.wsloop nowait {
// CHECK:             omp.terminator

// -----

// Wsloop followed by an explicit barrier or by a call: keep the barrier.
func.func private @some_func()

func.func @barrier_explicit_or_call(%arg0 : memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.0 : f32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    omp.barrier
    omp.terminator
  }
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c512) step (%c1) {
        memref.store %cst, %arg0[%i] : memref<512xf32>
        omp.yield
      }
    }
    func.call @some_func() : () -> ()
    omp.terminator
  }
  return
}
// CHECK-LABEL:  func.func @barrier_explicit_or_call
// CHECK-NOT:       nowait
// CHECK:           omp.wsloop {
// CHECK-NOT:       nowait
// CHECK:           omp.barrier
// CHECK-NOT:       nowait
// CHECK:           omp.wsloop {
// CHECK-NOT:       nowait
// CHECK:           func.call @some_func
// CHECK-NOT:       nowait
// CHECK:           return
