// RUN: onnx-mlir-opt --buffer-omploop-hoisting %s -split-input-file | FileCheck %s

// -----

func.func @omploop_hoist_basic() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c512 = arith.constant 512 : index
  
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%c512) step (%c4) {
          %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<4x16xf32>
          memref.dealloc %alloc_6 : memref<4x16xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoist_basic
// CHECK-SAME:   () {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK:           omp.parallel {
// CHECK:             [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[arg0_:%.+]]) : index = ([[CST_0_]]) to ([[CST_512_]]) step ([[CST_4_]]) {
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             memref.dealloc [[RES_]] : memref<4x16xf32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func.func @omploop_hoist_multiple(%arg0 : memref<1x?x768xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<1x?x768xf32>
  %c4 = arith.constant 4 : index
  %0 = llvm.mlir.constant(1 : i64) : i64
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%dim) step (%c4) {
          %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<4x16xf32>
          %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<16xf32>
          memref.dealloc %alloc_7 : memref<16xf32>
          memref.dealloc %alloc_6 : memref<4x16xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoist_multiple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x768xf32>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x768xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           omp.parallel {
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[ARG1_:%.+]]) : index = ([[CST_0_]]) to ([[VAR_dim_]]) step ([[CST_4_]]) {
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             memref.dealloc [[RES_1_]] : memref<16xf32>
// CHECK:             memref.dealloc [[RES_]] : memref<4x16xf32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
}

// -----

func.func @omploop_hoist_defined_inside(%arg0 : memref<1x?x768xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<1x?x768xf32>
  %c4 = arith.constant 4 : index
  %0 = llvm.mlir.constant(1 : i64) : i64
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%dim) step (%c4) {
          %alloc_6 = memref.alloc(%arg3) : memref<?x16xf32>
          memref.dealloc %alloc_6 : memref<?x16xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoist_defined_inside
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x768xf32>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x768xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           omp.parallel {
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[ARG1_:%.+]]) : index = ([[CST_0_]]) to ([[VAR_dim_]]) step ([[CST_4_]]) {
// CHECK:                 [[RES_:%.+]] = memref.alloc([[ARG1_]]) : memref<?x16xf32>
// CHECK:                 memref.dealloc [[RES_]] : memref<?x16xf32>
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
}


// -----

func.func @omploop_hoisting_defined_outside(%arg0 : memref<1x?x768xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<1x?x768xf32>
  %c4 = arith.constant 4 : index
  %0 = llvm.mlir.constant(1 : i64) : i64
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%dim) step (%c4) {
          %alloc_6 = memref.alloc(%dim) : memref<?x16xf32>
          memref.dealloc %alloc_6 : memref<?x16xf32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoisting_defined_outside
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x768xf32>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<1x?x768xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           omp.parallel {
// CHECK:             [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) : memref<?x16xf32>
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[arg1_:%.+]]) : index = ([[CST_0_]]) to ([[VAR_dim_]]) step ([[CST_4_]]) {
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             memref.dealloc [[RES_]] : memref<?x16xf32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
}


// -----

func.func @omploop_hoist_check_dealloc() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c512 = arith.constant 512 : index
  
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%c512) step (%c4) {
          %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<4x16xf32>
          %3 = arith.cmpi slt, %c4, %c0 : index
          scf.if %3 {
            memref.dealloc %alloc_6 : memref<4x16xf32>
          }
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoist_check_dealloc
// CHECK-SAME:   () {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK:           omp.parallel {
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[arg0_]]) : index = ([[CST_0_]]) to ([[CST_512_]]) step ([[CST_4_]]) {
// CHECK-DAG:             [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK-DAG:             [[VAR_0_:%.+]] = arith.cmpi slt, [[CST_4_]], [[CST_0_]] : index
// CHECK:                 scf.if [[VAR_0_]] {
// CHECK:                   memref.dealloc [[RES_]] : memref<4x16xf32>
// CHECK:                 }
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
}

// -----


func.func @omploop_hoist_alloca() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c512 = arith.constant 512 : index
  
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg3) : index = (%c0) to (%c512) step (%c4) {
        memref.alloca_scope {
          %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<4x16xf32>
          memref.dealloc %alloc_6 : memref<4x16xf32>
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return
// CHECK-LABEL:  func.func @omploop_hoist_alloca
// CHECK-SAME:   () {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK:           omp.parallel {
// CHECK:             [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest ([[arg0_]]) : index = ([[CST_0_]]) to ([[CST_512_]]) step ([[CST_4_]]) {
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             memref.dealloc [[RES_]] : memref<4x16xf32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
}
