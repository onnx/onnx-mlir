// RUN: onnx-mlir-opt --enable-memory-pool %s -split-input-file | FileCheck %s

/// One intermediate value to allocate in the memory pool.
func @test_enable_memory_pool(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
    %0 = memref.alloc() : memref<10x10xf32>
    %1 = memref.alloc() : memref<10x10xf32>
    %2:2 = krnl.define_loops 2
    krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg1 = 0 to 10, %2#1 -> %arg2 = 0 to 10) {
      %4 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %5 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %6 = arith.addf %4, %5 : f32
      krnl.store %6, %1[%arg1, %arg2] : memref<10x10xf32>
    }
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg1 = 0 to 10, %3#1 -> %arg2 = 0 to 10) {
      %4 = krnl.load %1[%arg1, %arg2] : memref<10x10xf32>
      %5 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %6 = arith.addf %4, %5 : f32
      krnl.store %6, %0[%arg1, %arg2] : memref<10x10xf32>
    }
    memref.dealloc %1 : memref<10x10xf32>
    return %0 : memref<10x10xf32>
  
// CHECK-LABEL:  func @test_enable_memory_pool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<400xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.getref"([[RES_1_]], [[CST_0_]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[VAR_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 10){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_7_1_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_0_MEM_3_]] : f32
// CHECK:             krnl.store [[VAR_7_1_]], [[RES_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[RES_1_]] : memref<400xi8>
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

/// Two intermediate values to allocate in the memory pool.
func @test_enable_memory_pool_2(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() : memref<10x20xf32>
    %1 = memref.alloc() : memref<10x20xf32>
    %2 = memref.alloc() : memref<10x10xf32>
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg2 = 0 to 10, %3#1 -> %arg3 = 0 to 10) {
      %6 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %7 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %8 = arith.addf %6, %7 : f32
      krnl.store %8, %2[%arg2, %arg3] : memref<10x10xf32>
    }
    %4:2 = krnl.define_loops 2
    krnl.iterate(%4#0, %4#1) with (%4#0 -> %arg2 = 0 to 10, %4#1 -> %arg3 = 0 to 20) {
      %6 = memref.alloca() : memref<f32>
      krnl.store %cst, %6[] : memref<f32>
      %7 = krnl.define_loops 1
      krnl.iterate(%7) with (%7 -> %arg4 = 0 to 10) {
        %9 = krnl.load %2[%arg2, %arg4] : memref<10x10xf32>
        %10 = krnl.load %arg1[%arg4, %arg3] : memref<10x20xf32>
        %11 = krnl.load %6[] : memref<f32>
        %12 = arith.mulf %9, %10 : f32
        %13 = arith.addf %11, %12 : f32
        krnl.store %13, %6[] : memref<f32>
      }
      %8 = krnl.load %6[] : memref<f32>
      krnl.store %8, %1[%arg2, %arg3] : memref<10x20xf32>
    }
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg2 = 0 to 10, %5#1 -> %arg3 = 0 to 20) {
      %6 = krnl.load %1[%arg2, %arg3] : memref<10x20xf32>
      %7 = krnl.load %arg1[%arg2, %arg3] : memref<10x20xf32>
      %8 = arith.addf %6, %7 : f32
      krnl.store %8, %0[%arg2, %arg3] : memref<10x20xf32>
    }
    memref.dealloc %2 : memref<10x10xf32>
    memref.dealloc %1 : memref<10x20xf32>
    return %0 : memref<10x20xf32>
  
// CHECK-LABEL:  func @test_enable_memory_pool_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x20xf32>) -> memref<10x20xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<10x20xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<800xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.getref"([[RES_1_]], [[CST_0_]]) : (memref<800xi8>, i64) -> memref<10x20xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<400xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.getref"([[RES_2_]], [[CST_0_]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[VAR_4_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 20){
// CHECK:             [[RES_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_3_]][] : memref<f32>
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 10){
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_4_]], [[I_3_]]{{.}} : memref<10x20xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_14_:%.+]] = arith.mulf [[LOAD_VAR_4_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_3_MEM_]], [[VAR_14_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[RES_3_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_3_MEM_1_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_3_MEM_1_]], [[VAR_2_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x20xf32>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 10, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 20){
// CHECK-DAG:         [[RES_3_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK:             [[VAR_10_1_:%.+]] = arith.addf [[RES_3_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[RES_2_]] : memref<400xi8>
// CHECK:           memref.dealloc [[RES_1_]] : memref<800xi8>
// CHECK:           return [[RES_]] : memref<10x20xf32>
// CHECK:         }
}
