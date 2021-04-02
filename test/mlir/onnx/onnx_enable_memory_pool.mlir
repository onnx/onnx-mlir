// RUN: onnx-mlir-opt --enable-memory-pool %s -split-input-file | FileCheck %s

/// One intermediate value to allocate in the memory pool.
func @test_enable_memory_pool(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
    %0 = alloc() : memref<10x10xf32>
    %1 = alloc() : memref<10x10xf32>
    %2:2 = krnl.define_loops 2
    krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg1 = 0 to 10, %2#1 -> %arg2 = 0 to 10) {
      %4 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %5 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %6 = addf %4, %5 : f32
      krnl.store %6, %1[%arg1, %arg2] : memref<10x10xf32>
    }
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg1 = 0 to 10, %3#1 -> %arg2 = 0 to 10) {
      %4 = krnl.load %1[%arg1, %arg2] : memref<10x10xf32>
      %5 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
      %6 = addf %4, %5 : f32
      krnl.store %6, %0[%arg1, %arg2] : memref<10x10xf32>
    }
    dealloc %1 : memref<10x10xf32>
    return %0 : memref<10x10xf32>
  
// CHECK-LABEL:  func @test_enable_memory_pool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : i64
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = alloc() : memref<400xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.getref"([[RES_1_]], [[CST_0_]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_7_:%.+]] = addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[VAR_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 10) {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_3_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_7_1_:%.+]] = addf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_0_MEM_3_]] : f32
// CHECK:             krnl.store [[VAR_7_1_]], [[RES_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           dealloc [[RES_1_]] : memref<400xi8>
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

/// Two intermediate values to allocate in the memory pool.
func @test_enable_memory_pool_2(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
    %cst = constant 0.000000e+00 : f32
    %0 = alloc() : memref<10x20xf32>
    %1 = alloc() : memref<10x20xf32>
    %2 = alloc() : memref<10x10xf32>
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg2 = 0 to 10, %3#1 -> %arg3 = 0 to 10) {
      %6 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %7 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %8 = addf %6, %7 : f32
      krnl.store %8, %2[%arg2, %arg3] : memref<10x10xf32>
    }
    %4:2 = krnl.define_loops 2
    krnl.iterate(%4#0, %4#1) with (%4#0 -> %arg2 = 0 to 10, %4#1 -> %arg3 = 0 to 20) {
      %6 = alloca() : memref<f32>
      krnl.store %cst, %6[] : memref<f32>
      %7 = krnl.define_loops 1
      krnl.iterate(%7) with (%7 -> %arg4 = 0 to 10) {
        %9 = krnl.load %2[%arg2, %arg4] : memref<10x10xf32>
        %10 = krnl.load %arg1[%arg4, %arg3] : memref<10x20xf32>
        %11 = krnl.load %6[] : memref<f32>
        %12 = mulf %9, %10 : f32
        %13 = addf %11, %12 : f32
        krnl.store %13, %6[] : memref<f32>
      }
      %8 = krnl.load %6[] : memref<f32>
      krnl.store %8, %1[%arg2, %arg3] : memref<10x20xf32>
    }
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg2 = 0 to 10, %5#1 -> %arg3 = 0 to 20) {
      %6 = krnl.load %1[%arg2, %arg3] : memref<10x20xf32>
      %7 = krnl.load %arg1[%arg2, %arg3] : memref<10x20xf32>
      %8 = addf %6, %7 : f32
      krnl.store %8, %0[%arg2, %arg3] : memref<10x20xf32>
    }
    dealloc %2 : memref<10x10xf32>
    dealloc %1 : memref<10x20xf32>
    return %0 : memref<10x20xf32>
  
// CHECK-LABEL:  func @test_enable_memory_pool_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x10xf32>, [[PARAM_1_:%.+]]: memref<10x20xf32>) -> memref<10x20xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : i64
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<10x20xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = alloc() : memref<800xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.getref"([[RES_1_]], [[CST_0_]]) : (memref<800xi8>, i64) -> memref<10x20xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = alloc() : memref<400xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.getref"([[RES_2_]], [[CST_0_]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:             [[VAR_10_:%.+]] = addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[VAR_4_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 20) {
// CHECK:             [[RES_3_:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_3_]][] : memref<f32>
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 10) {
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<10x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_4_]], [[I_3_]]{{.}} : memref<10x20xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_14_:%.+]] = mulf [[LOAD_VAR_4_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_15_:%.+]] = addf [[LOAD_RES_3_MEM_]], [[VAR_14_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[RES_3_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_3_MEM_1_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_3_MEM_1_]], [[VAR_2_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<10x20xf32>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 10, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 20) {
// CHECK-DAG:         [[RES_3_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK:             [[VAR_10_1_:%.+]] = addf [[RES_3_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<10x20xf32>
// CHECK:           }
// CHECK:           dealloc [[RES_2_]] : memref<400xi8>
// CHECK:           dealloc [[RES_1_]] : memref<800xi8>
// CHECK:           return [[RES_]] : memref<10x20xf32>
// CHECK:         }
}

// -----

// Two intermediate dynamic sized MemRefs.
#map = affine_map<()[s0] -> (s0, s0)>
  func @test_enable_memory_pool_3(%arg0: memref<?x?xf32>, %arg1: memref<?x10xf32>, %arg2: memref<10x10xf32>) -> memref<?x10xf32> {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %0 = dim %arg0, %c0 : memref<?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?xf32>
    %2 = alloc(%0) : memref<?x10xf32>
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg3 = 0 to %0, %3#1 -> %arg4 = 0 to 10) {
      %9 = alloca() : memref<f32>
      krnl.store %cst, %9[] : memref<f32>
      %10 = krnl.define_loops 1
      krnl.iterate(%10) with (%10 -> %arg5 = 0 to %1) {
        %12 = krnl.load %arg0[%arg3, %arg5] : memref<?x?xf32>
        %13 = krnl.load %arg1[%arg5, %arg4] : memref<?x10xf32>
        %14 = krnl.load %9[] : memref<f32>
        %15 = mulf %12, %13 : f32
        %16 = addf %14, %15 : f32
        krnl.store %16, %9[] : memref<f32>
      }
      %11 = krnl.load %9[] : memref<f32>
      krnl.store %11, %2[%arg3, %arg4] : memref<?x10xf32>
    }
    %4 = affine.max #map()[%0]
    %5 = alloc(%4) : memref<?x10xf32>
    %6:2 = krnl.define_loops 2
    krnl.iterate(%6#0, %6#1) with (%6#0 -> %arg3 = 0 to %4, %6#1 -> %arg4 = 0 to 10) {
      %9 = cmpi sgt, %0, %c1 : index
      %10 = select %9, %arg3, %c0 : index
      %11 = krnl.load %2[%10, %arg4] : memref<?x10xf32>
      %12 = cmpi sgt, %0, %c1 : index
      %13 = select %12, %arg3, %c0 : index
      %14 = krnl.load %2[%13, %arg4] : memref<?x10xf32>
      %15 = addf %11, %14 : f32
      krnl.store %15, %5[%arg3, %arg4] : memref<?x10xf32>
    }
    %7 = alloc(%0) : memref<?x10xf32>
    %8:2 = krnl.define_loops 2
    krnl.iterate(%8#0, %8#1) with (%8#0 -> %arg3 = 0 to %0, %8#1 -> %arg4 = 0 to 10) {
      %9 = alloca() : memref<f32>
      krnl.store %cst, %9[] : memref<f32>
      %10 = krnl.define_loops 1
      krnl.iterate(%10) with (%10 -> %arg5 = 0 to 10) {
        %12 = krnl.load %2[%arg3, %arg5] : memref<?x10xf32>
        %13 = krnl.load %5[%arg5, %arg4] : memref<?x10xf32>
        %14 = krnl.load %9[] : memref<f32>
        %15 = mulf %12, %13 : f32
        %16 = addf %14, %15 : f32
        krnl.store %16, %9[] : memref<f32>
      }
      %11 = krnl.load %9[] : memref<f32>
      krnl.store %11, %7[%arg3, %arg4] : memref<?x10xf32>
    }
    dealloc %2 : memref<?x10xf32>
    dealloc %5 : memref<?x10xf32>
    return %7 : memref<?x10xf32>
  
// CHECK-DAG: #map = affine_map<()[s0] -> (s0, s0)>
// CHECK-LABEL:  func @test_enable_memory_pool_3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>, [[PARAM_1_:%.+]]: memref<?x10xf32>, [[PARAM_2_:%.+]]: memref<10x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_10_:%.+]] = constant 10 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = constant 0 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK:           [[VAR_2_:%.+]] = muli [[DIM_0_]], [[CST_4_]] : index
// CHECK:           [[VAR_3_:%.+]] = muli [[VAR_2_]], [[CST_10_]] : index
// CHECK:           [[RES_:%.+]] = alloc([[VAR_3_]]) : memref<?xi8>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.getref"([[RES_]], [[CST_0_1_]], [[DIM_0_]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[DIM_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
// CHECK:             [[RES_1_:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[DIM_1_]]) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<?x?xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_21_:%.+]] = mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               [[VAR_22_:%.+]] = addf [[LOAD_RES_1_MEM_]], [[VAR_21_]] : f32
// CHECK:               krnl.store [[VAR_22_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[VAR_5_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x10xf32>
// CHECK:           }
// CHECK:           [[VAR_7_:%.+]] = affine.max #map(){{.}}[[DIM_0_]]{{.}}
// CHECK:           [[VAR_8_:%.+]] = muli [[VAR_7_]], [[CST_4_]] : index
// CHECK:           [[VAR_9_:%.+]] = muli [[VAR_8_]], [[CST_10_]] : index
// CHECK:           [[RES_2_:%.+]] = alloc([[VAR_9_]]) : memref<?xi8>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.getref"([[RES_2_]], [[CST_0_1_]], [[VAR_7_]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to [[VAR_7_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 10) {
// CHECK:             [[RES_1_:%.+]] = cmpi sgt, [[DIM_0_]], [[CST_1_]] : index
// CHECK:             [[LOOP_1_:%.+]] = select [[RES_1_]], [[I_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[LOOP_1_]], [[I_4_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = cmpi sgt, [[DIM_0_]], [[CST_1_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = select [[LOAD_PARAM_0_MEM_1_]], [[I_3_]], [[CST_0_]] : index
// CHECK:             [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[I_4_]]{{.}} : memref<?x10xf32>
// CHECK:             [[VAR_21_1_:%.+]] = addf [[LOAD_RES_1_MEM_1_]], [[LOAD_RES_1_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_21_1_]], [[VAR_11_]]{{.}}[[I_3_]], [[I_4_]]{{.}} : memref<?x10xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = alloc([[DIM_0_]]) : memref<?x10xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to [[DIM_0_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 10) {
// CHECK:             [[RES_4_:%.+]] = alloca() : memref<f32>
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_4_]][] : memref<f32>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 10) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[I_5_]], [[I_7_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[I_7_]], [[I_6_]]{{.}} : memref<?x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_21_2_:%.+]] = mulf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:               [[VAR_22_1_:%.+]] = addf [[LOAD_RES_1_MEM_2_]], [[VAR_21_2_]] : f32
// CHECK:               krnl.store [[VAR_22_1_]], [[RES_4_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_1_]], [[RES_3_]]{{.}}[[I_5_]], [[I_6_]]{{.}} : memref<?x10xf32>
// CHECK:           }
// CHECK:           dealloc [[RES_2_]] : memref<?xi8>
// CHECK:           dealloc [[RES_]] : memref<?xi8>
// CHECK:           return [[RES_3_]] : memref<?x10xf32>
// CHECK:         }
}
