// RUN: onnx-mlir-opt -O3 --process-krnl-parallel-clause %s -split-input-file | FileCheck %s

// -----

func.func @omp_threads_affinity(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_0[%c0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_0 : memref<1xindex>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_1[%c0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_1 : memref<1xindex>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_3[%c0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_3 : memref<1xindex>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg1) : index = (%c0) to (%c16384) step (%c32) {
        memref.alloca_scope  {
          %0 = vector.load %reshape[%arg1] : memref<16384xf32>, vector<32xf32>
          %1 = vector.load %reshape_2[%arg1] : memref<16384xf32>, vector<32xf32>
          %2 = arith.addf %0, %1 : vector<32xf32>
          vector.store %2, %reshape_4[%arg1] : memref<16384xf32>, vector<32xf32>
          krnl.parallel_clause(%arg1), num_threads(%c8_i32) {proc_bind = "spread"} : index
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return %alloc : memref<16x8x128xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @omp_threads_affinity
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
// CHECK:           [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK:           omp.parallel num_threads([[CST_8_]] : i32) proc_bind(spread) {
}
// -----
func.func @omp_threads(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_0[%c0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_0 : memref<1xindex>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_1[%c0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_1 : memref<1xindex>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_3[%c0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_3 : memref<1xindex>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg1) : index = (%c0) to (%c16384) step (%c32) {
        memref.alloca_scope  {
          %0 = vector.load %reshape[%arg1] : memref<16384xf32>, vector<32xf32>
          %1 = vector.load %reshape_2[%arg1] : memref<16384xf32>, vector<32xf32>
          %2 = arith.addf %0, %1 : vector<32xf32>
          vector.store %2, %reshape_4[%arg1] : memref<16384xf32>, vector<32xf32>
          krnl.parallel_clause(%arg1), num_threads(%c8_i32) : index
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return %alloc : memref<16x8x128xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @omp_threads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
// CHECK:           [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK:           omp.parallel num_threads([[CST_8_]] : i32) {
}

// -----

func.func @omp_affinity(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_0[%c0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_0 : memref<1xindex>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_1[%c0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_1 : memref<1xindex>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_3[%c0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_3 : memref<1xindex>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg1) : index = (%c0) to (%c16384) step (%c32) {
        memref.alloca_scope  {
          %0 = vector.load %reshape[%arg1] : memref<16384xf32>, vector<32xf32>
          %1 = vector.load %reshape_2[%arg1] : memref<16384xf32>, vector<32xf32>
          %2 = arith.addf %0, %1 : vector<32xf32>
          vector.store %2, %reshape_4[%arg1] : memref<16384xf32>, vector<32xf32>
          krnl.parallel_clause(%arg1) {proc_bind = "spread"} : index
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return %alloc : memref<16x8x128xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @omp_affinity
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
// CHECK:           omp.parallel proc_bind(spread) {
}

// -----

func.func @omp_normal(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_0[%c0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_0 : memref<1xindex>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_1[%c0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_1 : memref<1xindex>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  memref.store %c16384, %alloc_3[%c0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  memref.dealloc %alloc_3 : memref<1xindex>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%arg1) : index = (%c0) to (%c16384) step (%c32) {
        memref.alloca_scope  {
          %0 = vector.load %reshape[%arg1] : memref<16384xf32>, vector<32xf32>
          %1 = vector.load %reshape_2[%arg1] : memref<16384xf32>, vector<32xf32>
          %2 = arith.addf %0, %1 : vector<32xf32>
          vector.store %2, %reshape_4[%arg1] : memref<16384xf32>, vector<32xf32>
        }
        omp.yield
      }
    }
    omp.terminator
  }
  return %alloc : memref<16x8x128xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @omp_normal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
// CHECK:           omp.parallel {
}