// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

// -----

func.func @parallel_threads_affinity(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_0[0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_1[0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_3[0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %0 = krnl.define_loops 1
  %loop_block, %loop_local = krnl.block %0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.parallel(%loop_block), num_threads(%c8_i32) {proc_bind = "spread"} : !krnl.loop
  krnl.iterate(%loop_block) with (%0 -> %arg1 = 0 to 16384){
    %1 = krnl.get_induction_var_value(%loop_block) : (!krnl.loop) -> index
    %2 = vector.load %reshape[%1] : memref<16384xf32>, vector<32xf32>
    %3 = vector.load %reshape_2[%1] : memref<16384xf32>, vector<32xf32>
    %4 = arith.addf %2, %3 : vector<32xf32>
    vector.store %4, %reshape_4[%1] : memref<16384xf32>, vector<32xf32>
  }
  return %alloc : memref<16x8x128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @parallel_threads_affinity
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK:           [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK:           affine.parallel ([[arg1_:%.+]]) = (0) to (16384) step (32) {
// CHECK:             krnl.parallel_clause([[arg1_]]), num_threads([[CST_8_]]) {proc_bind = "spread"} : index
// CHECK:           }
// CHECK:         }
}

// -----

func.func @parallel_threads(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_0[0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_1[0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_3[0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %0 = krnl.define_loops 1
  %loop_block, %loop_local = krnl.block %0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.parallel(%loop_block), num_threads(%c8_i32) : !krnl.loop
  krnl.iterate(%loop_block) with (%0 -> %arg1 = 0 to 16384){
    %1 = krnl.get_induction_var_value(%loop_block) : (!krnl.loop) -> index
    %2 = vector.load %reshape[%1] : memref<16384xf32>, vector<32xf32>
    %3 = vector.load %reshape_2[%1] : memref<16384xf32>, vector<32xf32>
    %4 = arith.addf %2, %3 : vector<32xf32>
    vector.store %4, %reshape_4[%1] : memref<16384xf32>, vector<32xf32>
  }
  return %alloc : memref<16x8x128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @parallel_threads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK:           [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK:           affine.parallel ([[arg1_:%.+]]) = (0) to (16384) step (32) {
// CHECK:             krnl.parallel_clause([[arg1_]]), num_threads([[CST_8_]]) : index
// CHECK:           }
// CHECK:         }
}

// -----

func.func @parallel_affinity(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) {
  %c8_i32 = arith.constant 8 : i32
  %c16384 = arith.constant 16384 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_0[0] : memref<1xindex>
  %reshape = memref.reshape %arg0(%alloc_0) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_1[0] : memref<1xindex>
  %reshape_2 = memref.reshape %arg0(%alloc_1) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
  affine.store %c16384, %alloc_3[0] : memref<1xindex>
  %reshape_4 = memref.reshape %alloc(%alloc_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
  %0 = krnl.define_loops 1
  %loop_block, %loop_local = krnl.block %0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.parallel(%loop_block) {proc_bind = "spread"} : !krnl.loop
  krnl.iterate(%loop_block) with (%0 -> %arg1 = 0 to 16384){
    %1 = krnl.get_induction_var_value(%loop_block) : (!krnl.loop) -> index
    %2 = vector.load %reshape[%1] : memref<16384xf32>, vector<32xf32>
    %3 = vector.load %reshape_2[%1] : memref<16384xf32>, vector<32xf32>
    %4 = arith.addf %2, %3 : vector<32xf32>
    vector.store %4, %reshape_4[%1] : memref<16384xf32>, vector<32xf32>
  }
  return %alloc : memref<16x8x128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @parallel_affinity
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK:           affine.parallel ([[arg1_:%.+]]) = (0) to (16384) step (32) {
// CHECK:             krnl.parallel_clause([[arg1_]]) {proc_bind = "spread"} : index
// CHECK:           }
// CHECK:         }
}
