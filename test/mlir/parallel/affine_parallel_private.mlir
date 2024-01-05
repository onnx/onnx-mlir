// RUN: onnx-mlir-opt -O3 --march=x86-64 --affine-parallel-private  %s -split-input-file | FileCheck %s

// -----


func.func @add_with_par(%arg0: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
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
  affine.parallel (%arg1) = (0) to (16384) step (32) {
    %0 = vector.load %reshape[%arg1] : memref<16384xf32>, vector<32xf32>
    %1 = vector.load %reshape_2[%arg1] : memref<16384xf32>, vector<32xf32>
    %2 = arith.addf %0, %1 : vector<32xf32>
    vector.store %2, %reshape_4[%arg1] : memref<16384xf32>, vector<32xf32>
  }
  return %alloc : memref<16x8x128xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_with_par
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32> {onnx.name = "x"}) -> (memref<16x8x128xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_16384_:%.+]] = arith.constant 16384 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x8x128xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_16384_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_16384_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_2_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_2_]]) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_16384_]], [[RES_3_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_]]([[RES_]]_3) : (memref<16x8x128xf32>, memref<1xindex>) -> memref<16384xf32>
// CHECK:           affine.parallel ([[arg1_:%.+]]) = (0) to (16384) step (32) {
// CHECK:             memref.alloca_scope  {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[arg1_]]{{.}} : memref<16384xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_2_MEM_:%.+]] = vector.load [[VAR_reshape_2_]]{{.}}[[arg1_]]{{.}} : memref<16384xf32>, vector<32xf32>
// CHECK:               [[VAR_2_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_2_MEM_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_2_]], [[VAR_reshape_4_]]{{.}}[[arg1_]]{{.}} : memref<16384xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8x128xf32>
// CHECK:         }
}
