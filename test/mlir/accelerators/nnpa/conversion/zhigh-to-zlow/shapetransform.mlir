// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @transpose(%arg0: tensor<3x5xf32>) -> tensor<5x3xf32> {
    %0 = "zhigh.ShapeTransform"(%arg0) {index_map = #map} : (tensor<3x5xf32>) -> tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }

// CHECK-LABEL:  func.func @transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5xf32>) -> memref<5x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<3x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#1, [[VAR_1_]]#0] : memref<5x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5x3xf32>
// CHECK:         }
}
