// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Check lowering of onnx layout transform op and its introduction of maps for the mapped data.
// onnx-mlir-opt bibi.mlir -convert-onnx-to-krnl -canonicalize -convert-krnl-to-affine --normalize-memrefs

module {
  func.func @test_onnx_layout_transform(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32> {
    %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
    %1 = "onnx.LayoutTransform"(%0) : (tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x3x32x32xf32>
    return %1 : tensor<5x3x32x32xf32>
  }

// mlir2FileCheck.py -a '["input"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 4, d2, d3, d1 mod 4)>
// CHECK-LABEL:  func.func @test_onnx_layout_transform
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<5x3x32x32xf32>) -> memref<5x3x32x32xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 5, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<5x3x32x32xf32>
// CHECK:         }
}
