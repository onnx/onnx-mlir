// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Test multiple output
func.func @test_cumstom_multiple_output(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
   %cst = onnx.Constant dense<[
        [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0],
        [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]
      ]> : tensor<4x5xf32>
  %0,%1 = "onnx.Custom"(%cst) {function_name = "Decompose", r_value = 2 : si64, shape_infer_pattern = "SameAs"} : (tensor<4x5xf32>) -> (tensor<4x2xf32>, tensor<2x5xf32>)
  %2 = "onnx.Add"(%arg0, %0) : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}
// CHECK-LABEL:  func.func @test_cumstom_multiple_output
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x2xf32>) -> memref<4x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 5], value = dense<{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]{{.}}> : tensor<4x5xf32>} : () -> memref<4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x2xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x5xf32>
// CHECK:           "krnl.call"([[RES_]], [[RES_]]_0, [[VAR_0_]]) {funcName = "Decompose", numOfOutput = 2 : si64, r_value = 2 : si64} : (memref<4x2xf32>, memref<2x5xf32>, memref<4x5xf32>) -> ()
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_RES_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x2xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<4x2xf32>
// CHECK:         }

// -----

// Test attributes
func.func @test_custom3(%arg0: tensor<1024xi32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {function_name = "testcall", inputs_for_infer = [1], shape_infer_pattern = "SameAs"} : (tensor<1024xi32>, tensor<4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @test_custom3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xi32>, [[PARAM_1_:%.+]]: memref<4xf32>) -> memref<4xf32> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[PARAM_1_]]) {funcName = "testcall", numOfOutput = 1 : si64} : (memref<4xf32>, memref<1024xi32>, memref<4xf32>) -> ()
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }

// -----

// Test dynamic dim
func.func @test_custom_dynamic1(%arg0: tensor<1024xi32>, %arg1: tensor<?x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {function_name = "testcall", inputs_for_infer = [1], shape_infer_pattern = "SameAs"} : (tensor<1024xi32>, tensor<?x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @test_custom_dynamic1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xi32>, [[PARAM_1_:%.+]]: memref<?x4xf32>) -> memref<?x4xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x4xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x4xf32>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[PARAM_1_]]) {funcName = "testcall", numOfOutput = 1 : si64} : (memref<?x4xf32>, memref<1024xi32>, memref<?x4xf32>) -> ()
// CHECK:           return [[RES_]] : memref<?x4xf32>
// CHECK:         }
